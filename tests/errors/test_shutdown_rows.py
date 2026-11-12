"""Shutdown rows of the phase table: finalized (``OK``, 1001) or aborted.

Core's ``drain_endpoint_plugins`` sets ``state.draining`` and calls the
plugin's ``drain(state, timeout)``; the tests do the same with the plugin's
``FakeClock``. With a 20 s timeout, ``D`` is 15 s away (``engine_reserve`` is
5 s) and the settle point 11.25 s (``D - min(5 s, 15 s / 4)``), where the
trigger is raised. Each stream sends audio first, so the first-audio bound
(10 s) is disarmed and the idle bound (45 s) is far away.
"""

from __future__ import annotations

import asyncio

import pytest
from support.fake_session import Block, Raise, failed

SURFACES = ("grpc", "realtime")
TIMEOUT = 20.0
SETTLE = 11.25
DEADLINE = 15.0


@pytest.fixture(params=SURFACES)
def surface(request) -> str:
    return request.param


def _lease(fake_factory):
    assert fake_factory.leases, "open was not called"
    return fake_factory.leases[-1]


async def _drain(app_state, plugin) -> asyncio.Task[None]:
    from support.app import drain_plugin

    task = asyncio.create_task(drain_plugin(plugin, app_state, TIMEOUT))
    await asyncio.sleep(0)
    return task


async def _finish_drain(task, clock) -> None:
    for _ in range(20):
        if task.done():
            break
        await clock.advance(1)
    await asyncio.wait_for(asyncio.shield(task), 10)


# @spec ING-ERR-027, ING-ERR-011
async def test_shutdown_finalization_ends_ok_after_the_final(
    surface, started, app_state, stream, clock, fake_factory, helpers
) -> None:
    fake_factory.final_text = "hello world"
    drv = stream(surface, started)
    await drv.configure()
    await drv.send_audio(helpers.speech_bytes())
    lease = _lease(fake_factory)
    await helpers.until(lambda: "feed" in lease.names())
    drain = await _drain(app_state, started)
    await clock.advance(SETTLE + 0.5)
    outcome = await drv.outcome()
    assert lease.names()[-3:] == ["flush", "finish", "release"]
    assert len(outcome.finals) == 1
    if surface == "grpc":
        assert outcome.status == "OK"
        final = outcome.finals[0].results[0]
        assert final.alternatives[0].transcript == "hello world"
        assert outcome.responses[-1] is outcome.finals[0]
    else:
        assert outcome.close == 1001
        assert outcome.events[-1]["type"] == helpers.COMPLETED_EVENT
        assert outcome.terminal is None
    await _finish_drain(drain, clock)


# @spec ING-ERR-062, ING-ERR-011
async def test_failure_after_the_trigger_is_the_shutdown_abort_row(
    surface, started, app_state, stream, clock, fake_factory, helpers
) -> None:
    drv = stream(surface, started)
    await drv.configure()
    await drv.send_audio(helpers.speech_bytes())
    lease = _lease(fake_factory)
    lease.script("flush", Raise(RuntimeError("flush broke")))
    drain = await _drain(app_state, started)
    await clock.advance(SETTLE + 0.5)
    outcome = await drv.outcome()
    # not the ``internal`` row the flush failure would select otherwise
    helpers.assert_row(
        outcome, "service_unavailable", "UNAVAILABLE", "error", 1012
    )
    assert outcome.finals == []
    assert lease.names()[-2:] == ["abort", "release"]
    await _finish_drain(drain, clock)


# @spec ING-ERR-062
async def test_engine_dead_after_the_trigger_is_the_shutdown_abort_row(
    surface, started, app_state, stream, clock, fake_factory, helpers
) -> None:
    drv = stream(surface, started)
    await drv.configure()
    await drv.send_audio(helpers.speech_bytes())
    _lease(fake_factory).script("flush", Raise(failed("engine_dead")))
    drain = await _drain(app_state, started)
    await clock.advance(SETTLE + 0.5)
    outcome = await drv.outcome()
    helpers.assert_row(
        outcome, "service_unavailable", "UNAVAILABLE", "error", 1012
    )
    await _finish_drain(drain, clock)


# @spec ING-ERR-062, ING-ERR-056
async def test_drain_deadline_before_shutdown_finalization_is_the_abort_row(
    surface, started, app_state, stream, clock, fake_factory, helpers
) -> None:
    drv = stream(surface, started)
    await drv.configure()
    await drv.send_audio(helpers.speech_bytes())
    lease = _lease(fake_factory)
    lease.script("flush", Block())
    drain = await _drain(app_state, started)
    await clock.advance(SETTLE + 0.5)
    await helpers.until(lambda: "flush" in lease.names())
    # the shutdown finalization's bounds are later than D: the wait is
    # clipped to D and expires as D, not as finalization_timeout
    await clock.advance(DEADLINE - SETTLE)
    outcome = await drv.outcome()
    helpers.assert_row(
        outcome, "service_unavailable", "UNAVAILABLE", "error", 1012
    )
    assert outcome.code != "finalization_timeout"
    assert "finish" not in lease.names()
    await _finish_drain(drain, clock)


# @spec ING-ERR-062
async def test_drain_deadline_during_a_running_client_finalization(
    surface, started, app_state, stream, clock, fake_factory, helpers
) -> None:
    drv = stream(surface, started)
    await drv.configure()
    await drv.send_audio(helpers.speech_bytes())
    lease = _lease(fake_factory)
    lease.script("flush", Block())
    drain = await _drain(app_state, started)
    await clock.advance(SETTLE - 6)
    await drv.finalize()  # client-selected finalization starts first
    await helpers.until(lambda: "flush" in lease.names())
    await clock.advance(6.5)  # the trigger finds it running
    await clock.advance(DEADLINE - SETTLE)  # D before its own 30 s bound
    outcome = await drv.outcome()
    helpers.assert_row(
        outcome, "service_unavailable", "UNAVAILABLE", "error", 1012
    )
    await _finish_drain(drain, clock)


# @spec ING-ERR-062, ING-ERR-005
async def test_cause_selected_before_the_trigger_is_kept(
    started, app_state, stream, clock, fake_factory, helpers
) -> None:
    drv = stream("grpc", started)
    await drv.configure()
    lease = _lease(fake_factory)
    lease.script("abort", Block())  # idle-timeout cleanup still running
    drain = await _drain(app_state, started)
    await clock.advance(10.5)  # first-audio bound: idle_timeout selected
    await helpers.until(lambda: "abort" in lease.names())
    await clock.advance(DEADLINE - 10.5 + 1)
    outcome = await drv.outcome()
    assert outcome.code == "idle_timeout"
    assert outcome.status == "ABORTED"
    await _finish_drain(drain, clock)


# @spec ING-ERR-062
async def test_listener_stop_termination_before_the_status_send_completes(
    started, app_state, stream, clock, fake_factory, helpers
) -> None:
    """The listener stop ends the RPC while its ``UNAVAILABLE`` is in flight.

    The status reaches the client only when its send completes before the
    termination; the status the termination itself gives is the framework's
    and carries no catalog code (ING-ERR-061), so, as ING-LIFE-064 directs,
    the test asserts the selected row, that its send got one attempt and the
    cleanup, not the status the client receives.
    """
    from support.grpc_harness import RpcAbort

    drv = stream("grpc", started)
    await drv.configure()
    await drv.send_audio(helpers.speech_bytes())
    lease = _lease(fake_factory)
    lease.script("flush", Raise(RuntimeError("flush broke")))
    drv.context.block = True  # the client stops taking frames
    drain = await _drain(app_state, started)
    await clock.advance(SETTLE + 0.5)  # the settle point: the trigger
    await helpers.until(
        lambda: drv.context.parked, what="the terminal status send"
    )
    (attempt,) = drv.context.parked
    # the selected row is the shutdown abort row
    assert isinstance(attempt, RpcAbort), attempt
    assert attempt.code.name == "UNAVAILABLE"
    assert attempt.details.startswith("service_unavailable: ")
    # the listener stop has begun (the settle-point steps are done); its
    # termination of the RPC arrives before the send completes
    await clock.settle(20)
    drv.call.cancel()
    await _finish_drain(drain, clock)
    await helpers.until(lambda: drv.call.done, what="the handler's end")
    assert drv.context.stalled == 1, "the status send was retried"
    assert drv.context.aborts == [], "a status completed after termination"
    assert drv.call.finals == []
    assert lease.names()[-2:] == ["abort", "release"]


class _ListenerStopHook:
    """Stands in for the plugin's gRPC server and observes its stop.

    Every attribute is the server's, except ``stop``: it runs ``on_stop()``
    first (the moment the plugin began stopping the listener), then the
    server's own ``stop``.
    """

    def __init__(self, server, on_stop) -> None:
        self._server = server
        self._on_stop = on_stop
        self.stops: list[float | None] = []

    def __getattr__(self, name: str):
        return getattr(self._server, name)

    async def stop(self, grace: float | None) -> None:
        self.stops.append(grace)
        self._on_stop()
        await self._server.stop(grace)


def _owner(plugin):
    (record,) = plugin.registry.records.values()
    assert record.owner is not None, "the stream's owner is not attached"
    return record.owner


# @spec ING-LIFE-064, ING-ERR-062
async def test_listener_stop_termination_is_the_first_cause_of_a_stream(
    started, app_state, stream, clock, fake_factory, helpers
) -> None:
    """The listener stop's termination is the plugin's cancellation.

    The trigger finds the stream in shutdown finalization (provisional
    shutdown mode, no cause selected) with its ``flush`` still running.
    The plugin then begins its graceful listener stop, which terminates the
    RPC: the first cause the owner meets. It counts as the plugin's
    cancellation of the owner, so the owner takes the shutdown abort row and
    not a client disconnect. As ING-LIFE-064 directs, the test asserts the
    selected cause and the cleanup (on the fake context), not the status a
    client would receive.
    """
    drv = stream("grpc", started)
    await drv.configure()
    await drv.send_audio(helpers.speech_bytes())
    lease = _lease(fake_factory)
    await helpers.until(lambda: "feed" in lease.names())
    flush = Block()
    lease.script("flush", flush)
    owner = _owner(started)
    assert started.server is not None, "the plugin has no gRPC listener"
    # The cause the owner holds the moment the listener stop terminates the
    # RPC. It is read inside the hook, immediately before the termination,
    # and asserted below: an assertion raised in the hook would surface in
    # the plugin's stop, not here.
    causes_at_stop: list[object] = []

    def terminate() -> None:
        causes_at_stop.append(owner.cause)
        drv.call.cancel()

    hook = _ListenerStopHook(started.server, terminate)
    started.server = hook
    try:
        drain = await _drain(app_state, started)
        await clock.advance(SETTLE + 0.5)  # the settle point: the trigger
        await helpers.until(lambda: "flush" in lease.names(), what="flush")
        await helpers.until(lambda: hook.stops, what="the listener stop")
        # No cause was selected before the termination: a shutdown cause
        # chosen early (at the trigger or the listener stop's start) would
        # satisfy the assertions below without the termination counting.
        assert causes_at_stop == [None], causes_at_stop
        assert drv.context.cancelled(), "the listener stop ended the RPC"
        await helpers.until(lambda: drv.call.done, what="the handler's end")
        assert owner.cause is not None
        assert owner.cause.code == "service_unavailable", owner.cause
        assert lease.names()[-2:] == ["abort", "release"]
        assert "teardown" not in lease.names()
        await _finish_drain(drain, clock)
        assert not started.registry.records
    finally:
        flush.release()


# @spec ING-LIFE-064, ING-LIFE-038
async def test_client_cancellation_before_any_listener_stop_is_a_disconnect(
    started, stream, fake_factory, helpers
) -> None:
    """Control: a client cancellation with no listener stop is a disconnect.

    The same stream, its ``flush`` running, cancelled by the client while
    the plugin serves: the owner selects the transport failure of
    ING-LIFE-038, not the shutdown row, and cleans up the same way.
    """
    drv = stream("grpc", started)
    await drv.configure()
    await drv.send_audio(helpers.speech_bytes())
    lease = _lease(fake_factory)
    await helpers.until(lambda: "feed" in lease.names())
    flush = Block()
    lease.script("flush", flush)
    owner = _owner(started)
    try:
        await drv.finalize()
        await helpers.until(lambda: "flush" in lease.names(), what="flush")
        drv.call.cancel()
        await helpers.until(lambda: drv.call.done, what="the handler's end")
        assert owner.cause is not None
        assert owner.cause.phase == "transport", owner.cause
        assert owner.cause.code != "service_unavailable"
        assert lease.names()[-2:] == ["abort", "release"]
        assert not started.registry.records
    finally:
        flush.release()


# @spec ING-LIFE-064, ING-LIFE-038
async def test_client_cancellation_while_draining_before_listener_stop(
    started, app_state, stream, clock, fake_factory, helpers
) -> None:
    """Control: draining alone does not make a client cancellation shutdown.

    The same stream, its ``flush`` running, cancelled by the client after
    ``drain`` has begun (``state.draining`` is set) but before the settle
    point and before any listener stop. Only the listener stop's
    termination counts as the plugin's cancellation, so the owner selects
    the transport failure of ING-LIFE-038, not the shutdown row: an owner
    that keys the shutdown row on draining rather than on the listener stop
    having begun fails here.
    """
    drv = stream("grpc", started)
    await drv.configure()
    await drv.send_audio(helpers.speech_bytes())
    lease = _lease(fake_factory)
    await helpers.until(lambda: "feed" in lease.names())
    flush = Block()
    lease.script("flush", flush)
    owner = _owner(started)
    assert started.server is not None, "the plugin has no gRPC listener"
    hook = _ListenerStopHook(started.server, lambda: None)
    started.server = hook
    drain: asyncio.Task[None] | None = None
    try:
        await drv.finalize()
        await helpers.until(lambda: "flush" in lease.names(), what="flush")
        drain = await _drain(app_state, started)
        await clock.advance(1.0)  # draining, well before the settle point
        assert app_state.draining
        assert not drain.done()
        assert hook.stops == [], "the listener stop began before the settle"
        assert owner.cause is None, owner.cause
        drv.call.cancel()
        await helpers.until(lambda: drv.call.done, what="the handler's end")
        assert hook.stops == [], "the listener stop began before the settle"
        assert owner.cause is not None
        assert owner.cause.phase == "transport", owner.cause
        assert owner.cause.code != "service_unavailable"
        assert lease.names()[-2:] == ["abort", "release"]
        assert not started.registry.records
    finally:
        flush.release()
        if drain is not None:
            await _finish_drain(drain, clock)
