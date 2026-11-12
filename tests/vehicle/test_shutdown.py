"""``drain``, the settlement and ``stop``.

``drain`` is called directly on the plugin with the fake clock (core sets
``state.draining`` first, as ``drain_endpoint_plugins`` does), so its deadline
``D`` and settle point are virtual times the test reaches with
``FakeClock.advance``. Stream owners are real ``StreamOwner`` instances driven
through the servicer with ``RequestStream`` and ``FakeServicerContext``. The
fake factory models generation, cleanup and holding tasks, so the settlement
order (owners, listener, registry, collector, factory) is observable through
``fake_factory.calls`` and ``fake_factory.events``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
from typing import Any

import grpc
import pytest
from support.fake_session import Block
from support.riva import config_request

#: A stream idle since a 60 s drain began must still be open at its settle
#: point (49 s) and at ``D`` (54 s): ``session_idle_timeout`` 59 s outlasts
#: both and stays below the engine's 60 s parked-idle deadline (ING-EP-014).
OUTLASTS_DRAIN = pytest.mark.parametrize(
    "frontend_options", [{"session_idle_timeout": 59}], indirect=True
)


def spy_listener_stop(plugin: Any, monkeypatch, events: list, clock) -> None:
    """Record ``("listener.stop", grace, time)`` for each server stop."""
    real_stop = plugin.server.stop

    async def spy(grace):
        events.append(("listener.stop", grace, clock.monotonic()))
        return await real_stop(grace)

    monkeypatch.setattr(plugin.server, "stop", spy)


def spy_factory(factory: Any, monkeypatch, events: list, clock) -> None:
    """Record ``factory.drain`` and ``teardown_all`` calls in ``events``."""
    real_drain = factory.drain
    real_teardown = factory.teardown_all

    async def drain(deadline):
        events.append(("factory.drain", deadline, clock.monotonic()))
        return await real_drain(deadline)

    def teardown_all():
        events.append(("factory.teardown_all", None, clock.monotonic()))
        return real_teardown()

    monkeypatch.setattr(factory, "drain", drain)
    monkeypatch.setattr(factory, "teardown_all", teardown_all)


def watch_collector(plugin: Any, events: list, clock) -> None:
    """Record ``("collector.end", cancelled, time)`` when it ends."""
    plugin.fault_collector.add_done_callback(
        lambda task: events.append(
            ("collector.end", task.cancelled(), clock.monotonic())
        )
    )


def labels(events: list) -> list[str]:
    """The name of each event (a tuple's first item, or the string)."""
    return [e[0] if isinstance(e, tuple) else e for e in events]


def first(events: list, name: str, *, arg: Any = None) -> int:
    """Index of the first event called ``name`` (with ``arg``, if given)."""
    for index, event in enumerate(events):
        label = event[0] if isinstance(event, tuple) else event
        if label != name:
            continue
        if arg is not None and event[1] != arg:
            continue
        return index
    raise AssertionError(f"no {name} event ({arg=}) in {events}")


async def advance_until(clock, predicate, *, step=1.0, limit=120) -> None:
    """Advance ``clock`` by ``step`` until ``predicate()`` holds."""
    for _ in range(limit):
        if predicate():
            return
        await clock.advance(step)
    raise AssertionError("condition not reached")


# --------------------------------------------------------------- formulas


# @spec ING-EP-066
@pytest.mark.parametrize(
    "timeout, reserve", [(20, 5), (50, 5), (60, 6), (100, 10), (4, 5)]
)
def test_engine_reserve_is_the_larger_of_5s_and_10_percent(
    timeout, reserve
) -> None:
    from vllm_riva_frontend.lifecycle import engine_reserve

    assert engine_reserve(timeout) == pytest.approx(reserve)


# @spec ING-EP-036
@pytest.mark.parametrize(
    "now, deadline, settle", [(0, 54, 49), (0, 8, 6), (100, 120, 115)]
)
def test_settle_point_reserves_the_smaller_of_5s_and_a_quarter(
    now, deadline, settle
) -> None:
    from vllm_riva_frontend.lifecycle import settle_point

    assert settle_point(now, deadline) == pytest.approx(settle)


# --------------------------------------------------------------- drain


# @spec ING-EP-066, ING-EP-046
async def test_drain_computes_core_deadline_and_d_and_clips_graceful_waits(
    started, fake_factory, clock, helpers, monkeypatch
) -> None:
    plugin, state = started
    events: list[Any] = []
    spy_listener_stop(plugin, monkeypatch, events, clock)
    spy_factory(fake_factory, monkeypatch, events, clock)
    t0 = clock.monotonic()
    task = helpers.begin_drain(plugin, state, 60)
    await clock.settle()
    deadline = t0 + 54  # now + 60 - max(5, 6)
    assert plugin.drain_deadline == pytest.approx(deadline)
    await helpers.drive(clock, task)
    assert clock.monotonic() <= t0 + 60, "within core's deadline"
    stop = events[first(events, "listener.stop")]
    assert stop[1] == pytest.approx(deadline - stop[2]), "grace clipped to D"
    graceful = [
        e for e in events if e[0] == "factory.drain" and e[1] == deadline
    ]
    assert graceful, f"the factory wait is clipped to D: {events}"


# @spec ING-EP-066
async def test_drain_sets_not_serving_and_closes_admission(
    started, clock, helpers
) -> None:
    from grpc_health.v1 import health_pb2, health_pb2_grpc
    from support.grpc_harness import aio_channel

    from vllm_riva_frontend.grpc import RIVA_SERVICE_NAME

    plugin, state = started
    task = helpers.begin_drain(plugin, state, 60)
    await clock.settle()
    async with aio_channel(plugin.config.grpc_bind) as channel:
        stub = health_pb2_grpc.HealthStub(channel)
        for service in ("", RIVA_SERVICE_NAME):
            reply = await stub.Check(
                health_pb2.HealthCheckRequest(service=service), timeout=2.0
            )
            assert reply.status == health_pb2.HealthCheckResponse.NOT_SERVING
    assert not plugin.ingress.is_open
    refused = helpers.stream(plugin, plugin.clock, config_request())
    await asyncio.wait_for(refused.task, 2.0)
    assert refused.status is grpc.StatusCode.UNAVAILABLE
    assert refused.details.startswith("service_unavailable")
    await helpers.drive(clock, task)


# @spec ING-EP-109
async def test_listener_serves_not_serving_until_settle_point_without_owners(
    started, clock, helpers, monkeypatch
) -> None:
    from grpc_health.v1 import health_pb2, health_pb2_grpc
    from support.grpc_harness import aio_channel

    plugin, state = started
    events: list[Any] = []
    spy_listener_stop(plugin, monkeypatch, events, clock)
    t0 = clock.monotonic()
    task = helpers.begin_drain(plugin, state, 60)  # settle point t0 + 49
    await clock.advance(48.5)
    assert not task.done(), "drain waits for its settle point, owners or not"
    assert events == [], "the listener is not stopped before the settle point"
    assert state.request_shutdown.calls == 0
    assert plugin.registry.shutdown_trigger is None
    async with aio_channel(plugin.config.grpc_bind) as channel:
        reply = await health_pb2_grpc.HealthStub(channel).Check(
            health_pb2.HealthCheckRequest(service=""), timeout=2.0
        )
        assert reply.status == health_pb2.HealthCheckResponse.NOT_SERVING
    await helpers.drive(clock, task)
    assert events and events[0][2] >= t0 + 49


# @spec ING-EP-036
async def test_owners_finish_normally_before_the_settle_point(
    started, fake_factory, clock, helpers
) -> None:
    plugin, state = started
    run, lease = await helpers.stream_with_lease(plugin, fake_factory, clock)
    task = helpers.begin_drain(plugin, state, 60)  # D t0 + 54, S t0 + 49
    await clock.advance(40)
    assert "flush" not in lease.names(), "no shutdown before the settle point"
    # The client finishes on its own and gets its final.
    run.requests.close()
    await helpers.settle_until(clock, lambda: run.done)
    assert run.status is grpc.StatusCode.OK
    assert lease.names()[-3:] == ["flush", "finish", "release"]
    await helpers.drive(clock, task)


# @spec ING-EP-037
@OUTLASTS_DRAIN
async def test_settle_point_finalizes_remaining_streams(
    started, fake_factory, clock, helpers
) -> None:
    plugin, state = started
    run, lease = await helpers.stream_with_lease(plugin, fake_factory, clock)
    task = helpers.begin_drain(plugin, state, 60)
    await clock.advance(48.5)
    assert "flush" not in lease.names()
    await clock.advance(1.0)  # past the settle point at t0 + 49
    await helpers.settle_until(clock, lambda: run.done)
    assert run.status is grpc.StatusCode.OK, "gRPC OK after the final"
    assert lease.names()[-3:] == ["flush", "finish", "release"]
    await helpers.drive(clock, task)


# @spec ING-EP-074, ING-EP-100
@OUTLASTS_DRAIN
async def test_after_settle_listener_stop_then_collector_end_then_drain_d(
    started, fake_factory, clock, helpers, monkeypatch
) -> None:
    plugin, state = started
    run, _lease = await helpers.stream_with_lease(plugin, fake_factory, clock)
    events: list[Any] = []
    spy_listener_stop(plugin, monkeypatch, events, clock)
    spy_factory(fake_factory, monkeypatch, events, clock)
    watch_collector(plugin, events, clock)
    t0 = clock.monotonic()
    deadline = t0 + 54
    task = helpers.begin_drain(plugin, state, 60)
    await clock.advance(49.5)
    await helpers.settle_until(clock, lambda: run.done)
    await helpers.drive(clock, task)
    stop_at = first(events, "listener.stop")
    grace, at = events[stop_at][1], events[stop_at][2]
    assert grace == pytest.approx(deadline - at)
    collector_at = first(events, "collector.end")
    assert events[collector_at][1] is False, "the collector is not cancelled"
    drain_at = next(
        i
        for i, event in enumerate(events)
        if i > stop_at and event[0] == "factory.drain" and event[1] == deadline
    )
    assert stop_at < collector_at < drain_at, events


# @spec ING-EP-066, ING-EP-100, ING-EP-093
async def test_a_collector_end_blocked_past_d_does_not_hold_the_settlement(
    started, fake_factory, clock, helpers, monkeypatch
) -> None:
    """The graceful work's wait for the collector's end is bounded by ``D``.

    The collector does not take the graceful work's stop signal: at ``D``
    the drain stops waiting for it, skips ``factory.drain(D)`` and its
    settlement begins; the settlement's own signal then ends the collector,
    which is never cancelled.
    """
    plugin, state = started
    collector = plugin._collector
    real_request_stop = collector.request_stop
    signals: list[float] = []

    def request_stop() -> None:
        signals.append(clock.monotonic())
        if len(signals) > 1:
            real_request_stop()  # the settlement's signal is taken

    monkeypatch.setattr(collector, "request_stop", request_stop)
    events: list[Any] = []
    spy_factory(fake_factory, monkeypatch, events, clock)
    watch_collector(plugin, events, clock)
    t0 = clock.monotonic()
    deadline = t0 + 15  # timeout 20 less the 5 s reserve
    task = helpers.begin_drain(plugin, state, 20)
    await clock.advance(14.5)  # past the settle point at t0 + 11.25
    await helpers.settle_until(clock, lambda: len(signals) == 1)
    assert signals[0] < deadline
    assert not plugin.send_bounds_passed, "the settlement began before D"
    assert not plugin.fault_collector.done()
    assert not task.done()
    await clock.advance(1.0)  # past D
    await helpers.drive(clock, task)
    assert len(signals) == 2, "the settlement signals the collector again"
    assert signals[1] >= deadline, "the settlement began before D"
    collector_at = first(events, "collector.end")
    assert events[collector_at][1] is False, "the collector was cancelled"
    assert events[collector_at][2] >= deadline
    settle_at = first(events, "factory.drain", arg=math.inf)
    assert collector_at < settle_at, events
    assert not any(
        e[0] == "factory.drain" and e[1] == deadline for e in events
    ), "factory.drain(D) ran although the collector had not ended"
    helpers.factory_settlement(fake_factory)


# @spec ING-EP-066, ING-EP-093, ING-EP-110
async def test_a_factory_drain_blocked_past_d_does_not_hold_the_settlement(
    started, fake_factory, clock, helpers, monkeypatch
) -> None:
    """The factory wait of the graceful work is bounded by ``D``.

    ``factory.drain(D)`` does not return: at ``D`` the drain stops waiting
    for it (it is not cancelled) and its settlement begins, cancelling the
    owners; the settlement then still awaits the blocked call before the
    factory settlement runs.
    """
    plugin, state = started
    real_drain = fake_factory.drain
    entered, release = asyncio.Event(), asyncio.Event()
    cancelled: list[bool] = []
    t0 = clock.monotonic()
    deadline = t0 + 15  # timeout 20 less the 5 s reserve

    async def drain(at):
        if at == deadline:
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.append(True)
                raise
        return await real_drain(at)

    monkeypatch.setattr(fake_factory, "drain", drain)
    task = helpers.begin_drain(plugin, state, 20)
    await clock.advance(12)  # past the settle point at t0 + 11.25
    await helpers.settle_until(clock, entered.is_set)
    owner, _ = helpers.register_dummy(plugin, "file_route")
    await clock.advance(deadline - clock.monotonic() - 0.5)
    assert not plugin.send_bounds_passed, "the settlement began before D"
    assert not owner.done()
    await clock.advance(1.0)  # past D
    await helpers.settle_until(clock, lambda: plugin.send_bounds_passed)
    await helpers.settle_until(clock, owner.done)
    assert owner.cancelled(), "the settlement did not cancel the owners at D"
    assert not task.done(), "the settlement does not await the blocked call"
    assert inf_drains(fake_factory) == 0, "factory settled before the call"
    release.set()
    await helpers.drive(clock, task)
    assert cancelled == [], "the factory drain was cancelled"
    helpers.factory_settlement(fake_factory)


# @spec ING-EP-076, ING-EP-093, ING-EP-066
@pytest.mark.parametrize("late", ["raises", "cancelled"])
async def test_a_factory_drain_failing_after_d_settles_then_propagates(
    started, fake_factory, clock, helpers, monkeypatch, late
) -> None:
    """A late ``factory.drain(D)`` outcome is kept until settlement ends.

    The call is still running at ``D``, so the settlement begins and awaits
    it. It then raises, or ends cancelled: the settlement still runs the
    factory settlement, and only then does ``drain`` propagate that outcome
    (ING-EP-076).
    """
    plugin, state = started
    real_drain = fake_factory.drain
    entered, release = asyncio.Event(), asyncio.Event()
    deadline = clock.monotonic() + 15  # timeout 20 less the 5 s reserve

    async def drain(at):
        if at == deadline:
            entered.set()
            await release.wait()
            if late == "raises":
                raise RuntimeError("factory.drain(D) failed late")
            raise asyncio.CancelledError
        return await real_drain(at)

    monkeypatch.setattr(fake_factory, "drain", drain)
    task = helpers.begin_drain(plugin, state, 20)
    await clock.advance(12)  # past the settle point at t0 + 11.25
    await helpers.settle_until(clock, entered.is_set)
    await clock.advance(deadline - clock.monotonic() + 0.5)  # past D
    await helpers.settle_until(clock, lambda: plugin.send_bounds_passed)
    assert not task.done(), "the settlement awaits the late call"
    assert inf_drains(fake_factory) == 0, "factory settled before the call"
    release.set()
    expected = RuntimeError if late == "raises" else asyncio.CancelledError
    with pytest.raises(expected):
        await helpers.drive(clock, task)
    helpers.factory_settlement(fake_factory)
    assert not plugin.registry.records


# @spec ING-EP-076, ING-EP-093, ING-EP-066
async def test_a_late_factory_drain_failure_is_logged_when_drain_is_cancelled(
    started, fake_factory, clock, helpers, monkeypatch, caplog
) -> None:
    """``drain`` cancelled during the settlement re-raises its cancellation.

    The ``factory.drain(D)`` call is still running at ``D``; ``drain`` is
    cancelled while the settlement awaits it, and the call then raises. The
    settlement still runs the factory settlement, ``drain`` raises
    ``CancelledError``, and the late failure is logged once (ING-EP-076).
    """
    caplog.set_level(logging.ERROR)
    plugin, state = started
    real_drain = fake_factory.drain
    entered, release = asyncio.Event(), asyncio.Event()
    deadline = clock.monotonic() + 15  # timeout 20 less the 5 s reserve

    async def drain(at):
        if at == deadline:
            entered.set()
            await release.wait()
            raise RuntimeError("factory.drain(D) failed late")
        return await real_drain(at)

    monkeypatch.setattr(fake_factory, "drain", drain)
    task = helpers.begin_drain(plugin, state, 20)
    await clock.advance(12)  # past the settle point at t0 + 11.25
    await helpers.settle_until(clock, entered.is_set)
    await clock.advance(deadline - clock.monotonic() + 0.5)  # past D
    await helpers.settle_until(clock, lambda: plugin.send_bounds_passed)
    assert not task.done(), "the settlement awaits the late call"
    task.cancel()
    await clock.settle(5)
    assert not task.done(), "the cancellation interrupted the settlement"
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await helpers.drive(clock, task)
    helpers.factory_settlement(fake_factory)
    assert not plugin.registry.records
    reports = [
        record
        for record in caplog.records
        if "factory.drain(D) failed" in record.getMessage()
    ]
    assert len(reports) == 1, [r.getMessage() for r in caplog.records]
    assert reports[0].exc_info is not None
    assert isinstance(reports[0].exc_info[1], RuntimeError)


def inf_drains(factory: Any) -> int:
    """How many ``drain(math.inf)`` calls the factory has had."""
    return sum(1 for at in factory.drains if at == math.inf)


# @spec ING-EP-037, ING-EP-066
@OUTLASTS_DRAIN
async def test_shutdown_finalization_failing_by_d_aborts_with_unavailable(
    started, fake_factory, clock, helpers
) -> None:
    plugin, state = started
    run, lease = await helpers.stream_with_lease(plugin, fake_factory, clock)
    lease.script("flush", Block())
    t0 = clock.monotonic()
    task = helpers.begin_drain(plugin, state, 60)
    await clock.advance(49.5)
    await helpers.settle_until(clock, lambda: "flush" in lease.names())
    assert not run.done
    await clock.advance(54 - 49.5 + 0.5)  # past D
    await helpers.settle_until(clock, lambda: run.done)
    assert run.status is grpc.StatusCode.UNAVAILABLE
    assert run.details.startswith("service_unavailable")
    assert lease.names()[-2:] == ["abort", "release"]
    await helpers.drive(clock, task)
    assert clock.monotonic() <= t0 + 60, "drain finished within its budget"


# @spec ING-EP-073
async def test_settle_point_cancels_file_requests_and_bounds_their_wait(
    started, test_app, fake_transcription, clock, helpers
) -> None:
    from support.asgi import asgi_http
    from support.fake_session import Block as CallBlock

    from vllm_riva_frontend.admission import OwnerKind
    from vllm_riva_frontend.errors import catalog

    plugin, state = started
    call = CallBlock()
    fake_transcription.script(call)
    body, headers = helpers.file_request_body()
    request = asyncio.ensure_future(
        asgi_http(
            test_app,
            "POST",
            helpers.file_route(),
            headers=headers,
            body=body,
        )
    )
    await helpers.settle_until(clock, lambda: fake_transcription.calls != [])
    t0 = clock.monotonic()
    task = helpers.begin_drain(plugin, state, 60)  # S t0 + 49, D t0 + 54
    await clock.advance(49.5)
    response = await asyncio.wait_for(request, 5.0)
    assert response.status == catalog().http_status("service_unavailable")
    # Core's call has not settled: the request's record stays under its
    # running cleanup continuation, with its accounting.
    records = list(plugin.registry.records.values())
    assert [record.kind for record in records] == [OwnerKind.file_route]
    continuation = records[0].continuation
    assert continuation is not None and not continuation.done()
    assert plugin.ingress.file_count == 1
    # The wait for it is bounded by D: drain goes on to its later steps and
    # its settlement, which awaits the continuation with no time limit.
    await clock.advance(t0 + 54 - clock.monotonic() + 1)
    assert not task.done(), "the settlement awaits the continuation"
    assert not continuation.cancelled()
    call.release()
    await helpers.drive(clock, task)
    assert continuation.done() and not continuation.cancelled()
    assert not plugin.registry.records
    assert plugin.ingress.file_count == 0


# @spec ING-EP-075
async def test_timeout_within_reserve_skips_graceful_steps_and_settles(
    started, fake_factory, clock, helpers, monkeypatch
) -> None:
    plugin, state = started
    run, lease = await helpers.stream_with_lease(plugin, fake_factory, clock)
    dummy, _ = helpers.register_dummy(plugin)
    events: list[Any] = []
    spy_listener_stop(plugin, monkeypatch, events, clock)
    before = len(fake_factory.calls)
    t0 = clock.monotonic()
    task = helpers.begin_drain(plugin, state, 4)  # not above the 5 s reserve
    await helpers.settle_until(clock, lambda: run.done and dummy.done())
    assert clock.monotonic() == t0, "no graceful phase was waited"
    assert "flush" not in lease.names()
    assert lease.names()[-2:] == ["abort", "release"]
    assert run.status is grpc.StatusCode.UNAVAILABLE
    assert dummy.cancelled()
    await helpers.drive(clock, task)
    assert plugin.registry.shutdown_trigger is None, "no shutdown trigger"
    assert all(grace in (None, 0) for _, grace, _ in events), (
        "no graceful listener stop"
    )
    helpers.factory_settlement(fake_factory, since=before)
    finite = [d for d in fake_factory.drains if math.isfinite(d)]
    assert all(d <= t0 + 4 for d in finite[-1:])


# @spec ING-EP-076, ING-EP-093
@pytest.mark.parametrize("ending", ["cancelled", "step_raises"])
async def test_cancelled_or_failing_drain_completes_settlement_then_reraises(
    started, fake_factory, clock, helpers, monkeypatch, ending
) -> None:
    plugin, state = started
    release_owner = asyncio.Event()
    dummy, _ = helpers.register_dummy(plugin, stubborn=release_owner)
    if ending == "step_raises":
        real_stop = plugin.server.stop
        calls = {"n": 0}

        async def failing_stop(grace):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("listener stop failed")
            return await real_stop(grace)

        monkeypatch.setattr(plugin.server, "stop", failing_stop)
    task = helpers.begin_drain(plugin, state, 20)  # D t0 + 15, S t0 + 11.25
    if ending == "cancelled":
        await clock.advance(5)
        for _ in range(3):
            task.cancel()
            await clock.settle()
    else:
        # Past D: the graceful listener stop raises, the settlement begins.
        await clock.advance(16)
    await clock.settle()
    assert not task.done(), "the settlement awaits the owner's cleanup"
    assert dummy.cancelling() or dummy.cancelled(), "the owner is aborted"
    assert fake_factory.teardown_calls == 0, "owners settle first"
    if ending == "cancelled":
        task.cancel()  # one more, while the settlement waits
        await clock.settle()
        assert not task.done()
    release_owner.set()
    expected = asyncio.CancelledError if ending == "cancelled" else RuntimeError
    with pytest.raises(expected):
        await helpers.drive(clock, task)
    assert dummy.done()
    helpers.factory_settlement(fake_factory)
    assert not plugin.registry.records


# @spec ING-EP-076
async def test_cancellation_during_factory_settlement_is_deferred(
    started, fake_factory, clock, helpers
) -> None:
    plugin, state = started
    task = helpers.begin_drain(plugin, state, 4)  # straight to settlement
    block = Block()
    fake_factory.script("drain", block)
    await helpers.settle_until(clock, lambda: block.entered.is_set())
    for _ in range(3):
        task.cancel()
        await clock.settle()
    assert not task.done(), "the factory settlement is not interrupted"
    assert fake_factory.teardown_calls == 0
    block.release()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5.0)
    helpers.factory_settlement(fake_factory)


# @spec ING-EP-102
async def test_core_draining_without_drain_refuses_and_stop_settles_the_rest(
    started, fake_factory, clock, helpers
) -> None:
    from grpc_health.v1 import health_pb2, health_pb2_grpc
    from support.grpc_harness import aio_channel

    from vllm_riva_frontend.grpc import RIVA_SERVICE_NAME

    plugin, state = started
    run, lease = await helpers.stream_with_lease(plugin, fake_factory, clock)
    state.draining = True  # zero budget: core calls no drain
    refused = helpers.stream(plugin, clock, config_request())
    await asyncio.wait_for(refused.task, 2.0)
    assert refused.status is grpc.StatusCode.UNAVAILABLE
    assert plugin.ready(state) is False
    async with aio_channel(plugin.config.grpc_bind) as channel:
        stub = health_pb2_grpc.HealthStub(channel)
        for service in ("", RIVA_SERVICE_NAME):
            reply = await stub.Check(
                health_pb2.HealthCheckRequest(service=service), timeout=2.0
            )
            assert reply.status == health_pb2.HealthCheckResponse.SERVING, (
                "health stays as start left it"
            )
    await clock.advance(30)
    assert plugin.drain_deadline is None
    assert not run.done and "abort" not in lease.names(), "nothing clipped"
    before = len(fake_factory.calls)
    await helpers.drive(clock, plugin.stop(state))
    assert run.done
    assert lease.names()[-2:] == ["abort", "release"]
    assert lease.teardown_calls == 0
    helpers.factory_settlement(fake_factory, since=before)
    assert not plugin.registry.records


# ------------------------------------------------------------- settlement


# @spec ING-EP-093, ING-EP-100
@pytest.mark.parametrize("ending", ["drain", "stop"])
async def test_settlement_order_owners_listener_registry_collector_factory(
    started, fake_factory, clock, helpers, monkeypatch, ending
) -> None:
    plugin, state = started
    events: list[Any] = []
    release_owner = asyncio.Event()
    dummy, _ = helpers.register_dummy(
        plugin, stubborn=release_owner, events=events
    )
    spy_listener_stop(plugin, monkeypatch, events, clock)
    spy_factory(fake_factory, monkeypatch, events, clock)
    watch_collector(plugin, events, clock)
    if ending == "drain":
        task = helpers.begin_drain(plugin, state, 4)  # straight to settlement
    else:
        task = asyncio.ensure_future(plugin.stop(state))
    await helpers.settle_until(clock, lambda: "listener.stop" in labels(events))
    await clock.settle(20)
    assert "owner.end" not in labels(events)
    assert "collector.end" not in labels(events), (
        "the collector ends only after every owner"
    )
    assert not any(name.startswith("factory.") for name in labels(events))
    release_owner.set()
    await helpers.drive(clock, task)
    cancelled = first(events, "owner.cancelled")
    stopped = first(events, "listener.stop")
    ended = first(events, "owner.end")
    collector = first(events, "collector.end")
    assert events[collector][1] is False, "ended by its signal, not cancelled"
    settle = first(events, "factory.drain", arg=math.inf)
    teardown = first(events, "factory.teardown_all")
    assert cancelled < stopped < ended < collector < settle < teardown, events
    assert events[stopped][1] in (None, 0), "stopped without grace"
    assert dummy.cancelled()


# @spec ING-EP-110
@pytest.mark.parametrize("ending", ["skips_graceful", "cancelled_graceful"])
async def test_settlement_ends_clipping_and_makes_send_bounds_passed(
    started, fake_factory, clock, helpers, ending
) -> None:
    from support.grpc_harness import RpcAbort

    plugin, state = started
    run, lease = await helpers.stream_with_lease(plugin, fake_factory, clock)
    abort = Block()
    lease.script("abort", abort)
    run.context.block = True  # a stalled reader: the status gets one attempt
    cleanup = plugin.config.session_cleanup_timeout
    if ending == "skips_graceful":
        task = helpers.begin_drain(plugin, state, 4)  # settles at once
        began = clock.monotonic()
    else:
        task = helpers.begin_drain(plugin, state, 20)  # D t0 + 15
        await clock.advance(5)
        began = clock.monotonic()
        task.cancel()  # the settlement begins in the graceful phase
    await helpers.settle_until(clock, lambda: abort.entered.is_set())
    await clock.advance(began + 16 - clock.monotonic())  # past D, core's too
    assert lease.names()[-1] == "abort", (
        "the cleanup begun in the settlement runs under "
        "session_cleanup_timeout, not clipped to D"
    )
    await clock.advance(began + cleanup - clock.monotonic() + 1)
    await helpers.settle_until(clock, lambda: "release" in lease.names())
    if ending == "skips_graceful":
        await helpers.drive(clock, task)
    else:
        with pytest.raises(asyncio.CancelledError):
            await helpers.drive(clock, task)
    assert run.done
    response_send_timeout = plugin.config.response_send_timeout
    assert clock.monotonic() < began + cleanup + response_send_timeout, (
        "no send waited for its own bound"
    )
    assert run.context.stalled == 1, "one terminal attempt, nothing more"
    status = run.context.parked[0]
    assert isinstance(status, RpcAbort)
    assert status.code is grpc.StatusCode.UNAVAILABLE


# @spec ING-EP-110, ING-LIFE-061
@pytest.mark.parametrize("graceful", [False, True])
async def test_send_bounds_pass_only_for_the_rest_of_that_drain_call(
    started, clock, helpers, graceful
) -> None:
    """After ``drain`` returns, a send gets its ordinary bound again.

    The send bounds count as passed for the rest of the ``drain`` call
    whose settlement began (or that skipped its graceful steps); once that
    call has returned, a refusal send is bounded by
    ``response_send_timeout``, until ``stop`` makes every bound pass.
    """
    from vllm_riva_frontend import surfaces

    plugin, state = started
    timeout = 20 if graceful else 4  # 4 s is within the reserve
    await helpers.drive(clock, helpers.begin_drain(plugin, state, timeout))
    assert not plugin.send_bounds_passed, "the drain's latch outlived it"
    assert plugin.drain_deadline is None
    assert not plugin.drain_announced().done()
    stalled = asyncio.Event()

    async def send() -> None:
        await stalled.wait()

    sending = asyncio.ensure_future(
        surfaces.bounded_send(plugin, send(), terminal=True)
    )
    await clock.settle()
    bound = plugin.config.response_send_timeout
    await clock.advance(bound - 1)
    assert not sending.done(), "the send did not get its ordinary bound"
    await clock.advance(2)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(sending, 2)
    await helpers.drive(clock, plugin.stop(state))
    assert plugin.send_bounds_passed, "every bound passes from stop on"


# @spec ING-EP-110, ING-EP-079
@pytest.mark.parametrize("surface", ["realtime", "file_route"])
async def test_terminal_attempt_covers_whole_response(
    started, test_app, fake_factory, fake_transcription, clock, helpers, surface
) -> None:
    from support.asgi import asgi_http
    from support.fake_session import Block as CallBlock
    from support.realtime import CLOSE, ERROR, FAILED, RealtimeClient

    from vllm_riva_frontend.errors import catalog

    plugin, state = started
    call = CallBlock()
    if surface == "realtime":
        client = RealtimeClient(test_app, clock)
        # Each ASGI message the app sends, with the event-loop pass it left
        # in: one terminal attempt sends its messages with no wait between.
        passes = [0]
        sent: list[tuple[str, int]] = []
        inner_send = client.ws._send

        async def recording_send(message):
            sent.append((message["type"], passes[0]))
            await inner_send(message)

        async def count_passes():
            while True:
                passes[0] += 1
                await asyncio.sleep(0)

        client.ws._send = recording_send  # type: ignore[method-assign]
        ticker = asyncio.ensure_future(count_passes())
        try:
            await client.open_session(helpers.realtime_update()["session"])
            await helpers.drive(clock, plugin.stop(state))
            events = await client.outcome()
        finally:
            ticker.cancel()
        types = [event["type"] for event in events]
        assert types[-1] == CLOSE, types
        assert types[-2] in (ERROR, FAILED), (
            "the terminal event and its close are one attempt"
        )
        (_, error_pass), (close_type, close_pass) = sent[-2:]
        assert close_type == "websocket.close", sent
        assert close_pass == error_pass, (
            "the close waited for its own attempt after the terminal event"
        )
        return
    fake_transcription.script(call)
    body, headers = helpers.file_request_body()
    request = asyncio.ensure_future(
        asgi_http(
            test_app, "POST", helpers.file_route(), headers=headers, body=body
        )
    )
    await helpers.settle_until(clock, lambda: fake_transcription.calls != [])
    stopping = asyncio.ensure_future(plugin.stop(state))
    response = await asyncio.wait_for(request, 5.0)
    assert response.status == catalog().http_status("service_unavailable")
    assert response.completed, "start and body are one terminal attempt"
    call.release()
    await asyncio.wait_for(stopping, 5.0)


# @spec ING-EP-096, ING-EP-097
async def test_factory_settlement_steps(
    started, fake_factory, clock, helpers, caplog
) -> None:
    caplog.set_level(logging.INFO)
    plugin, state = started
    first_fault = RuntimeError("fault-from-first-drain")
    second_fault = RuntimeError("fault-from-second-drain")

    # A task the factory still runs at ``teardown_all``: a last resort.
    async def lingering() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            for _ in range(3):
                await asyncio.sleep(0)
            raise RuntimeError("lingering-cancel-failed") from None

    async def slow_to_end() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            for _ in range(5):
                await asyncio.sleep(0)

    raising = asyncio.ensure_future(lingering())
    slow = asyncio.ensure_future(slow_to_end())
    fake_factory.tasks |= {raising, slow}
    real_drain = fake_factory.drain
    returned = iter([[first_fault], [second_fault]])

    async def drain(deadline):
        faults = await real_drain(deadline)
        return [*faults, *next(returned, [])]

    fake_factory.drain = drain
    before = len(fake_factory.calls)
    await asyncio.wait_for(plugin.stop(state), 5.0)
    helpers.factory_settlement(fake_factory, since=before)
    assert raising.done() and slow.done(), "every returned task awaited"
    text = caplog.text
    assert "fault-from-first-drain" in text
    assert "fault-from-second-drain" in text
    for task in (raising, slow):
        assert any(
            repr(task) in record.getMessage() or task in _flatten(record.args)
            for record in caplog.records
        ), f"no last-resort fault keyed by {task!r}"
    assert state.request_shutdown.calls == 0, "nothing escalates in stop"
    assert plugin.fault is None
    with contextlib.suppress(BaseException):
        raising.result()


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [*value.keys(), *value.values()]
    if isinstance(value, tuple | list | set):
        out: list[Any] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [value]


# @spec ING-EP-097
@pytest.mark.parametrize("ending", ["drain", "stop"])
async def test_factory_settlement_runs_after_owners_and_collector(
    started, fake_factory, clock, helpers, monkeypatch, ending
) -> None:
    plugin, state = started
    release_owner = asyncio.Event()
    dummy, _ = helpers.register_dummy(plugin, stubborn=release_owner)
    snapshots: list[tuple[bool, bool]] = []
    real_drain = fake_factory.drain

    async def drain(deadline):
        if deadline == math.inf:
            snapshots.append(
                (
                    all(
                        record.task.done()
                        and (
                            record.continuation is None
                            or record.continuation.done()
                        )
                        for record in plugin.registry.records.values()
                    )
                    and dummy.done(),
                    plugin.fault_collector.done(),
                )
            )
        return await real_drain(deadline)

    monkeypatch.setattr(fake_factory, "drain", drain)
    if ending == "drain":
        task = helpers.begin_drain(plugin, state, 4)
    else:
        task = asyncio.ensure_future(plugin.stop(state))
    await clock.settle(20)
    release_owner.set()
    await helpers.drive(clock, task)
    helpers.factory_settlement(fake_factory)
    assert snapshots == [(True, True)], snapshots
    assert state.request_shutdown.calls == 0


# @spec ING-EP-098
@pytest.mark.parametrize("ending", ["drain", "stop"])
async def test_vehicle_cancels_no_factory_task(
    started, fake_factory, fake_seam, clock, helpers, ending
) -> None:
    plugin, state = started
    run, lease = await helpers.stream_with_lease(plugin, fake_factory, clock)
    engine_abort = fake_seam.slow_engine_abort(lease)
    if ending == "drain":
        task = helpers.begin_drain(plugin, state, 4)
    else:
        task = asyncio.ensure_future(plugin.stop(state))
    await advance_until(
        clock,
        lambda: task.done() or math.inf in fake_factory.drains,
        limit=int(plugin.config.session_cleanup_timeout) + 20,
    )
    engine_abort.release()
    await helpers.drive(clock, task)
    assert run.done
    assert fake_seam.foreign_cancellations() == []
    assert fake_seam.late_cancellations() == []


# @spec ING-EP-099, ING-EP-097
async def test_faults_reported_as_returned_and_stop_reports_nothing_new(
    started, fake_factory, clock, helpers, caplog
) -> None:
    caplog.set_level(logging.INFO)
    plugin, state = started
    fault = RuntimeError("retained-cleanup-fault")
    fake_factory.faults.append(fault)
    # A task already ended with an exception: the plugin reads none. Its
    # ``exception()`` and ``result()`` count every read, so a plugin that
    # read the exception silently, without logging it, still fails.
    ended = _ReadSpyTask(
        _raise("task-exception-not-read"), loop=asyncio.get_running_loop()
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert ended.done() and not ended.cancelled()
    fake_factory.tasks.add(ended)
    await helpers.drive(clock, helpers.begin_drain(plugin, state, 4))
    # The fault is reported once, as returned: the object itself.
    reports = [
        record
        for record in caplog.records
        if fault in _flatten(record.args)
        or (record.exc_info is not None and record.exc_info[1] is fault)
    ]
    assert len(reports) == 1, [r.getMessage() for r in caplog.records]
    assert "task-exception-not-read" not in caplog.text
    assert ended.reads == [], f"the plugin read the task's {ended.reads}"
    assert plugin.fault is None, "faults at shutdown are not escalated"
    assert state.request_shutdown.calls == 0
    caplog.clear()
    before = len(fake_factory.calls)
    await asyncio.wait_for(plugin.stop(state), 5.0)
    helpers.factory_settlement(fake_factory, since=before)
    assert "retained-cleanup-fault" not in caplog.text, "nothing reported twice"
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR], [
        r.getMessage() for r in caplog.records
    ]
    assert ended.reads == [], f"the plugin read the task's {ended.reads}"


async def _raise(message: str) -> None:
    raise RuntimeError(message)


class _ReadSpyTask(asyncio.tasks._PyTask):  # type: ignore[name-defined]
    """A task that records every read of its outcome (``reads``).

    The test's own read, when it ends, is not recorded, so no "exception
    was never retrieved" report is left whatever the test's outcome.
    """

    def __init__(self, coro: Any, *, loop: Any) -> None:
        super().__init__(coro, loop=loop)
        self.reads: list[str] = []
        self.add_done_callback(lambda task: task.retrieve())

    def exception(self) -> BaseException | None:
        self.reads.append("exception")
        return super().exception()

    def result(self) -> Any:
        self.reads.append("result")
        return super().result()

    def retrieve(self) -> None:
        """Read the outcome without recording it (the test's own read)."""
        if not self.cancelled():
            super().exception()


# @spec ING-EP-103, ING-EP-093, ING-EP-096
async def test_settlement_order_on_a_stopped_engine_with_the_fake_factory(
    started, fake_factory, clock, helpers
) -> None:
    plugin, state = started
    run, lease = await helpers.stream_with_lease(plugin, fake_factory, clock)
    lease.script("release", Block())  # the owner hands release over
    retirement = Block()
    fake_factory.script("cleanup", retirement)  # outlives its owner
    state.draining = True  # zero budget: core calls no drain, only stop
    stopping = asyncio.ensure_future(plugin.stop(state))
    await advance_until(
        clock,
        lambda: lease in fake_factory.handed_over,
        limit=int(plugin.config.session_cleanup_timeout) + 10,
    )
    await helpers.settle_until(clock, lambda: run.done)
    await helpers.settle_until(clock, lambda: math.inf in fake_factory.drains)
    assert not stopping.done(), "drain(math.inf) awaits the cleanup task"
    assert ("teardown_all", None) not in fake_factory.events
    retirement.release()
    await asyncio.wait_for(stopping, 5.0)
    events = fake_factory.events
    assert events.index(("cleanup.end", lease.index)) < events.index(
        ("teardown_all", None)
    )
    helpers.factory_settlement(fake_factory)
    assert lease.names()[-2:] == ["abort", "release"]
    assert not plugin.registry.records


# ---------------------------------------------------------------- stop


# @spec ING-EP-079
async def test_stop_marks_torn_down_settles_then_cancels_monitor(
    started, fake_factory, monkeypatch, clock, helpers
) -> None:
    plugin, state = started
    seen: list[tuple[bool, bool, bool]] = []
    real = fake_factory.teardown_all
    monitor = plugin.listener_monitor

    def spy():
        seen.append((plugin.torn_down, plugin.ready(state), monitor.done()))
        return real()

    monkeypatch.setattr(fake_factory, "teardown_all", spy)
    await helpers.drive(clock, plugin.stop(state))
    assert seen == [(True, False, False)], (
        "torn down before the settlement; the monitor ends after it"
    )
    assert monitor.done()
    assert plugin.torn_down
    helpers.factory_settlement(fake_factory)


# @spec ING-EP-079
async def test_cancelled_stop_completes_then_reraises(
    started, fake_factory, clock, helpers
) -> None:
    plugin, state = started
    release_owner = asyncio.Event()
    dummy, _ = helpers.register_dummy(plugin, stubborn=release_owner)
    task = asyncio.ensure_future(plugin.stop(state))
    await clock.settle(20)
    for _ in range(3):
        task.cancel()
        await clock.settle()
    assert not task.done(), "stop's steps are not interrupted"
    assert fake_factory.teardown_calls == 0
    release_owner.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5.0)
    assert dummy.done()
    helpers.factory_settlement(fake_factory)
    assert plugin.listener_monitor.done()


# @spec ING-EP-093, ING-EP-106
async def test_stop_awaits_continuations_without_cancelling_them(
    started, clock, helpers, grpc_port
) -> None:
    plugin, state = started
    owner, record = helpers.register_dummy(plugin, "file_route")
    finish = asyncio.Event()

    async def cleanup() -> str:
        await finish.wait()
        return "settled"

    continuation = asyncio.ensure_future(cleanup())
    plugin.registry.attach_continuation(record, continuation)
    collector = plugin.fault_collector
    task = asyncio.ensure_future(plugin.stop(state))
    await clock.settle(20)
    assert owner.cancelled() or owner.cancelling()
    assert not continuation.done(), "the continuation is awaited, not cancelled"
    assert not task.done()
    finish.set()
    await asyncio.wait_for(task, 5.0)
    assert continuation.result() == "settled"
    assert collector.done() and not collector.cancelled()
    assert plugin.listener_monitor.done()
    assert not helpers.port_open(grpc_port)


# @spec ING-EP-077
async def test_stop_starts_no_engine_work_beyond_owner_abort_release(
    started, fake_factory, fake_seam, fake_engine, clock, helpers
) -> None:
    plugin, state = started
    run, lease = await helpers.stream_with_lease(plugin, fake_factory, clock)
    calls_before = len(lease.calls)
    opens_before = fake_factory.names().count("open")
    aborted_before = list(fake_engine.aborted)
    health_before = fake_engine.health_checks
    await helpers.drive(clock, plugin.stop(state))
    await helpers.settle_until(clock, lambda: run.done)
    after = [c.name for c in lease.calls[calls_before:]]
    assert after == ["abort", "release"], after
    assert lease.teardown_calls == 0
    assert fake_seam.engine_calls(lease) == ["generate", "abort"]
    assert fake_factory.names().count("open") == opens_before
    assert fake_engine.aborted == aborted_before
    assert fake_engine.health_checks == health_before
