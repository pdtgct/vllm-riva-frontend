"""``drain`` and ``stop``.

``drain`` is called directly on the plugin with the fake clock (core sets
``state.draining`` first, as ``drain_endpoint_plugins`` does), so its deadline
``D`` and settle point are virtual times the test reaches with
``FakeClock.advance``. Stream owners are real ``StreamOwner`` instances driven
through the servicer with ``RequestStream`` and ``FakeServicerContext``.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import grpc
import pytest
from support.fake_session import Block
from support.riva import config_request


def begin_drain(plugin: Any, state: Any, timeout: float) -> asyncio.Task:
    state.draining = True
    return asyncio.ensure_future(plugin.drain(state, timeout))


def register_dummy(plugin: Any, kind_name: str = "stream", *, stubborn=None):
    """Register a plain task as an owner; ``stubborn`` delays its cleanup."""
    from vllm_riva_frontend.admission import OwnerKind

    async def body() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if stubborn is not None:
                await stubborn.wait()
            raise

    task = asyncio.ensure_future(body())
    record = plugin.registry.register(getattr(OwnerKind, kind_name), task)
    return task, record


#: A stream idle since a 60 s drain began must still be open at its settle
#: point (49 s) and at ``D`` (54 s): ``session_idle_timeout`` 59 s outlasts
#: both and stays below the engine's 60 s parked-idle deadline (ING-EP-014).
OUTLASTS_DRAIN = pytest.mark.parametrize(
    "frontend_options", [{"session_idle_timeout": 59}], indirect=True
)


# --------------------------------------------------------------- formulas


# @spec ING-EP-010
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


# @spec ING-EP-010, ING-EP-046
async def test_drain_deadline_comes_from_cores_timeout(started, clock) -> None:
    plugin, state = started
    t0 = clock.monotonic()
    task = begin_drain(plugin, state, 60)
    await clock.settle()
    assert plugin.drain_deadline == pytest.approx(t0 + 54)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


# @spec ING-EP-010
async def test_drain_sets_not_serving_and_closes_admission(
    started, clock, helpers
) -> None:
    from grpc_health.v1 import health_pb2, health_pb2_grpc
    from support.grpc_harness import aio_channel

    plugin, state = started
    task = begin_drain(plugin, state, 60)
    await clock.settle()
    from vllm_riva_frontend.grpc import RIVA_SERVICE_NAME

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
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


# @spec ING-EP-036
async def test_owners_finish_normally_before_the_settle_point(
    started, fake_factory, clock, helpers
) -> None:
    plugin, state = started
    run, lease = await helpers.stream_with_lease(plugin, fake_factory, clock)
    task = begin_drain(plugin, state, 60)  # D = t0 + 54, settle = t0 + 49
    await clock.advance(40)
    assert "flush" not in lease.names(), "no shutdown before the settle point"
    # The client finishes on its own and gets its final.
    run.requests.close()
    await helpers.settle_until(clock, lambda: run.done)
    assert run.status is grpc.StatusCode.OK
    assert lease.names()[-3:] == ["flush", "finish", "release"]
    await helpers.drive(clock, task)


# @spec ING-EP-037, ING-EP-039
@OUTLASTS_DRAIN
async def test_settle_point_finalizes_remaining_streams_then_drains_factory(
    started, fake_factory, clock, helpers, monkeypatch
) -> None:
    plugin, state = started
    run, lease = await helpers.stream_with_lease(plugin, fake_factory, clock)
    stops: list[tuple[Any, float]] = []
    real_stop = plugin.server.stop

    async def spy(grace):
        stops.append((grace, clock.monotonic()))
        return await real_stop(grace)

    monkeypatch.setattr(plugin.server, "stop", spy)
    t0 = clock.monotonic()
    task = begin_drain(plugin, state, 60)
    deadline = t0 + 54
    await clock.advance(48.5)
    assert "flush" not in lease.names()
    await clock.advance(1.0)  # past the settle point at t0 + 49
    await helpers.settle_until(clock, lambda: run.done)
    assert run.status is grpc.StatusCode.OK, "gRPC OK after the final"
    assert lease.names()[-3:] == ["flush", "finish", "release"]
    await helpers.drive(clock, task)
    assert stops, "the gRPC server is stopped"
    grace, at = stops[0]
    assert grace == pytest.approx(deadline - at)
    assert deadline in fake_factory.drains
    drain_index = max(
        i
        for i, call in enumerate(fake_factory.calls)
        if call.name == "drain" and call.args == (deadline,)
    )
    release_index = max(
        i for i, call in enumerate(fake_factory.calls) if call.name == "release"
    )
    assert release_index < drain_index


# @spec ING-EP-037, ING-EP-010
@OUTLASTS_DRAIN
async def test_shutdown_finalization_failing_by_d_aborts_with_unavailable(
    started, fake_factory, clock, helpers
) -> None:
    plugin, state = started
    run, lease = await helpers.stream_with_lease(plugin, fake_factory, clock)
    lease.script("flush", Block())
    t0 = clock.monotonic()
    task = begin_drain(plugin, state, 60)
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


# @spec ING-EP-038
async def test_settle_point_cancels_running_file_requests(
    started, test_app, fake_transcription, clock, helpers
) -> None:
    from support.asgi import asgi_http
    from support.fake_session import Block as CallBlock

    from vllm_riva_frontend.admission import OwnerKind
    from vllm_riva_frontend.errors import catalog

    plugin, state = started
    fake_transcription.script(CallBlock())
    body, headers = helpers.file_request_body()
    request = asyncio.ensure_future(
        asgi_http(
            test_app,
            "POST",
            "/v1/audio/transcriptions",
            headers=headers,
            body=body,
        )
    )
    await helpers.settle_until(clock, lambda: fake_transcription.calls != [])
    task = begin_drain(plugin, state, 60)
    await clock.advance(49.5)
    response = await asyncio.wait_for(request, 5.0)
    assert response.status == catalog().http_status("service_unavailable")
    await helpers.drive(clock, task)
    # Core's call never settles: the request's record stays registered under
    # its running cleanup continuation until stop (ING-EP-038, ING-FILE-071).
    records = list(plugin.registry.records.values())
    assert [record.kind for record in records] == [OwnerKind.file_route]
    assert records[0].continuation is not None
    assert not records[0].continuation.done()


# @spec ING-EP-040
async def test_timeout_within_engine_reserve_cancels_owners_immediately(
    started, fake_factory, clock, helpers
) -> None:
    plugin, state = started
    run, lease = await helpers.stream_with_lease(plugin, fake_factory, clock)
    dummy, _ = register_dummy(plugin)
    t0 = clock.monotonic()
    task = begin_drain(plugin, state, 4)  # not larger than the 5 s reserve
    await helpers.settle_until(clock, lambda: run.done and dummy.done())
    assert clock.monotonic() == t0, "no graceful phase was waited"
    assert "flush" not in lease.names()
    assert lease.names()[-2:] == ["abort", "release"]
    assert run.status is grpc.StatusCode.UNAVAILABLE
    assert dummy.cancelled()
    await helpers.drive(clock, task)
    assert fake_factory.drains[-1] <= t0 + 4


# @spec ING-EP-041
async def test_cancelled_drain_cancels_owners_without_awaiting_cleanup(
    started, clock, helpers
) -> None:
    plugin, state = started
    slow_cleanup = asyncio.Event()
    dummy, _ = register_dummy(plugin, stubborn=slow_cleanup)
    task = begin_drain(plugin, state, 60)
    await clock.advance(10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2.0)
    await asyncio.sleep(0)
    assert not dummy.done(), "drain did not await the owner's cleanup"
    assert dummy.cancelling(), "drain cancelled the owner"
    slow_cleanup.set()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(dummy, 2.0)


# @spec ING-EP-041
async def test_drain_honours_cancellation_while_the_factory_drains(
    started, fake_factory, clock
) -> None:
    plugin, state = started
    block = Block()
    task = begin_drain(plugin, state, 4)
    fake_factory.script("drain", block)
    await clock.settle(20)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2.0)
    block.release()


# ---------------------------------------------------------------- stop


# @spec ING-EP-059
async def test_stop_marks_torn_down_before_tearing_the_factory_down(
    started, fake_factory, monkeypatch, clock, helpers
) -> None:
    plugin, state = started
    seen: list[bool] = []
    real = fake_factory.teardown_all

    def spy():
        seen.append(plugin.torn_down)
        return real()

    monkeypatch.setattr(fake_factory, "teardown_all", spy)
    await helpers.drive(clock, plugin.stop(state))
    assert seen == [True]
    assert plugin.torn_down


# @spec ING-EP-059, ING-EP-060
async def test_stop_cancels_owners_continuations_and_supervisors(
    started, clock, helpers, grpc_port
) -> None:
    plugin, state = started
    owner, record = register_dummy(plugin, "file_route")
    continuation = asyncio.ensure_future(asyncio.Event().wait())
    plugin.registry.attach_continuation(record, continuation)
    collector, monitor = plugin.fault_collector, plugin.listener_monitor
    await helpers.drive(clock, plugin.stop(state))
    await asyncio.sleep(0)
    for task in (owner, continuation, collector, monitor):
        assert task.cancelled() or task.done()
    assert owner.cancelled() and continuation.cancelled()
    assert not helpers.port_open(grpc_port)


# @spec ING-EP-059
async def test_stop_waits_once_with_a_one_second_bound(
    started, clock, helpers
) -> None:
    plugin, state = started
    never = asyncio.Event()
    stubborn, _ = register_dummy(plugin, stubborn=never)
    t0 = clock.monotonic()
    await helpers.drive(clock, plugin.stop(state), step=0.25)
    assert clock.monotonic() - t0 <= 1.5
    assert not stubborn.done(), "stop returned whether or not tasks finished"
    never.set()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(stubborn, 2.0)


# @spec ING-EP-042, ING-EP-043
async def test_stop_awaits_no_engine_work_and_owners_tear_down(
    started, fake_factory, fake_engine, clock, helpers
) -> None:
    plugin, state = started
    run, lease = await helpers.stream_with_lease(plugin, fake_factory, clock)
    lease.script("release", Block())
    lease.script("abort", Block())
    calls_before = len(lease.calls)
    drains_before = len(fake_factory.drains)
    aborted_before = list(fake_engine.aborted)
    health_before = fake_engine.health_checks
    await helpers.drive(clock, plugin.stop(state))
    await helpers.settle_until(clock, lambda: run.done)
    after = [c.name for c in lease.calls[calls_before:]]
    assert after == ["teardown"], after
    assert lease.teardown_calls == 1
    assert len(fake_factory.drains) == drains_before
    assert fake_engine.aborted == aborted_before
    assert fake_engine.health_checks == health_before
