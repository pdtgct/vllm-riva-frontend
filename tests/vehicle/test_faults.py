"""The owner registry, the two supervision tasks and fault escalation."""

from __future__ import annotations

import asyncio
import contextlib

import grpc
import pytest
from support.fake_session import Block, Raise, Return, failed, refused
from support.riva import config_request, recognize, recognize_request

# ------------------------------------------------------------- registry


# @spec ING-EP-106
async def test_registry_records_persist_until_unregistered() -> None:
    from vllm_riva_frontend.admission import OwnerKind
    from vllm_riva_frontend.lifecycle import OwnerRegistry

    registry = OwnerRegistry()
    task = asyncio.ensure_future(asyncio.sleep(0))
    record = registry.register(OwnerKind.file_route, task)
    await task
    assert record in registry.records.values(), (
        "a finished task keeps its record"
    )
    continuation = asyncio.ensure_future(asyncio.Event().wait())
    registry.attach_continuation(record, continuation)
    assert record.continuation is continuation
    assert record in registry.records.values(), "kept under its continuation"
    continuation.cancel()
    await asyncio.wait({continuation})
    registry.unregister(record)
    assert record not in registry.records.values()


# @spec ING-EP-106, ING-ADM-013
async def test_one_stream_owner_is_registered_for_the_stream_lifetime(
    started, fake_factory, clock, helpers
) -> None:
    from vllm_riva_frontend.admission import OwnerKind

    plugin, _ = started
    run, lease = await helpers.stream_with_lease(plugin, fake_factory, clock)
    records = list(plugin.registry.records.values())
    assert [r.kind for r in records] == [OwnerKind.stream]
    block = Block()
    lease.script("release", block)
    run.requests.close()
    await helpers.settle_until(clock, lambda: block.entered.is_set())
    assert len(plugin.registry.records) == 1, "kept until the last cleanup step"
    assert plugin.ingress.count == 1
    block.release()
    await helpers.settle_until(clock, lambda: run.done)
    assert not plugin.registry.records
    assert plugin.ingress.count == 0


# @spec ING-EP-106
async def test_one_registered_owner_per_admitted_surface_and_no_other_tasks(
    started, test_app, fake_factory, fake_transcription, clock, helpers
) -> None:
    from support.asgi import asgi_http
    from support.realtime import RealtimeClient

    from vllm_riva_frontend.admission import OwnerKind

    plugin, _ = started
    supervisors = {plugin.fault_collector, plugin.listener_monitor}
    baseline = asyncio.all_tasks()
    run, _lease = await helpers.stream_with_lease(plugin, fake_factory, clock)
    client = RealtimeClient(test_app, clock)
    await client.open_session(helpers.realtime_update()["session"])
    block = Block()
    fake_transcription.script(block)
    body, headers = helpers.file_request_body()
    request = asyncio.ensure_future(
        asgi_http(
            test_app, "POST", helpers.file_route(), headers=headers, body=body
        )
    )
    await helpers.settle_until(clock, lambda: fake_transcription.calls != [])
    kinds = sorted(r.kind for r in plugin.registry.records.values())
    assert kinds == sorted(
        [OwnerKind.stream, OwnerKind.realtime, OwnerKind.file_route]
    )
    test_tasks = {run.task, client.ws.task, request}
    # Tasks running the frontend's or the fakes' code; the ASGI framework's
    # and core's own tasks are not the plugin's.
    new = {
        task
        for task in asyncio.all_tasks() - baseline - test_tasks
        if any(
            part in _code_file(task)
            for part in ("vllm_riva_frontend", "/tests/support/")
        )
    }
    allowed = (
        helpers.frontend_tasks(plugin)
        | fake_factory.owned_tasks()
        | supervisors
    )
    assert new <= allowed, f"tasks outside the registry: {new - allowed}"
    block.release()
    await asyncio.wait_for(request, 5.0)
    await client.disconnect()
    run.requests.close()
    await helpers.settle_until(clock, lambda: run.done)


def _code_file(task: asyncio.Task) -> str:
    """The source file of the coroutine ``task`` runs, or ``""``."""
    code = getattr(task.get_coro(), "cr_code", None)
    return "" if code is None else code.co_filename


# @spec ING-EP-106
async def test_recognize_registers_one_recognize_owner(
    started, fake_transcription, clock, helpers
) -> None:
    from support.audio import pcm16_bytes, pcm16_sine

    from vllm_riva_frontend.admission import OwnerKind

    plugin, _ = started
    block = Block()
    fake_transcription.script(block)
    call = asyncio.ensure_future(
        recognize(
            plugin.servicer, recognize_request(pcm16_bytes(pcm16_sine(0.5)))
        )
    )
    await helpers.settle_until(clock, lambda: fake_transcription.calls != [])
    kinds = [r.kind for r in plugin.registry.records.values()]
    assert kinds == [OwnerKind.recognize]
    block.release()
    await asyncio.wait_for(call, 5.0)
    await helpers.settle_until(clock, lambda: not plugin.registry.records)


# @spec ING-EP-106
async def test_file_route_registers_one_file_route_owner(
    started, test_app, fake_transcription, clock, helpers
) -> None:
    from support.asgi import asgi_http

    from vllm_riva_frontend.admission import OwnerKind

    plugin, _ = started
    block = Block()
    fake_transcription.script(block)
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
    kinds = [r.kind for r in plugin.registry.records.values()]
    assert kinds == [OwnerKind.file_route]
    block.release()
    response = await asyncio.wait_for(request, 5.0)
    assert response.status == 200
    await helpers.settle_until(clock, lambda: not plugin.registry.records)


# @spec ING-EP-106
async def test_realtime_session_registers_one_realtime_owner(
    started, test_app, fake_factory, clock, helpers
) -> None:
    from support.realtime import RealtimeClient

    from vllm_riva_frontend.admission import OwnerKind

    plugin, _ = started
    client = RealtimeClient(test_app, clock)
    assert await client.connect()
    await helpers.settle_until(clock, lambda: bool(plugin.registry.records))
    kinds = [r.kind for r in plugin.registry.records.values()]
    assert kinds == [OwnerKind.realtime]
    await client.disconnect()
    await helpers.settle_until(clock, lambda: not plugin.registry.records)


# ---------------------------------------------------------- fault collector


# @spec ING-EP-067
async def test_collector_drains_the_factory_every_second(
    started, fake_factory, clock
) -> None:
    plugin, _ = started
    first = len(fake_factory.drains)
    times = []
    for _ in range(3):
        await clock.advance(1.0)
        times.append(clock.monotonic())
    collected = fake_factory.drains[first:]
    assert len(collected) == 3
    for value, now in zip(collected, times, strict=True):
        assert value <= now, "an immediate deadline"
        assert value >= now - 1.0


# @spec ING-EP-067, ING-EP-111
async def test_collected_fault_is_recorded_and_escalated(
    started, fake_factory, clock, helpers
) -> None:
    plugin, state = started
    fake_factory.faults.append(RuntimeError("orphan retirement failed"))
    await clock.advance(1.0)
    await helpers.settle_until(clock, lambda: plugin.fault is not None)
    assert plugin.ready(state) is False
    assert state.request_shutdown.calls == 1


# @spec ING-EP-067
async def test_collector_drains_every_second_serialized(
    started, fake_factory, clock, helpers
) -> None:
    plugin, state = started
    block = Block()
    fake_factory.script("drain", block)
    await clock.advance(1.0)
    await helpers.settle_until(clock, lambda: block.entered.is_set())
    in_flight = len(fake_factory.drains)
    task = helpers.begin_drain(plugin, state, 4)
    await clock.settle(20)
    assert len(fake_factory.drains) == in_flight, (
        "the shutdown's factory drain waits for the collector's call"
    )
    block.release()
    await helpers.drive(clock, task)
    assert fake_factory.max_concurrent_drains == 1
    assert len(fake_factory.drains) > in_flight
    helpers.factory_settlement(fake_factory)


# @spec ING-EP-100
@pytest.mark.parametrize("ending", ["drain", "stop"])
async def test_drain_and_stop_end_collector_without_cancel(
    started, fake_factory, clock, helpers, ending
) -> None:
    plugin, state = started
    collector = plugin.fault_collector
    fault = RuntimeError("collected while ending")
    # A collection call in progress when the collector is told to end.
    block = Block(then=Return([fault]))
    fake_factory.script("drain", block)
    await clock.advance(1.0)
    await helpers.settle_until(clock, lambda: block.entered.is_set())
    if ending == "drain":
        task = helpers.begin_drain(plugin, state, 4)
    else:
        task = asyncio.ensure_future(plugin.stop(state))
    await clock.settle(20)
    assert not collector.done(), "the call in progress is not cut short"
    block.release()
    await helpers.drive(clock, task)
    assert collector.done()
    assert not collector.cancelled(), "ended by its signal"
    assert plugin.fault is fault, "its faults are recorded"


class _CancelCountingTask(asyncio.tasks._PyTask):  # type: ignore[name-defined]
    """A task that counts the calls to its ``cancel``."""

    def __init__(self, coro, *, loop) -> None:
        super().__init__(coro, loop=loop)
        self.cancels = 0

    def cancel(self, msg=None) -> bool:
        self.cancels += 1
        return super().cancel(msg)


# @spec ING-EP-100, ING-EP-067
@pytest.mark.parametrize("when", ["between_calls", "during_a_call"])
async def test_collector_ends_by_its_signal_and_is_never_cancelled(
    clock, when
) -> None:
    """``request_stop`` ends the collector without cancelling its task.

    Between two calls the period's wait ends on the signal; during a call
    the call completes and its faults are recorded, then the collector ends.
    """
    from vllm_riva_frontend.lifecycle import FaultCollector

    fault = RuntimeError("collected while stopping")
    entered, release = asyncio.Event(), asyncio.Event()
    calls: list[float] = []

    class Factory:
        async def drain(self, deadline):
            calls.append(deadline)
            entered.set()
            await release.wait()
            return [fault]

    recorded: list[BaseException] = []
    collector = FaultCollector(Factory(), 1.0, recorded.append, clock=clock)
    task = _CancelCountingTask(collector.run(), loop=asyncio.get_running_loop())
    await clock.settle()
    if when == "during_a_call":
        await clock.advance(1.0)
        await asyncio.wait_for(entered.wait(), 2)
    collector.request_stop()
    await clock.settle()
    if when == "during_a_call":
        assert not task.done(), "the call in progress was cut short"
        release.set()
    await asyncio.wait_for(task, 2)
    assert task.cancels == 0, "the collector was cancelled"
    assert not task.cancelled()
    assert clock.sleepers == 0, "the period's timer outlived the collector"
    if when == "during_a_call":
        assert recorded == [fault]
    else:
        assert calls == [] and recorded == []


# @spec ING-EP-100
async def test_rolled_back_start_cancels_the_collector(
    initialized, monkeypatch
) -> None:
    import vllm_riva_frontend.lifecycle as lifecycle

    plugin, state = initialized

    class Boom:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("monitor failed")

    monkeypatch.setattr(lifecycle, "ListenerMonitor", Boom)
    with pytest.raises(RuntimeError, match="monitor failed"):
        await plugin.start(state)
    assert plugin.fault_collector is not None
    assert plugin.fault_collector.cancelled()


# --------------------------------------------------------- listener monitor


# @spec ING-EP-016, ING-EP-111
async def test_unexpected_listener_exit_is_a_fault(started, helpers) -> None:
    plugin, state = started
    await plugin.server.stop(None)
    await helpers.until(lambda: plugin.fault is not None)
    assert plugin.ready(state) is False
    assert state.request_shutdown.calls == 1


# @spec ING-EP-016
async def test_listener_exit_caused_by_drain_or_stop_is_not_a_fault(
    started, clock, helpers
) -> None:
    plugin, state = started
    await helpers.drive(clock, helpers.begin_drain(plugin, state, 20))
    await asyncio.sleep(0.02)
    assert plugin.fault is None
    await helpers.drive(clock, plugin.stop(state))
    await asyncio.sleep(0.02)
    assert plugin.fault is None
    assert state.request_shutdown.calls == 0


# ------------------------------------------------------------- escalation


# @spec ING-EP-111
async def test_engine_dead_on_the_stream_path_escalates(
    started, fake_factory, clock, helpers
) -> None:
    plugin, state = started
    run, lease = await helpers.stream_with_lease(plugin, fake_factory, clock)
    lease.script("feed", Raise(failed("engine_dead")))
    run.requests.push(helpers.audio_message())
    await helpers.settle_until(clock, lambda: run.done)
    run.assert_status(grpc.StatusCode.UNAVAILABLE, "service_unavailable")
    assert plugin.fault is not None
    assert plugin.ready(state) is False
    assert state.request_shutdown.calls == 1


# @spec ING-EP-111
async def test_engine_dead_open_refusal_escalates(
    started, fake_factory, clock, helpers
) -> None:
    plugin, state = started
    fake_factory.script("open", Raise(failed("engine_dead")))
    run = helpers.stream(
        plugin, plugin.clock, config_request(), helpers.audio_message()
    )
    await helpers.settle_until(clock, lambda: run.done)
    assert state.request_shutdown.calls == 1
    assert plugin.ready(state) is False


# @spec ING-EP-101
@pytest.mark.parametrize(
    "reason", ["rejected_input", "expired", "abnormal_end"]
)
async def test_other_lease_failures_do_not_escalate(
    started, fake_factory, clock, helpers, reason
) -> None:
    plugin, state = started
    run, lease = await helpers.stream_with_lease(plugin, fake_factory, clock)
    lease.script("feed", Raise(failed(reason)))
    run.requests.push(helpers.audio_message())
    await helpers.settle_until(clock, lambda: run.done)
    assert state.request_shutdown.calls == 0
    assert plugin.fault is None
    assert plugin.ready(state) is True


# @spec ING-EP-044
async def test_cleanup_bound_expiry_is_not_an_escalation_trigger(
    started, fake_factory, clock, helpers
) -> None:
    plugin, state = started
    run, lease = await helpers.stream_with_lease(plugin, fake_factory, clock)
    lease.script("release", Block())
    run.context.cancel()
    await helpers.settle_until(clock, lambda: "release" in lease.names())
    await clock.advance(plugin.config.session_cleanup_timeout + 1)
    await helpers.settle_until(clock, lambda: run.done)
    assert state.request_shutdown.calls == 0
    assert plugin.fault is None
    assert plugin.ready(state) is True


# @spec ING-EP-045
async def test_engine_death_on_the_file_path_is_not_escalated(
    started, test_app, fake_transcription, clock, helpers
) -> None:
    from support.asgi import asgi_http
    from vllm.v1.engine.exceptions import EngineDeadError

    plugin, state = started
    fake_transcription.script(Raise(EngineDeadError()))
    body, headers = helpers.file_request_body()
    response = await asgi_http(
        test_app, "POST", helpers.file_route(), headers=headers, body=body
    )
    assert response.status is not None and response.status >= 500
    with contextlib.suppress(AssertionError):
        await helpers.settle_until(
            clock, lambda: plugin.fault is not None, rounds=20
        )
    assert plugin.fault is None
    assert state.request_shutdown.calls == 0


# @spec ING-EP-111
async def test_fault_closes_admission_before_request_shutdown_and_leaves_health(
    started, fake_factory, test_app, clock, helpers
) -> None:
    from grpc_health.v1 import health_pb2, health_pb2_grpc
    from support.grpc_harness import aio_channel
    from support.state import RecordingShutdown

    from vllm_riva_frontend.admission import OwnerKind
    from vllm_riva_frontend.grpc import RIVA_SERVICE_NAME

    plugin, state = started
    run, lease = await helpers.stream_with_lease(plugin, fake_factory, clock)

    def observe():
        admitted = plugin.ingress.try_admit(OwnerKind.stream)
        if admitted is not None:
            admitted.release()
        gate = plugin.file_gate
        return {
            "admission_open": plugin.admission_open,
            "admits_stream": admitted is not None,
            "gate_closed": gate is None or gate.closed,
            "ready": plugin.ready(state),
            "draining": getattr(state, "draining", False),
        }

    state.request_shutdown = RecordingShutdown(on_call=observe)
    fake_factory.faults.append(RuntimeError("retained cleanup fault"))
    await clock.advance(1.0)
    await helpers.settle_until(clock, lambda: state.request_shutdown.calls)
    assert state.request_shutdown.observations == [
        {
            "admission_open": False,
            "admits_stream": False,
            "gate_closed": True,
            "ready": False,
            "draining": False,
        }
    ], "admission closes before the shutdown request"
    assert getattr(state, "draining", False) is False
    async with aio_channel(plugin.config.grpc_bind) as channel:
        stub = health_pb2_grpc.HealthStub(channel)
        for service in ("", RIVA_SERVICE_NAME):
            reply = await stub.Check(
                health_pb2.HealthCheckRequest(service=service), timeout=2.0
            )
            assert reply.status == health_pb2.HealthCheckResponse.SERVING, (
                "a fault leaves health as start left it"
            )
    # The admitted stream keeps its own path.
    run.requests.close()
    await helpers.settle_until(clock, lambda: run.done)
    assert run.status is grpc.StatusCode.OK
    assert lease.names()[-3:] == ["flush", "finish", "release"]


# @spec ING-EP-101
async def test_ordinary_refusal_and_generation_failure_do_not_escalate(
    started, fake_factory, clock, helpers
) -> None:
    plugin, state = started
    fake_factory.script("open", Raise(refused("capacity")))
    refused_run = helpers.stream(
        plugin, plugin.clock, config_request(), helpers.audio_message()
    )
    await helpers.settle_until(clock, lambda: refused_run.done)
    assert refused_run.status is grpc.StatusCode.RESOURCE_EXHAUSTED
    run, lease = await helpers.stream_with_lease(plugin, fake_factory, clock)
    lease.fail_generation(failed("abnormal_end"))
    run.requests.push(helpers.audio_message())
    await helpers.settle_until(clock, lambda: run.done)
    for _ in range(3):
        await clock.advance(1.0)  # the collector runs
    collected = await fake_factory.drain(clock.monotonic())
    assert collected == [], "nothing retained"
    assert plugin.fault is None
    assert state.request_shutdown.calls == 0
    assert plugin.ready(state) is True
