"""The owner registry, the two supervision tasks and fault escalation."""

from __future__ import annotations

import asyncio
import contextlib

import grpc
import pytest
from support.fake_session import Block, Raise, failed
from support.riva import config_request, recognize, recognize_request

# ------------------------------------------------------------- registry


# @spec ING-EP-060
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
    cancelled = registry.cancel_all()
    assert continuation in cancelled
    await asyncio.sleep(0)
    assert continuation.cancelled()
    registry.unregister(record)
    assert record not in registry.records.values()


# @spec ING-EP-060, ING-ADM-013
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


# @spec ING-EP-060
async def test_owner_tasks_are_the_only_frontend_tasks_of_a_stream(
    started, fake_factory, clock, helpers
) -> None:
    plugin, _ = started
    baseline = asyncio.all_tasks()
    run, _lease = await helpers.stream_with_lease(plugin, fake_factory, clock)
    new = asyncio.all_tasks() - baseline - {run.task}
    allowed = helpers.frontend_tasks(plugin) | set(fake_factory.tasks)
    assert new <= allowed, f"tasks outside the registry: {new - allowed}"
    run.requests.close()
    await helpers.settle_until(clock, lambda: run.done)


# @spec ING-EP-060
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


# @spec ING-EP-060
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
            "/v1/audio/transcriptions",
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


# @spec ING-EP-060
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


# @spec ING-EP-015
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


# @spec ING-EP-015, ING-EP-017
async def test_collected_fault_is_recorded_and_escalated(
    started, fake_factory, clock, helpers
) -> None:
    plugin, state = started
    fake_factory.faults.append(RuntimeError("orphan retirement failed"))
    await clock.advance(1.0)
    await helpers.settle_until(clock, lambda: plugin.fault is not None)
    assert plugin.ready(state) is False
    assert state.request_shutdown.calls == 1


# @spec ING-EP-015
async def test_collector_is_serialized_with_the_shutdown_drain(
    started, fake_factory, clock, helpers
) -> None:
    plugin, state = started
    block = Block()
    fake_factory.script("drain", block)
    await clock.advance(1.0)
    await helpers.settle_until(clock, lambda: block.entered.is_set())
    in_flight = len(fake_factory.drains)
    state.draining = True
    task = asyncio.ensure_future(plugin.drain(state, 4))
    await clock.settle(20)
    assert len(fake_factory.drains) == in_flight, (
        "the shutdown drain waits for the collector's call"
    )
    block.release()
    await helpers.drive(clock, task)
    assert len(fake_factory.drains) == in_flight + 1


# --------------------------------------------------------- listener monitor


# @spec ING-EP-016, ING-EP-017
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
    state.draining = True
    await helpers.drive(clock, plugin.drain(state, 20))
    await asyncio.sleep(0.02)
    assert plugin.fault is None
    await helpers.drive(clock, plugin.stop(state))
    await asyncio.sleep(0.02)
    assert plugin.fault is None
    assert state.request_shutdown.calls == 0


# ------------------------------------------------------------- escalation


# @spec ING-EP-017
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


# @spec ING-EP-017
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


# @spec ING-EP-017
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
        test_app, "POST", "/v1/audio/transcriptions", headers=headers, body=body
    )
    assert response.status is not None and response.status >= 500
    with contextlib.suppress(AssertionError):
        await helpers.settle_until(
            clock, lambda: plugin.fault is not None, rounds=20
        )
    assert plugin.fault is None
    assert state.request_shutdown.calls == 0
