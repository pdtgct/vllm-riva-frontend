"""``attach_router``, ``start``, ``ready`` and the one-process topology.

``start`` runs a real ``grpc.aio`` server on a free loopback port; the tests
talk to it through ``support.grpc_harness.aio_channel``.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from typing import Any

import grpc
import pytest
from support.riva import config_request

SERVING = 1
NOT_SERVING = 2


async def health_status(address: str, service: str) -> int:
    from grpc_health.v1 import health_pb2, health_pb2_grpc
    from support.grpc_harness import aio_channel

    async with aio_channel(address) as channel:
        stub = health_pb2_grpc.HealthStub(channel)
        reply = await stub.Check(
            health_pb2.HealthCheckRequest(service=service), timeout=2.0
        )
        return int(reply.status)


def address_of(plugin: Any) -> str:
    return plugin.config.grpc_bind


# ------------------------------------------------------------ attach_router


# @spec ING-EP-007
def test_attach_router_adds_bootstrap_operational_and_file_routes(
    frontend, test_app
) -> None:
    from support.app import find_route

    for path in (
        "/v1/health/ready",
        "/v1/health/live",
        "/v1/version",
        "/v1/metadata",
    ):
        route = find_route(test_app, "GET", path)
        assert route is not None, path
    assert find_route(test_app, "POST", "/v1/realtime/transcription_sessions")
    file_route = find_route(test_app, "POST", "/v1/audio/transcriptions")
    assert file_route is not None
    assert "speech_to_text" not in file_route.endpoint.__module__, (
        "core's transcription route is replaced by the frontend's"
    )


# @spec ING-EP-007
def test_dispatcher_is_middleware_inside_cores_authentication(
    frontend, helpers
) -> None:
    from support.app import middleware_stack
    from vllm.entrypoints.serve.middleware.authenticate import (
        AuthenticationMiddleware,
    )

    from vllm_riva_frontend.nim_ws import RealtimeDispatcher

    app = helpers.build(frontend, api_key="secret")
    stack = middleware_stack(app)
    assert RealtimeDispatcher in stack
    assert AuthenticationMiddleware in stack
    assert stack.index(AuthenticationMiddleware) < stack.index(
        RealtimeDispatcher
    ), "core's authentication is outermost"


# @spec ING-EP-007
async def test_api_key_applies_to_claimed_scopes_and_plugin_routes(
    clock, fake_factory, helpers
) -> None:
    from support.app import http_client
    from support.realtime import REALTIME_URL, RealtimeClient

    plugin = helpers.make_frontend(
        clock, factories={"nemotron": fake_factory.constructor}
    )
    app, state = await helpers.initialize(
        plugin, helpers.engine_with(), api_key="secret"
    )
    await plugin.start(state)
    try:
        async with http_client(app) as client:
            denied = await client.post("/v1/realtime/transcription_sessions")
            assert denied.status_code == 401
        async with http_client(app, api_key="secret") as client:
            allowed = await client.post("/v1/realtime/transcription_sessions")
            assert allowed.status_code == 200
        client = RealtimeClient(app, clock, url=REALTIME_URL)
        assert not await client.connect()
        assert plugin.ingress.count == 0
        assert fake_factory.names() == []
    finally:
        await helpers.drive(clock, plugin.stop(state))


# ------------------------------------------------------------------- start


# @spec ING-EP-005
async def test_start_serves_riva_health_and_reflection_on_the_bind(
    started,
) -> None:
    from grpc_reflection.v1alpha import reflection_pb2, reflection_pb2_grpc
    from riva.client.proto import riva_asr_pb2, riva_asr_pb2_grpc
    from support.grpc_harness import aio_channel

    from vllm_riva_frontend.grpc import RIVA_SERVICE_NAME

    plugin, _ = started
    address = address_of(plugin)
    async with aio_channel(address) as channel:
        await asyncio.wait_for(channel.channel_ready(), 2.0)
        stub = riva_asr_pb2_grpc.RivaSpeechRecognitionStub(channel)
        reply = await stub.GetRivaSpeechRecognitionConfig(
            riva_asr_pb2.RivaSpeechRecognitionConfigRequest(), timeout=2.0
        )
        assert [c.model_name for c in reply.model_config]
        reflection = reflection_pb2_grpc.ServerReflectionStub(channel)

        async def requests():
            yield reflection_pb2.ServerReflectionRequest(list_services="")

        names: set[str] = set()
        async for answer in reflection.ServerReflectionInfo(requests()):
            names |= {s.name for s in answer.list_services_response.service}
            break
    assert RIVA_SERVICE_NAME in names
    assert "grpc.health.v1.Health" in names
    assert await health_status(address, "") == SERVING
    assert await health_status(address, RIVA_SERVICE_NAME) == SERVING


# @spec ING-EP-005
async def test_listener_is_reachable_when_start_returns(
    initialized, grpc_port, helpers
) -> None:
    plugin, state = initialized
    assert not helpers.port_open(grpc_port)
    await plugin.start(state)
    assert helpers.port_open(grpc_port)


# @spec ING-EP-005
async def test_port_zero_is_refused(clock, fake_factory, helpers) -> None:
    from vllm_riva_frontend.lifecycle import StartupError

    plugin = helpers.make_frontend(
        clock, factories={"nemotron": fake_factory.constructor}
    )
    engine = helpers.engine_with(options={"grpc_bind": "127.0.0.1:0"})
    with pytest.raises(StartupError):
        _, state = await helpers.initialize(plugin, engine)
        await plugin.start(state)


# @spec ING-EP-005, ING-EP-035
async def test_health_is_not_serving_until_admission_opens_last(
    initialized, monkeypatch
) -> None:
    from grpc_health.v1 import health, health_pb2

    plugin, state = initialized
    seen: list[tuple[str, int, bool, bool]] = []
    # The listener is a grpc.aio server, so its health service is the aio
    # HealthServicer, whose set is a coroutine.
    original = health.aio.HealthServicer.set

    async def spy(self, service, status):
        seen.append(
            (
                service,
                int(status),
                bool(getattr(plugin, "admission_open", False)),
                getattr(plugin, "listener_monitor", None) is not None
                and getattr(plugin, "fault_collector", None) is not None,
            )
        )
        return await original(self, service, status)

    monkeypatch.setattr(health.aio.HealthServicer, "set", spy)
    await plugin.start(state)

    assert seen, "start sets the health status"
    first_serving = next(
        i
        for i, item in enumerate(seen)
        if item[1] == health_pb2.HealthCheckResponse.SERVING
    )
    assert any(
        item[1] == health_pb2.HealthCheckResponse.NOT_SERVING
        for item in seen[:first_serving]
    ), "NOT_SERVING is set before SERVING"
    for _service, status, admitted, supervised in seen:
        if status == health_pb2.HealthCheckResponse.NOT_SERVING:
            assert not admitted
        if status == health_pb2.HealthCheckResponse.SERVING:
            assert supervised, "collector and monitor exist before SERVING"
    assert plugin.admission_open
    assert not plugin.fault_collector.done()
    assert not plugin.listener_monitor.done()


# @spec ING-EP-035
async def test_no_owner_is_admitted_before_start(initialized, helpers) -> None:
    plugin, _ = initialized
    run = helpers.stream(plugin, plugin.clock, config_request())
    await asyncio.wait_for(run.task, 2.0)
    assert run.status is grpc.StatusCode.UNAVAILABLE
    assert run.details.startswith("service_unavailable")
    assert run.requests.pulls == 0
    assert len(plugin.registry.records) == 0
    assert plugin.ingress.count == 0


# @spec ING-EP-008
async def test_start_failure_after_bind_rolls_back(
    initialized, fake_factory, grpc_port, monkeypatch, helpers
) -> None:
    import vllm_riva_frontend.lifecycle as lifecycle

    plugin, state = initialized

    class Boom:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("monitor failed")

    monkeypatch.setattr(lifecycle, "ListenerMonitor", Boom)
    with pytest.raises(RuntimeError, match="monitor failed"):
        await plugin.start(state)
    await asyncio.sleep(0.01)
    assert not helpers.port_open(grpc_port), "server stopped with no grace"
    assert fake_factory.teardown_calls == 1
    collector = getattr(plugin, "fault_collector", None)
    if collector is not None:
        await asyncio.sleep(0)
        assert collector.done()
    assert state.request_shutdown.calls == 0, "rollback exit is not a fault"
    assert plugin.fault is None
    assert not plugin.admission_open


# @spec ING-EP-055
@pytest.mark.parametrize("keepalive", [None, 20])
async def test_keepalive_option_sets_the_server_ping_interval(
    clock, fake_factory, helpers, monkeypatch, keepalive
) -> None:
    import grpc.aio

    recorded: list[list[tuple[str, Any]]] = []
    real = grpc.aio.server

    def spy(*args, **kwargs):
        recorded.append(list(kwargs.get("options") or ()))
        return real(*args, **kwargs)

    monkeypatch.setattr(grpc.aio, "server", spy)
    options = {} if keepalive is None else {"grpc_keepalive_seconds": keepalive}
    plugin = helpers.make_frontend(
        clock, factories={"nemotron": fake_factory.constructor}
    )
    _, state = await helpers.initialize(
        plugin, helpers.engine_with(options=options)
    )
    await plugin.start(state)
    try:
        assert len(recorded) == 1, "one grpc.aio.server"
        keepalive_options = [
            value
            for name, value in recorded[0]
            if name == "grpc.keepalive_time_ms"
        ]
        if keepalive is None:
            assert keepalive_options == []
        else:
            assert keepalive_options == [20_000]
    finally:
        await helpers.drive(clock, plugin.stop(state))


# ------------------------------------------------------------------- ready


# @spec ING-EP-009
async def test_ready_is_a_synchronous_attribute_read(
    started, fake_engine
) -> None:
    plugin, state = started
    assert not inspect.iscoroutinefunction(type(plugin).ready)
    checks = fake_engine.health_checks
    assert plugin.ready(state) is True
    assert fake_engine.health_checks == checks


# @spec ING-EP-009
async def test_ready_is_false_before_start(initialized) -> None:
    plugin, state = initialized
    assert plugin.ready(state) is False


# @spec ING-EP-009
async def test_ready_is_false_while_draining(started, clock) -> None:
    plugin, state = started
    state.draining = True
    task = asyncio.ensure_future(plugin.drain(state, 60))
    await clock.settle()
    assert plugin.ready(state) is False
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# @spec ING-EP-009
async def test_ready_is_false_after_stop_and_after_a_fault(
    started, fake_factory, clock, helpers
) -> None:
    plugin, state = started
    fake_factory.faults.append(RuntimeError("orphan cleanup failed"))
    await clock.advance(1.0)
    await helpers.settle_until(clock, lambda: plugin.fault is not None)
    assert plugin.ready(state) is False
    await helpers.drive(clock, plugin.stop(state))
    assert plugin.ready(state) is False


# ---------------------------------------------------- one process, one engine


# @spec ING-EP-004
async def test_one_process_serves_http_routes_and_one_grpc_listener(
    clock, fake_factory, helpers, grpc_port
) -> None:
    from support.app import find_route, http_client
    from support.asgi import AsgiWebSocket

    constructor = helpers.counting(fake_factory.constructor)
    plugin = helpers.make_frontend(clock, factories={"nemotron": constructor})
    engine = helpers.engine_with(grpc_port)
    app, state = await helpers.initialize(plugin, engine)
    await plugin.start(state)
    try:
        assert constructor.calls == 1
        assert plugin.factory is fake_factory
        assert fake_factory.constructed_with[0] is engine
        assert state.engine_client is engine
        # Core's /v1/realtime is still served for unclaimed scopes.
        ws = AsgiWebSocket(app, "/v1/realtime")
        assert await ws.connect()
        assert (await ws.receive_json())["type"] == "core.realtime"
        assert find_route(app, "GET", "/v1/metadata") is not None
        async with http_client(app) as client:
            assert (await client.get("/v1/health/live")).status_code == 200
        assert helpers.port_open(grpc_port)
    finally:
        await helpers.drive(clock, plugin.stop(state))


# @spec ING-VEH-002
async def test_concurrent_streams_commit_no_thread_per_session(
    started, fake_factory, helpers
) -> None:
    from riva.client.proto import riva_asr_pb2_grpc
    from support.grpc_harness import aio_channel

    plugin, _ = started
    sessions = 6
    threads_before = threading.active_count()
    async with aio_channel(address_of(plugin)) as channel:
        stub = riva_asr_pb2_grpc.RivaSpeechRecognitionStub(channel)
        calls = []
        for _ in range(sessions):
            call = stub.StreamingRecognize()
            await call.write(config_request())
            await call.write(helpers.audio_message())
            calls.append(call)
        await helpers.until(lambda: len(fake_factory.leases) == sessions)
        assert threading.active_count() - threads_before < sessions
        for call in calls:
            await call.done_writing()
        for call in calls:
            async for _ in call:
                pass
            assert await call.code() == grpc.StatusCode.OK
    assert all(
        lease.names()[-3:] == ["flush", "finish", "release"]
        for lease in fake_factory.leases
    )
