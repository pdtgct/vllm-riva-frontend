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
from support.fake_session import Block
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


# @spec ING-EP-063
def test_attach_router_adds_namespace_routes_beside_cores_file_route(
    frontend, test_app
) -> None:
    from support.app import (
        CORE_TRANSCRIPTION_ROUTE,
        PLUGIN_PREFIX,
        all_routes,
        find_route,
        frontend_routes,
    )

    rows = all_routes(test_app)
    for method, path in sorted(frontend_routes()):
        if method == "WEBSOCKET":
            assert any(
                kind == "websocket" and route_path == path
                for kind, route_path, _, _ in rows
            ), path
        else:
            route = find_route(test_app, method, path)
            assert route is not None, (method, path)
            assert path.startswith(f"{PLUGIN_PREFIX}/")
    core_file = find_route(test_app, *CORE_TRANSCRIPTION_ROUTE)
    assert core_file is not None
    assert "speech_to_text" in core_file.endpoint.__module__, (
        "core's transcription route is left to core"
    )


# @spec ING-EP-063
def test_frontend_middleware_sits_inside_cores_authentication(
    frontend, helpers
) -> None:
    from support.app import middleware_stack
    from vllm.entrypoints.serve.middleware.authenticate import (
        AuthenticationMiddleware,
    )

    app = helpers.build(frontend, api_key="secret")
    stack = middleware_stack(app)
    ours = [
        index
        for index, cls in enumerate(stack)
        if cls.__module__.startswith("vllm_riva_frontend")
    ]
    assert len(ours) == 1, stack
    assert AuthenticationMiddleware in stack
    assert stack.index(AuthenticationMiddleware) < ours[0], (
        "core's authentication is outermost"
    )


# @spec ING-EP-084
async def test_api_key_applies_to_frontend_routes_and_realtime_handshake(
    clock, fake_factory, helpers
) -> None:
    from support.app import http_client
    from support.realtime import BOOTSTRAP_PATH, RealtimeClient

    plugin = helpers.make_frontend(
        clock, factories={"nemotron": fake_factory.constructor}
    )
    app, state = await helpers.initialize(
        plugin, helpers.engine_with(), api_key="secret"
    )
    await plugin.start(state)
    try:
        async with http_client(app) as client:
            denied = await client.post(BOOTSTRAP_PATH)
            assert denied.status_code == 401
        async with http_client(app, api_key="secret") as client:
            allowed = await client.post(BOOTSTRAP_PATH)
            assert allowed.status_code == 200
        client = RealtimeClient(app, clock)
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


# @spec ING-EP-005, ING-EP-080
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


# @spec ING-EP-080
async def test_no_owner_is_admitted_before_start(initialized, helpers) -> None:
    plugin, _ = initialized
    run = helpers.stream(plugin, plugin.clock, config_request())
    await asyncio.wait_for(run.task, 2.0)
    assert run.status is grpc.StatusCode.UNAVAILABLE
    assert run.details.startswith("service_unavailable")
    assert run.requests.pulls == 0
    assert len(plugin.registry.records) == 0
    assert plugin.ingress.count == 0


# @spec ING-EP-080
async def test_health_serving_set_last_unless_core_is_draining(
    initialized, helpers
) -> None:
    from vllm_riva_frontend.grpc import RIVA_SERVICE_NAME

    plugin, state = initialized
    state.draining = True  # core began shutting down before start's last step
    await plugin.start(state)
    address = address_of(plugin)
    assert helpers.port_open(int(address.rsplit(":", 1)[1]))
    for service in ("", RIVA_SERVICE_NAME):
        assert await health_status(address, service) == NOT_SERVING, service
    assert plugin.ready(state) is False
    run = helpers.stream(plugin, plugin.clock, config_request())
    await asyncio.wait_for(run.task, 2.0)
    assert run.status is grpc.StatusCode.UNAVAILABLE
    assert run.details.startswith("service_unavailable")
    assert not plugin.registry.records


class _Boom:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("monitor failed")


# @spec ING-EP-064, ING-EP-096, ING-EP-100
async def test_start_rollback_cancels_supervisors_stops_and_settles_factory(
    initialized, fake_factory, grpc_port, monkeypatch, helpers
) -> None:
    import vllm_riva_frontend.lifecycle as lifecycle

    plugin, state = initialized
    before = len(fake_factory.calls)
    monkeypatch.setattr(lifecycle, "ListenerMonitor", _Boom)
    with pytest.raises(RuntimeError, match="monitor failed"):
        await plugin.start(state)
    await asyncio.sleep(0.01)
    assert not helpers.port_open(grpc_port), "server stopped with no grace"
    collector = plugin.fault_collector
    assert collector is not None and collector.done()
    assert collector.cancelled(), "a rolled-back start cancels the collector"
    helpers.factory_settlement(fake_factory, since=before)
    assert fake_factory.teardown_calls == 1
    assert state.request_shutdown.calls == 0, "rollback exit is not a fault"
    assert plugin.fault is None
    assert not plugin.admission_open


# @spec ING-EP-064
async def test_cancelled_start_rollback_is_not_interrupted(
    initialized, fake_factory, monkeypatch, helpers
) -> None:
    import vllm_riva_frontend.lifecycle as lifecycle

    plugin, state = initialized
    clock = plugin.clock
    monkeypatch.setattr(lifecycle, "ListenerMonitor", _Boom)
    block = Block()
    fake_factory.script("drain", block)  # the settlement's drain(math.inf)
    task = asyncio.ensure_future(plugin.start(state))
    await helpers.settle_until(clock, lambda: block.entered.is_set())
    for _ in range(3):
        task.cancel()
        await clock.settle()
    assert not task.done(), "the rollback runs to its end"
    assert fake_factory.teardown_calls == 0
    block.release()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5.0)
    assert fake_factory.teardown_calls == 1
    helpers.factory_settlement(fake_factory)


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


# @spec ING-EP-065
async def test_ready_is_a_synchronous_attribute_read(
    started, fake_engine
) -> None:
    plugin, state = started
    assert not inspect.iscoroutinefunction(type(plugin).ready)
    checks = fake_engine.health_checks
    assert plugin.ready(state) is True
    assert fake_engine.health_checks == checks


# @spec ING-EP-065
async def test_ready_is_false_before_start(initialized) -> None:
    plugin, state = initialized
    assert plugin.ready(state) is False


# @spec ING-EP-065
async def test_ready_is_false_while_draining(started, clock) -> None:
    plugin, state = started
    state.draining = True
    task = asyncio.ensure_future(plugin.drain(state, 60))
    await clock.settle()
    assert plugin.ready(state) is False
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# @spec ING-EP-065
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


# @spec ING-EP-082, ING-EP-065
async def test_core_draining_alone_closes_admission_and_ready_without_d(
    started, fake_factory, test_app, clock, helpers
) -> None:
    from support.asgi import asgi_http

    from vllm_riva_frontend.errors import catalog

    plugin, state = started
    run, lease = await helpers.stream_with_lease(plugin, fake_factory, clock)
    state.draining = True  # core set the flag; no drain call runs
    assert plugin.ready(state) is False
    refused = helpers.stream(plugin, clock, config_request())
    await asyncio.wait_for(refused.task, 2.0)
    assert refused.status is grpc.StatusCode.UNAVAILABLE
    assert refused.details.startswith("service_unavailable")
    body, headers = helpers.file_request_body()
    response = await asgi_http(
        test_app, "POST", helpers.file_route(), headers=headers, body=body
    )
    assert response.status == catalog().http_status("service_unavailable")
    assert plugin.drain_deadline is None, "being draining sets no D"
    # No bound is clipped: the admitted stream keeps its own bounds.
    await clock.advance(30)
    assert not run.done
    run.requests.push(helpers.audio_message())
    run.requests.close()
    await helpers.settle_until(clock, lambda: run.done)
    assert run.status is grpc.StatusCode.OK
    assert lease.names()[-3:] == ["flush", "finish", "release"]


# ---------------------------------------------------- one process, one engine


# @spec ING-EP-061
async def test_one_process_serves_core_routes_namespace_and_one_grpc_listener(
    clock, fake_factory, helpers, grpc_port
) -> None:
    from support.app import (
        CORE_TRANSCRIPTION_ROUTE,
        PLUGIN_PREFIX,
        find_route,
        http_client,
    )
    from support.asgi import AsgiWebSocket
    from support.realtime import CORE_REALTIME_PATH

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
        # Core's /v1/realtime is served by core, unchanged.
        ws = AsgiWebSocket(app, CORE_REALTIME_PATH)
        assert await ws.connect()
        assert (await ws.receive_json())["type"] == "core.realtime"
        core_file = find_route(app, *CORE_TRANSCRIPTION_ROUTE)
        assert core_file is not None
        assert "speech_to_text" in core_file.endpoint.__module__
        assert find_route(app, "GET", f"{PLUGIN_PREFIX}/metadata") is not None
        async with http_client(app) as client:
            live = await client.get(f"{PLUGIN_PREFIX}/health/live")
            assert live.status_code == 200
        assert helpers.port_open(grpc_port)
        assert plugin.ingress.count == 0
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
