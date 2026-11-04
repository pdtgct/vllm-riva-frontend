"""Core endpoint-plugin seam, without a GPU or a vLLM install."""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path
from types import SimpleNamespace

import grpc
import pytest
from grpc_health.v1 import health_pb2, health_pb2_grpc

from vllm_riva_frontend.config import (
    load_plugin_config,
    resolve_config_value,
    validate_frontend_config,
)
from vllm_riva_frontend.lifecycle import RivaFrontendPlugin
from vllm_riva_frontend.session_factory import (
    SESSION_FACTORY_GROUP,
    load_session_factory,
    require_served_cadence,
)
from vllm_riva_frontend.session_factory import (
    importlib_metadata as session_metadata,
)


class _EntryPoint:
    def __init__(self, name: str, target: object) -> None:
        self.name = name
        self._target = target

    def load(self) -> object:
        return self._target


class _Lease:
    async def feed(self, samples: object, *, on_accepted: object) -> list[str]:
        on_accepted(0)  # type: ignore[operator]
        del samples
        return []

    async def update_locale(self, locale: str) -> None:
        del locale

    async def flush(self) -> str:
        return ""

    async def finish(self) -> None:
        return None

    async def abort(self) -> None:
        return None

    async def release(self) -> None:
        return None


class _Factory:
    def __init__(self, cadences: frozenset[str] | None = None) -> None:
        self.opened: list[tuple[str, str]] = []
        if cadences is not None:
            self.served_cadences = cadences

    async def open(self, *, cadence: str, locale: str) -> _Lease:
        self.opened.append((cadence, locale))
        return _Lease()


class _App:
    def __init__(self, state: object) -> None:
        self.state = state
        self.routes: list[object] = []
        self.middleware: list[object] = []
        self.calls: list[object] = []

    def add_middleware(self, middleware: type, **kwargs: object) -> None:
        self.middleware.append(middleware(self, **kwargs))

    async def __call__(
        self, scope: object, receive: object, send: object
    ) -> None:
        del receive
        self.calls.append(scope)
        await send(  # type: ignore[operator]
            {"type": "http.response.start", "status": 299, "headers": []}
        )
        await send(  # type: ignore[operator]
            {"type": "http.response.body", "body": b"", "more_body": False}
        )


def _engine() -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                prompt_dictionary={"auto": "a", "en-US": "b"}
            )
        )
    )


def _state() -> SimpleNamespace:
    return SimpleNamespace(
        openai_serving_models=SimpleNamespace(
            model_name=lambda: "nemotron-asr"
        ),
        draining=False,
        admission=None,
    )


def _install(
    monkeypatch: pytest.MonkeyPatch, entries: list[_EntryPoint]
) -> None:
    def entry_points(*, group: str) -> tuple[_EntryPoint, ...]:
        if group == SESSION_FACTORY_GROUP:
            return tuple(entries)
        return ()

    monkeypatch.setattr(session_metadata, "entry_points", entry_points)


def test_package_registers_the_core_endpoint_group() -> None:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert '[project.entry-points."vllm.endpoint_plugins"]' in text
    discovered = {
        entry.name: entry.value
        for entry in session_metadata.entry_points(
            group="vllm.endpoint_plugins"
        )
    }
    assert discovered["riva_frontend"] == (
        "vllm_riva_frontend:RivaFrontendPlugin"
    )
    assert "vllm_omni" not in "\n".join(
        path.read_text(encoding="utf-8")
        for path in (Path(__file__).parents[1] / "src").rglob("*.py")
    )


def test_fake_session_factory_is_discovered_without_a_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _Factory(frozenset({"560ms", "160ms"}))
    engine = object()

    def build(engine_client: object) -> _Factory:
        assert engine_client is engine
        return factory

    _install(monkeypatch, [_EntryPoint("nemotron_asr", build)])
    loaded = load_session_factory(engine)
    assert loaded is factory
    require_served_cadence(loaded, "160ms")
    with pytest.raises(ValueError, match="served lookahead"):
        require_served_cadence(loaded, "1120ms")


def test_session_factory_discovery_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, [])
    with pytest.raises(LookupError, match="no vllm_riva.session_factories"):
        load_session_factory(object())

    _install(
        monkeypatch,
        [
            _EntryPoint("a", lambda engine: _Factory()),
            _EntryPoint("b", lambda engine: _Factory()),
        ],
    )
    with pytest.raises(LookupError, match="multiple"):
        load_session_factory(object())
    with pytest.raises(LookupError, match="named 'missing'"):
        load_session_factory(object(), name="missing")

    _install(monkeypatch, [_EntryPoint("bad", object())])
    with pytest.raises(TypeError, match="must be callable"):
        load_session_factory(object())

    _install(monkeypatch, [_EntryPoint("bad", lambda engine: object())])
    with pytest.raises(TypeError, match="async open"):
        load_session_factory(object())

    with pytest.raises(TypeError, match="collection"):
        require_served_cadence(
            SimpleNamespace(served_cadences="560ms"), "560ms"
        )
    with pytest.raises(TypeError, match="non-empty"):
        require_served_cadence(SimpleNamespace(served_cadences=[]), "560ms")


def test_config_channel_prefers_additional_config_over_env() -> None:
    args = SimpleNamespace(
        additional_config={"riva_frontend": {"grpc_bind": "127.0.0.1:9"}}
    )
    selected = resolve_config_value(
        args,
        environ={"VLLM_RIVA_FRONTEND_CONFIG": '{"grpc_bind": "127.0.0.1:8"}'},
    )
    frontend, _metadata = load_plugin_config(selected)
    assert frontend.grpc_bind == "127.0.0.1:9"

    path_args = SimpleNamespace(
        additional_config={"riva_frontend": '{"streaming_cadence": "160ms"}'}
    )
    inline = resolve_config_value(path_args, environ={})
    assert load_plugin_config(inline)[0].streaming_cadence == "160ms"

    from_env = resolve_config_value(
        SimpleNamespace(additional_config={}),
        environ={"VLLM_RIVA_FRONTEND_CONFIG": "  "},
    )
    assert from_env is None
    assert resolve_config_value(None, environ={}) is None

    with pytest.raises(ValueError, match="additional-config"):
        resolve_config_value(
            SimpleNamespace(additional_config={"riva_frontend": 5}),
            environ={},
        )
    with pytest.raises(ValueError, match="streaming_cadence"):
        validate_frontend_config(
            load_plugin_config('{"streaming_cadence": "nope"}')[0]
        )


def test_plugin_lifecycle_with_fake_factory_needs_no_gpu() -> None:
    """Bind real gRPC, serve health, then drain and stop. No engine, no GPU."""
    factory = _Factory(frozenset({"160ms", "560ms"}))
    seen: list[object] = []

    def build(engine_client: object, *, name: str | None = None) -> _Factory:
        seen.append((engine_client, name))
        return factory

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    async def exercise() -> None:
        state = _state()
        app = _App(state)
        plugin = RivaFrontendPlugin(
            version_resolver=lambda: "0.33.0rc3",
            session_loader=build,
        )
        assert plugin.ready(state) is False
        await plugin.drain(state, 0.1)
        await plugin.stop(state)
        with pytest.raises(RuntimeError, match="attach_router"):
            await plugin.init_state(_engine(), state, SimpleNamespace())
        plugin.attach_router(app)
        await plugin.init_state(
            _engine(),
            state,
            SimpleNamespace(
                api_server_count=1,
                additional_config={
                    "riva_frontend": {
                        "grpc_bind": f"127.0.0.1:{port}",
                        "streaming_cadence": "160ms",
                    }
                },
            ),
        )
        assert seen and seen[0][1] is None
        assert plugin.ready(state) is False
        await plugin.start(state)
        assert plugin.ready(state) is True

        channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
        stub = health_pb2_grpc.HealthStub(channel)
        response = await stub.Check(
            health_pb2.HealthCheckRequest(service=""), timeout=5
        )
        assert response.status == health_pb2.HealthCheckResponse.SERVING
        await channel.close()

        middleware = app.middleware[0]

        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": b""}

        sent: list[dict[str, object]] = []

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        await middleware(  # type: ignore[operator]
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/audio/transcriptions",
                "headers": [],
            },
            receive,
            send,
        )
        assert sent[0]["status"] == 299
        sent.clear()
        await middleware(  # type: ignore[operator]
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/realtime/transcription_sessions",
            },
            receive,
            send,
        )
        assert sent[0]["status"] == 200

        state.draining = True
        assert plugin.ready(state) is False
        await plugin.drain(state, 1.0)
        await plugin.stop(state)
        assert factory.opened == []

    asyncio.run(exercise())


def test_attach_router_requires_middleware_and_cadence_must_be_served() -> None:
    plugin = RivaFrontendPlugin(version_resolver=lambda: "0.33.0")
    with pytest.raises(TypeError, match="add_middleware"):
        plugin.attach_router(SimpleNamespace())

    def build(engine_client: object, *, name: str | None = None) -> _Factory:
        del engine_client, name
        return _Factory(frozenset({"560ms"}))

    async def exercise() -> None:
        state = _state()
        app = _App(state)
        gated = RivaFrontendPlugin(
            version_resolver=lambda: "0.33.0",
            session_loader=build,
        )
        gated.attach_router(app)
        with pytest.raises(ValueError, match="served lookahead"):
            await gated.init_state(
                _engine(),
                state,
                SimpleNamespace(
                    additional_config={
                        "riva_frontend": {"streaming_cadence": "160ms"}
                    }
                ),
            )

    asyncio.run(exercise())
