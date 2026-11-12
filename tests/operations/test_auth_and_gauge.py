"""Path-scoped authentication of ``/plugins/riva/`` and core's WebSocket gauge.

The app is built by ``support.app.build_app``; ``api_key`` installs core's
``AuthenticationMiddleware`` as ``--api-key`` does. Core's middleware only
guards its own prefixes, so the frontend adds a path-scoped middleware in
``attach_router`` that checks every request under ``/plugins/riva/`` with
the ``verify_token`` of a core ``AuthenticationMiddleware`` built in
``init_state``. Core's WebSocket metrics middleware wraps it.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
from typing import Any
from unittest import mock

import pytest
from support.app import (
    BOOTSTRAP_ROUTE,
    FILE_ROUTE,
    OPERATIONAL_ROUTES,
    PLUGIN_PREFIX,
    auth_registrations,
    build_app,
    http_client,
    init_state,
    start_plugins,
    stop_plugins,
)
from support.asgi import AsgiWebSocket
from support.realtime import REALTIME_PATH, REALTIME_URL
from support.state import make_args

TOKEN = "s3cret-token"
OTHER = "0ther-token"
UNAUTHORIZED = {"error": "Unauthorized"}
UNAVAILABLE = {"error": "Service Unavailable"}
GAUGE = "vllm:websocket_connections_active"
TOTAL = "vllm:websocket_connections_total"
VERSION_PATH = f"{PLUGIN_PREFIX}/version"
METADATA_PATH = f"{PLUGIN_PREFIX}/metadata"
UNKNOWN_PATH = f"{PLUGIN_PREFIX}/no-such-route"
BOOTSTRAP_PATH = BOOTSTRAP_ROUTE[1]
FRONTEND_HTTP = [*OPERATIONAL_ROUTES, BOOTSTRAP_ROUTE, FILE_ROUTE]


@pytest.fixture
def test_app(plugins, supported_tasks):
    return build_app(plugins, supported_tasks=supported_tasks, api_key=TOKEN)


def _client(app, token=None, **kwargs):
    return http_client(app, api_key=token, **kwargs)


def _bearer(token: str = TOKEN) -> list[tuple[str, str]]:
    return [("authorization", f"Bearer {token}")]


async def _init(app, engine, models, transcription):
    return await init_state(
        app, engine, models=models, transcription=transcription
    )


async def _request(client, method: str, path: str):
    if method == "POST":
        return await client.post(path, json={})
    return await client.request(method, path)


def _frontend_middleware(app, inner):
    """An instance of the middleware the frontend added, around ``inner``."""
    ours = [
        m
        for m in app.user_middleware
        if m.cls.__module__.startswith("vllm_riva_frontend")
    ]
    assert len(ours) == 1, [m.cls for m in app.user_middleware]
    entry = ours[0]
    return entry.cls(inner, *entry.args, **entry.kwargs)


class _Inner:
    """An ASGI app that records the calls the middleware passes through."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any, Any]] = []

    async def __call__(self, scope, receive, send) -> None:
        self.calls.append((scope, receive, send))


def _scope(kind: str, path: str, *, method="GET", headers=(), ext=False):
    scope: dict[str, Any] = {
        "type": kind,
        "asgi": {"version": "3.0"},
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"intent=transcription" if kind == "websocket" else b"",
        "headers": [(k.encode(), v.encode()) for k, v in headers],
    }
    if kind == "http":
        scope["method"] = method
    if ext:
        scope["extensions"] = {"websocket.http.response": {}}
    return scope


# --------------------------------------------------------------- scope (083)


# @spec ING-EP-083
async def test_path_scoped_middleware_scope_rules(app_state, test_app) -> None:
    async with _client(test_app) as anonymous:
        in_scope = await anonymous.get(VERSION_PATH)
        slash = await anonymous.get("/plugins/riva/")
        bare = await anonymous.get("/plugins/riva")
        sibling = await anonymous.get("/plugins/rivax/v1/version")
        options = await anonymous.options(VERSION_PATH)
        core_health = await anonymous.get("/health")
    assert in_scope.status_code == 401
    assert in_scope.json() == UNAUTHORIZED
    assert slash.status_code == 401
    assert bare.status_code == 404
    assert sibling.status_code == 404
    assert options.status_code != 401
    assert core_health.status_code == 200

    # A sibling WebSocket path is not refused by the namespace's check.
    ws = AsgiWebSocket(
        test_app, "/plugins/rivax/v1/realtime", denial_response=True
    )
    assert await ws.connect() is False
    assert ws.denied is None


# @spec ING-EP-083
async def test_scope_rules_remove_root_path_as_core_does(
    plugin, fake_engine, fake_models, fake_transcription
) -> None:
    app = build_app([plugin], api_key=TOKEN, root_path="/proxy")
    await _init(app, fake_engine, fake_models, fake_transcription)
    async with _client(app, root_path="/proxy") as anonymous:
        refused = await anonymous.get(f"/proxy{VERSION_PATH}")
        sibling = await anonymous.get("/proxy/plugins/rivax/v1/version")
    async with _client(app, TOKEN, root_path="/proxy") as authorized:
        allowed = await authorized.get(f"/proxy{VERSION_PATH}")
    assert refused.status_code == 401
    assert refused.json() == UNAUTHORIZED
    assert sibling.status_code == 404
    assert allowed.status_code == 200

    ws = AsgiWebSocket(app, denial_response=True, root_path="/proxy")
    assert await ws.connect() is False
    assert (await ws.read_denial())[0] == 401


# @spec ING-EP-083
async def test_other_scope_types_pass_through_unchanged(
    app_state, test_app
) -> None:
    inner = _Inner()
    middleware = _frontend_middleware(test_app, inner)

    async def receive():
        return {"type": "lifespan.startup"}

    async def send(message):
        raise AssertionError(f"the middleware sent {message}")

    passing = [
        {"type": "lifespan", "asgi": {"version": "3.0"}},
        _scope("http", "/plugins/riva"),
        _scope("http", "/plugins/rivax/v1/version"),
        _scope("http", "/v1/models"),
        _scope("http", VERSION_PATH, method="OPTIONS"),
        _scope("websocket", "/v1/realtime"),
    ]
    for scope in passing:
        await middleware(scope, receive, send)
    assert [call[0] for call in inner.calls] == passing
    for (scope, got_receive, got_send), expected in zip(
        inner.calls, passing, strict=True
    ):
        assert scope is expected
        assert got_receive is receive
        assert got_send is send


# ---------------------------------------------------------- the check (084)


# @spec ING-EP-084
async def test_in_scope_requests_checked_with_cores_verify_token_and_unknown_path_401(  # noqa: E501
    started, test_app
) -> None:
    from vllm.entrypoints.serve.middleware.authenticate import (
        AuthenticationMiddleware,
    )

    checked: list[Any] = []
    original = AuthenticationMiddleware.verify_token

    def recording(self, headers):
        checked.append(self)
        return original(self, headers)

    with mock.patch.object(AuthenticationMiddleware, "verify_token", recording):
        async with _client(test_app) as anonymous:
            unknown = await anonymous.get(UNKNOWN_PATH)
        assert checked, "the namespace check uses core's verify_token"
        async with _client(test_app, TOKEN) as authorized:
            routed = await authorized.get(UNKNOWN_PATH)
            served = await authorized.get(VERSION_PATH)
        async with _client(test_app, OTHER) as wrong:
            refused = await wrong.get(VERSION_PATH)
        ws = AsgiWebSocket(test_app, headers=_bearer())
        count = len(checked)
        assert await ws.connect() is True
        assert len(checked) > count
        await ws.disconnect()
        await ws.wait_closed()

    assert unknown.status_code == 401
    assert unknown.json() == UNAUTHORIZED
    assert routed.status_code == 404  # authenticated, then routing answers
    assert served.status_code == 200
    assert refused.status_code == 401
    expected = [hashlib.sha256(TOKEN.encode()).digest()]
    for verifier in checked:
        assert isinstance(verifier, AuthenticationMiddleware)
        assert verifier.api_tokens == expected


# @spec ING-EP-084
async def test_auth_off_passes_everything(
    plugin, fake_engine, fake_models, fake_transcription, monkeypatch
) -> None:
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    app = build_app([plugin])
    state = await _init(app, fake_engine, fake_models, fake_transcription)
    await start_plugins(state)
    try:
        async with _client(app) as anonymous:
            version = await anonymous.get(VERSION_PATH)
            unknown = await anonymous.get(UNKNOWN_PATH)
            bootstrap = await anonymous.post(BOOTSTRAP_PATH, json={})
        ws = AsgiWebSocket(app)
        assert await ws.connect() is True
        await ws.disconnect()
        await ws.wait_closed()
    finally:
        await stop_plugins(state)
    assert version.status_code == 200
    assert unknown.status_code == 404
    assert bootstrap.status_code == 200


# @spec ING-EP-084, ING-EP-087, ING-EP-088
@pytest.mark.parametrize("resolved", [True, False], ids=["401", "503"])
async def test_refusals_are_awaited_inline_starting_no_task(
    resolved,
    plugin,
    fake_engine,
    fake_models,
    fake_transcription,
) -> None:
    app = build_app([plugin], api_key=TOKEN)
    if resolved:
        await _init(app, fake_engine, fake_models, fake_transcription)
    inner = _Inner()
    middleware = _frontend_middleware(app, inner)
    status = 401 if resolved else 503
    loop = asyncio.get_running_loop()

    async def receive():
        return {"type": "websocket.connect"}

    for scope in (
        _scope("http", VERSION_PATH),
        _scope("websocket", REALTIME_PATH, ext=True),
        _scope("websocket", REALTIME_PATH),
    ):
        sent: list[dict[str, Any]] = []

        async def send(message, sent=sent):
            sent.append(message)

        with mock.patch.object(
            loop,
            "create_task",
            side_effect=AssertionError("the refusal started a task"),
        ):
            await middleware(scope, receive, send)
        # Every message of the refusal was sent before the call returned.
        assert inner.calls == []
        assert sent, scope["type"]
        kinds = [m["type"] for m in sent]
        assert "websocket.accept" not in kinds
        if scope["type"] == "http":
            assert sent[0]["status"] == status
            assert not sent[-1].get("more_body", False)
        elif "extensions" in scope:
            assert sent[0]["type"] == "websocket.http.response.start"
            assert sent[0]["status"] == status
            assert not sent[-1].get("more_body", False)
        else:
            assert kinds == ["websocket.close"]
    if resolved:
        assert not plugin.registry.records


# ------------------------------------------------------ selection (085, 086)


def _app_for(plugin, api_key, env, monkeypatch, registrations=()):
    if env is None:
        monkeypatch.delenv("VLLM_API_KEY", raising=False)
    else:
        monkeypatch.setenv("VLLM_API_KEY", env)
    return build_app(
        [plugin],
        args=make_args(api_key=api_key),
        auth_registrations=list(registrations),
    )


# @spec ING-EP-085
@pytest.mark.parametrize(
    ("api_key", "env", "token"),
    [
        ([""], "envkey", None),
        (None, "envkey", "envkey"),
        ([], "envkey", "envkey"),
        (["k1", ""], None, "k1"),
        (["k1"], "envkey", "k1"),
        (None, None, None),
        (None, "", None),
    ],
)
async def test_api_key_selection_follows_cores_rule(
    api_key,
    env,
    token,
    plugin,
    fake_engine,
    fake_models,
    fake_transcription,
    monkeypatch,
) -> None:
    app = _app_for(plugin, api_key, env, monkeypatch)
    await _init(app, fake_engine, fake_models, fake_transcription)
    async with _client(app) as anonymous:
        response = await anonymous.get(VERSION_PATH)
    if token is None:
        assert response.status_code == 200
        return
    assert response.status_code == 401
    async with _client(app, token) as authorized:
        assert (await authorized.get(VERSION_PATH)).status_code == 200


# @spec ING-EP-085
async def test_empty_cli_key_beats_the_environment_key(
    plugin, fake_engine, fake_models, fake_transcription, monkeypatch
) -> None:
    """``--api-key ""`` with ``VLLM_API_KEY`` set configures no key.

    A registration then has no key to match, so startup fails; a rule that
    fell back to the environment would turn authentication on instead.
    """
    from vllm_riva_frontend.lifecycle import StartupError

    app = _app_for(
        plugin, [""], "envkey", monkeypatch, [{"tokens": ["envkey"]}]
    )
    with pytest.raises(StartupError):
        await _init(app, fake_engine, fake_models, fake_transcription)


#: (case, key configured, extra registrations, core installs its own)
FAILING = [
    ("key-without-registration", True, [], False),
    ("registration-without-key", False, [{"tokens": [TOKEN]}], False),
    ("two-registrations", True, [{"tokens": [TOKEN]}], True),
    ("empty-token-list", True, [{"tokens": []}], False),
    ("empty-token", True, [{"tokens": [""]}], False),
    ("one-empty-among", True, [{"tokens": [TOKEN, ""]}], False),
    ("no-tokens-keyword", True, [{}], False),
    ("tokens-not-a-list", True, [{"tokens": TOKEN}], False),
    ("token-not-a-string", True, [{"tokens": [7]}], False),
]


# @spec ING-EP-086
@pytest.mark.parametrize(
    ("case", "key", "extra", "core"), FAILING, ids=[c[0] for c in FAILING]
)
async def test_auth_decision_matrix_fails_startup(
    case,
    key,
    extra,
    core,
    plugin,
    fake_engine,
    fake_models,
    fake_transcription,
    monkeypatch,
    caplog,
) -> None:
    from vllm_riva_frontend.lifecycle import StartupError

    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    app = build_app(
        [plugin],
        args=make_args(api_key=[TOKEN] if core else None),
        auth_registrations=extra,
    )
    if key:
        # The key is configured after core registered its middleware (or
        # did not), so the two signals disagree.
        app.state.args.api_key = [TOKEN]
    caplog.set_level(logging.DEBUG)
    with pytest.raises(StartupError) as raised:
        await _init(app, fake_engine, fake_models, fake_transcription)
    message = str(raised.value)
    assert "AuthenticationMiddleware" in message, message
    assert any(
        name in message for name in ("api_key", "api-key", "VLLM_API_KEY")
    ), message
    assert TOKEN not in message
    assert TOKEN not in caplog.text


# @spec ING-EP-086
@pytest.mark.parametrize("key", [False, True], ids=["off", "on"])
async def test_auth_decision_matrix(
    key,
    plugin,
    fake_engine,
    fake_models,
    fake_transcription,
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    app = build_app([plugin], api_key=TOKEN if key else None)
    assert len(auth_registrations(app)) == (1 if key else 0)
    caplog.set_level(logging.DEBUG)
    await _init(app, fake_engine, fake_models, fake_transcription)
    async with _client(app) as anonymous:
        response = await anonymous.get(VERSION_PATH)
    assert response.status_code == (401 if key else 200)
    if key:
        async with _client(app, TOKEN) as authorized:
            assert (await authorized.get(VERSION_PATH)).status_code == 200
    assert TOKEN not in caplog.text


# -------------------------------------------------- WebSocket refusal (087)


# @spec ING-EP-087
@pytest.mark.parametrize("token", [None, OTHER], ids=["missing", "wrong"])
async def test_failed_ws_handshake_denied_with_extension_else_closed_before_accept(  # noqa: E501
    token, started, test_app, fake_factory
) -> None:
    headers = _bearer(token) if token else []
    denied = AsgiWebSocket(test_app, headers=headers, denial_response=True)
    assert await denied.connect() is False
    assert await denied.read_denial() == (401, UNAUTHORIZED)
    assert not denied.accepted

    closed = AsgiWebSocket(test_app, headers=headers)
    assert await closed.connect() is False
    assert closed.refused_before_accept
    assert closed.denied is None
    assert not closed.accepted
    assert fake_factory.leases == []


# @spec ING-EP-087, ING-SHIM-021
async def test_ws_handshake_with_token_is_served(started, test_app) -> None:
    ws = AsgiWebSocket(test_app, headers=_bearer())
    assert await ws.connect() is True
    event = await ws.receive_json()
    assert event["type"] == "conversation.created"
    await ws.disconnect()
    await ws.wait_closed()


# --------------------------------------------------- fail closed (088)


# @spec ING-EP-088
@pytest.mark.parametrize("key", [True, False], ids=["key", "no-key"])
async def test_unresolved_auth_fails_closed_503_http_and_ws(
    key, plugin, monkeypatch
) -> None:
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    app = build_app([plugin], api_key=TOKEN if key else None)
    # No init_state: the tokens are not resolved yet.
    async with _client(app, TOKEN) as authorized:
        with_token = await authorized.get(VERSION_PATH)
    async with _client(app) as anonymous:
        without = await anonymous.get(VERSION_PATH)
        unknown = await anonymous.get(UNKNOWN_PATH)
        outside = await anonymous.get("/no-such-core-route")
    for response in (with_token, without, unknown):
        assert response.status_code == 503
        assert response.json() == UNAVAILABLE
    assert outside.status_code == 404

    denied = AsgiWebSocket(app, headers=_bearer(), denial_response=True)
    assert await denied.connect() is False
    assert await denied.read_denial() == (503, UNAVAILABLE)
    closed = AsgiWebSocket(app, headers=_bearer())
    assert await closed.connect() is False
    assert closed.refused_before_accept
    assert closed.denied is None


# --------------------------------------------------------- core pins (089)


# @spec ING-EP-089
def test_core_auth_internals_are_pinned() -> None:
    from starlette.datastructures import Headers
    from vllm.entrypoints.serve.middleware.authenticate import (
        AuthenticationMiddleware,
    )

    parameters = inspect.signature(AuthenticationMiddleware).parameters
    assert "tokens" in parameters

    async def inner(scope, receive, send):
        raise AssertionError("not reached")

    verifier = AuthenticationMiddleware(inner, tokens=[TOKEN])

    def check(value: str | None) -> bool:
        raw = [] if value is None else [(b"authorization", value.encode())]
        return verifier.verify_token(Headers(raw=raw))

    assert check(f"Bearer {TOKEN}") is True
    assert check(f"bearer {TOKEN}") is True
    assert check(f"Bearer {OTHER}") is False
    assert check(f"Basic {TOKEN}") is False
    assert check(None) is False

    app = build_app(api_key=TOKEN)
    assert auth_registrations(app) == [{"tokens": [TOKEN]}]


# @spec ING-EP-089
async def test_core_401_body_is_pinned() -> None:
    from support.asgi import asgi_http
    from vllm.entrypoints.serve.middleware.authenticate import (
        AuthenticationMiddleware,
    )

    async def inner(scope, receive, send):
        raise AssertionError("not reached")

    middleware = AuthenticationMiddleware(inner, tokens=[TOKEN])
    response = await asgi_http(middleware, "GET", "/v1/models")
    assert response.status == 401
    assert response.json() == UNAUTHORIZED


# --------------------------------------------------------------- gRPC (090)


async def _grpc_config(fake_engine) -> list[str]:
    from riva.client.proto import riva_asr_pb2, riva_asr_pb2_grpc
    from support.grpc_harness import aio_channel

    address = fake_engine.additional_config["riva_frontend"]["grpc_bind"]
    async with aio_channel(address) as channel:
        await asyncio.wait_for(channel.channel_ready(), 5.0)
        stub = riva_asr_pb2_grpc.RivaSpeechRecognitionStub(channel)
        reply = await stub.GetRivaSpeechRecognitionConfig(
            riva_asr_pb2.RivaSpeechRecognitionConfigRequest(), timeout=5.0
        )
    return [c.model_name for c in reply.model_config]


# @spec ING-EP-090
async def test_grpc_listener_ignores_api_key(
    plugin, fake_engine, fake_models, fake_transcription, monkeypatch
) -> None:
    monkeypatch.setenv("VLLM_API_KEY", OTHER)
    app = build_app([plugin], api_key=TOKEN)
    state = await _init(app, fake_engine, fake_models, fake_transcription)
    await start_plugins(state)
    try:
        # No metadata at all: the gRPC listener is outside --api-key.
        assert await _grpc_config(fake_engine)
    finally:
        await stop_plugins(state)


# ---------------------------------------------------- frontend routes (021)


# @spec ING-SHIM-021
async def test_every_frontend_route_requires_token_core_routes_unchanged_grpc_exempt(  # noqa: E501
    started, test_app, fake_engine
) -> None:
    async with _client(test_app) as anonymous:
        for method, path in FRONTEND_HTTP:
            refused = await _request(anonymous, method, path)
            assert refused.status_code == 401, (method, path)
            assert refused.json() == UNAUTHORIZED, (method, path)
            options = await anonymous.options(path)
            assert options.status_code != 401, (method, path)
        core_health = await anonymous.get("/health")
        core_ready = await anonymous.get("/ready")
        core_models = await anonymous.get("/v1/models")
    async with _client(test_app, TOKEN) as authorized:
        for method, path in FRONTEND_HTTP:
            allowed = await _request(authorized, method, path)
            assert allowed.status_code != 401, (method, path)
            if (method, path) != FILE_ROUTE:
                assert allowed.status_code == 200, (method, path)
        core_models_authorized = await authorized.get("/v1/models")
    assert core_health.status_code == 200
    assert core_ready.status_code == 200
    assert core_models.status_code == 401  # core's own check, unchanged
    assert core_models_authorized.status_code != 401

    ws = AsgiWebSocket(test_app)
    assert await ws.connect() is False
    assert ws.refused_before_accept
    assert await _grpc_config(fake_engine)


# ---------------------------------------------------------- the gauge (022)


def _sample(name: str) -> float:
    from prometheus_client import REGISTRY

    value = REGISTRY.get_sample_value(name)
    return 0.0 if value is None else value


# @spec ING-SHIM-022
async def test_ws_gauge_counts_open_sessions_and_accepted_refusals_not_query_refusals(  # noqa: E501
    started, app_state, test_app
) -> None:
    active, total = _sample(GAUGE), _sample(TOTAL)

    # An open Riva Realtime session is counted while it is open.
    async with _client(test_app, TOKEN) as authorized:
        bootstrap = await authorized.post(BOOTSTRAP_PATH, json={})
    assert bootstrap.status_code == 200
    ws = AsgiWebSocket(test_app, REALTIME_URL, headers=_bearer())
    assert await ws.connect()
    assert (await ws.receive_json())["type"] == "conversation.created"
    await ws.send_json(
        {"type": "transcription_session.update", "session": bootstrap.json()}
    )
    assert (await ws.receive_json())["type"] == "transcription_session.updated"
    assert _sample(GAUGE) == active + 1
    assert _sample(TOTAL) == total + 1
    await ws.send_json({"type": "input_audio_buffer.done"})
    while ws.close_code is None:
        await ws.receive()
    await ws.wait_closed()
    assert ws.close_code == 1000
    assert _sample(GAUGE) == active

    # A query refusal is never accepted, so it is never counted.
    query = AsgiWebSocket(
        test_app, f"{REALTIME_PATH}?intent=other", headers=_bearer()
    )
    assert await query.connect() is False
    await query.wait_closed()
    assert _sample(TOTAL) == total + 1
    assert _sample(GAUGE) == active

    # A draining refusal is accepted before it closes, so it is counted.
    app_state.draining = True
    refused = AsgiWebSocket(test_app, REALTIME_URL, headers=_bearer())
    assert await refused.connect() is True
    event = await refused.receive_json()
    assert event["type"] == "error"
    while refused.close_code is None:
        await refused.receive()
    await refused.wait_closed()
    assert refused.close_code == 1013
    assert _sample(TOTAL) == total + 2
    assert _sample(GAUGE) == active


# ----------------------------------------------- evaluation discipline (006)


# @spec ING-CLI-006
async def test_bearer_token_behaviour_with_and_without_token(
    started, test_app
) -> None:
    async with _client(test_app) as anonymous:
        metadata = await anonymous.get(METADATA_PATH)
        bootstrap = await anonymous.post(BOOTSTRAP_PATH, json={})
    async with _client(test_app, TOKEN) as authorized:
        metadata_ok = await authorized.get(METADATA_PATH)
        bootstrap_ok = await authorized.post(BOOTSTRAP_PATH, json={})
    assert (metadata.status_code, bootstrap.status_code) == (401, 401)
    assert (metadata_ok.status_code, bootstrap_ok.status_code) == (200, 200)

    without = AsgiWebSocket(test_app)
    assert await without.connect() is False
    with_token = AsgiWebSocket(test_app, headers=_bearer())
    assert await with_token.connect() is True
    await with_token.disconnect()
    await with_token.wait_closed()
