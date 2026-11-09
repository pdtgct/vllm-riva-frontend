"""The Realtime dispatcher: claim rule, pass-through and the receive caps."""

from __future__ import annotations

import pytest
from support.app import build_app, route_table
from support.asgi import AsgiWebSocket
from support.realtime import RealtimeClient, error_code

from .conftest import riva_options, served_lease, serving


def _scope(
    kind: str = "websocket",
    path: str = "/v1/realtime",
    query: bytes = b"intent=transcription",
    root_path: str = "",
) -> dict:
    return {
        "type": kind,
        "path": path,
        "raw_path": path.encode(),
        "root_path": root_path,
        "query_string": query,
        "headers": [(b"host", b"testserver")],
    }


CLAIMED = [
    _scope(),
    _scope(query=b"intent=%74ranscription"),
    _scope(query=b"%69ntent=transcription"),
    _scope(root_path="/api", path="/api/v1/realtime"),
]

PASSED = [
    _scope(kind="http"),
    _scope(kind="lifespan"),
    _scope(path="/v1/realtime/"),
    _scope(path="/v1/realtimes"),
    _scope(path="/v1/realtime/transcription_sessions"),
    _scope(query=b""),
    _scope(query=b"intent="),
    _scope(query=b"intent"),
    _scope(query=b"intent=conversation"),
    _scope(query=b"intent=Transcription"),
    _scope(query=b"intent=transcription&intent=transcription"),
    _scope(query=b"intent=transcription&model=x"),
    _scope(query=b"model=x&intent=transcription"),
    _scope(query=b"intent=transcription%ZZ"),
    _scope(query=b"intent=%FFtranscription"),
    _scope(query=b"intent=transcription%"),
]


@pytest.mark.parametrize(
    "scope", CLAIMED, ids=lambda s: repr(s["query_string"])
)
# @spec ING-NIMWS-011
def test_only_the_exact_transcription_intent_is_claimed(scope):
    from vllm_riva_frontend.nim_ws import DispatchDecision, dispatch_decision

    assert dispatch_decision(scope) == DispatchDecision.CLAIM


@pytest.mark.parametrize(
    "scope",
    PASSED,
    ids=lambda s: f"{s['type']}:{s['path']}?{s['query_string']!r}",
)
# @spec ING-NIMWS-011
def test_every_other_scope_passes_to_core(scope):
    from vllm_riva_frontend.nim_ws import DispatchDecision, dispatch_decision

    assert dispatch_decision(scope) == DispatchDecision.PASS


@pytest.mark.parametrize(
    "url",
    [
        "/v1/realtime",
        "/v1/realtime?intent=conversation",
        "/v1/realtime?intent=transcription&intent=transcription",
        "/v1/realtime?intent=transcription&extra=1",
        "/v1/realtime?intent=%FF",
    ],
)
# @spec ING-NIMWS-011
# @spec ING-NIMWS-019
async def test_passed_scopes_reach_core_unchanged(served, url):
    socket = AsgiWebSocket(served.app, url, headers=[("x-probe", "kept")])
    assert await socket.connect()
    assert await socket.receive_json() == {"type": "core.realtime"}
    await socket.receive()
    assert socket.close_code == 1000
    (seen,) = served.app.state.core_realtime_scopes
    assert seen["path"] == "/v1/realtime"
    assert seen["query_string"] == socket.scope["query_string"]
    assert (b"x-probe", b"kept") in seen["headers"]
    assert not served.plugin.registry.records
    assert served.factory.leases == []


# @spec ING-NIMWS-011
async def test_a_claimed_scope_is_served_by_the_adapter_not_core(served):
    client = RealtimeClient(served.app, served.clock)
    assert await client.connect()
    first = await client.next()
    assert first["type"] == "conversation.created"
    assert not getattr(served.app.state, "core_realtime_scopes", [])


# @spec ING-NIMWS-019
async def test_dispatcher_does_not_alter_cores_routes(clock):
    core = route_table(
        build_app([], supported_tasks=("transcription", "realtime"))
    )
    async with serving(clock=clock) as served:
        with_plugin = route_table(served.app)
    core_paths = [(path, methods) for path, methods, _ in core]
    remaining = [
        (path, methods)
        for path, methods, _ in with_plugin
        if (path, methods) in set(core_paths)
    ]
    assert remaining == core_paths
    core_endpoints = {(p, m): e for p, m, e in core}
    for path, methods, endpoint in with_plugin:
        if (
            path,
            methods,
        ) in core_endpoints and path != "/v1/audio/transcriptions":
            assert endpoint is core_endpoints[(path, methods)], path
    realtime = [
        route
        for route in served.app.routes
        if getattr(route, "path", None) == "/v1/realtime"
    ]
    assert len(realtime) == 1


# ------------------------------------------------------------ receive caps


def _cap_options(**overrides):
    return riva_options(
        ws_event_envelope_max_bytes=1024,
        ws_receive_max_bytes=4096,
        **overrides,
    )


# @spec ING-NIMWS-012
async def test_init_state_requires_receive_cap_at_least_the_envelope(clock):
    options = riva_options(
        ws_event_envelope_max_bytes=4096, ws_receive_max_bytes=1024
    )
    with pytest.raises(Exception) as caught:
        async with serving(clock=clock, options=options, start=False):
            pass
    text = " ".join(
        [str(caught.value), *map(str, getattr(caught.value, "failures", []))]
    )
    assert "ws_receive_max_bytes" in text


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("binary", [False, True])
# @spec ING-NIMWS-012
# @spec ING-ERR-026
async def test_a_message_above_the_receive_cap_closes_1009(
    clock, configured, binary
):
    async with serving(clock=clock, options=_cap_options()) as served:
        client = RealtimeClient(served.app, served.clock)
        if configured:
            await client.open_session()
        else:
            assert await client.connect()
            assert (await client.next())["type"] == "conversation.created"
        payload = b"{" + b" " * 5000 + b"}"
        if binary:
            await client.ws.send_bytes(payload)
        else:
            await client.ws.send_text(payload.decode())
        before = await client.assert_closes(1009)
        errors = [e for e in before if e["type"] == "error"]
        assert [error_code(e) for e in errors] == ["request_too_large"]
        if not configured:
            assert served.factory.leases == []


# @spec ING-NIMWS-015
# @spec ING-ERR-026
async def test_an_event_above_the_envelope_is_rejected_before_any_decode(clock):
    async with serving(clock=clock, options=_cap_options()) as served:
        client = RealtimeClient(served.app, served.clock)
        await client.open_session()
        lease = served_lease(served)
        # Not JSON at all: a decode would report ``invalid_event`` instead.
        await client.ws.send_text("x" * 2000)
        before = await client.assert_closes(1009)
        assert [error_code(e) for e in before if e["type"] == "error"] == [
            "request_too_large"
        ]
        assert lease.accepted == []


# @spec ING-NIMWS-015
async def test_an_event_within_the_envelope_is_decoded(clock):
    async with serving(clock=clock, options=_cap_options()) as served:
        client = RealtimeClient(served.app, served.clock)
        await client.open_session()
        await client.ws.send_text("x" * 900)
        before = await client.assert_closes(1008)
        assert [error_code(e) for e in before if e["type"] == "error"] == [
            "invalid_event"
        ]
