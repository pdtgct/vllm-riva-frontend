"""The Realtime WebSocket route: its query rule, core's route and the caps.

The frontend serves the Riva Realtime WebSocket on its own route,
``/plugins/riva/v1/realtime``, and leaves core's ``/v1/realtime`` alone. The
route's first step is the query check: only the exact query
``intent=transcription`` (after strict percent and UTF-8 decoding) is
served; any other query is closed with 1008 before ``websocket.accept``.
"""

from __future__ import annotations

import pytest
from support.app import CORE_REALTIME_ROUTE, all_routes, core_realtime_stand_in
from support.asgi import AsgiWebSocket
from support.realtime import (
    CORE_REALTIME_PATH,
    REALTIME_PATH,
    RealtimeClient,
    error_code,
)

from .conftest import riva_options, served_lease, serving

#: Queries the route serves: each decodes to exactly intent=transcription.
SERVED_QUERIES = [
    "intent=transcription",
    "intent=%74ranscription",
    "%69ntent=transcription",
    "intent=transcriptio%6E",
]

#: Queries the route refuses before accept (ING-NIMWS-027).
REFUSED_QUERIES = [
    "",
    "intent=",
    "intent",
    "=transcription",
    "intent=conversation",
    "intent=Transcription",
    "Intent=transcription",
    "intent=transcription&intent=transcription",
    "intent=transcription&model=x",
    "model=x&intent=transcription",
    "intent=transcription%ZZ",
    "intent=%FFtranscription",
    "intent=transcription%",
    "intent=%C3%28",
]


def _url(query: str) -> str:
    return f"{REALTIME_PATH}?{query}" if query else REALTIME_PATH


def _sends(socket: AsgiWebSocket) -> list[dict]:
    return [m for m in socket.received if m["type"] == "websocket.send"]


# ------------------------------------------------------------ query rule


@pytest.mark.parametrize("query", SERVED_QUERIES)
# @spec ING-NIMWS-026
async def test_route_serves_only_exact_intent_transcription(served, query):
    client = RealtimeClient(served.app, served.clock, url=_url(query))
    assert await client.connect(), client.ws.received
    first = await client.next()
    assert first["type"] == "conversation.created"
    assert not getattr(served.app.state, "core_realtime_scopes", [])


@pytest.mark.parametrize("query", REFUSED_QUERIES)
# @spec ING-NIMWS-027
async def test_bad_query_closes_1008_before_accept_first_step(served, query):
    count = served.plugin.ingress.count
    socket = AsgiWebSocket(served.app, _url(query))
    assert not await socket.connect()
    assert socket.refused_before_accept
    assert socket.close_code == 1008
    await socket.wait_closed()
    # no event, no client message read beyond the handshake
    assert _sends(socket) == []
    assert [m["type"] for m in socket.received] == ["websocket.close"]
    assert socket.consumed <= 1
    # no owner, no lease, no ingress-cap count
    assert not served.plugin.registry.records
    assert served.factory.leases == []
    assert served.plugin.ingress.count == count
    assert not getattr(served.app.state, "core_realtime_scopes", [])


# @spec ING-NIMWS-027
async def test_bad_query_is_refused_before_the_ingress_cap(clock):
    options = riva_options(load_shed_max_sessions=1)
    async with serving(
        clock=clock, options=options, tasks=("realtime",)
    ) as served:
        holder = RealtimeClient(served.app, served.clock)
        await holder.open_session()
        count = served.plugin.ingress.count
        socket = AsgiWebSocket(served.app, _url("intent=conversation"))
        assert not await socket.connect()
        # 1008 before accept, not the cap's accept, error and 1013
        assert socket.close_code == 1008
        assert not socket.accepted
        assert _sends(socket) == []
        assert served.plugin.ingress.count == count
        assert len(served.plugin.registry.records) == 1


# @spec ING-NIMWS-027
async def test_bad_query_is_refused_before_the_draining_check(served):
    served.state.draining = True
    socket = AsgiWebSocket(served.app, _url("intent=transcription&x=1"))
    assert not await socket.connect()
    assert socket.close_code == 1008
    assert not socket.accepted
    assert _sends(socket) == []


# ------------------------------------------------------- core's route


@pytest.mark.parametrize(
    "query",
    [
        "intent=transcription",
        "",
        "intent=conversation",
        "intent=transcription&intent=transcription",
        "intent=%FF",
    ],
)
# @spec ING-NIMWS-026
async def test_core_v1_realtime_untouched(served, query):
    url = f"{CORE_REALTIME_PATH}?{query}" if query else CORE_REALTIME_PATH
    socket = AsgiWebSocket(served.app, url, headers=[("x-probe", "kept")])
    assert await socket.connect()
    assert await socket.receive_json() == {"type": "core.realtime"}
    await socket.receive()
    assert socket.close_code == 1000
    (seen,) = served.app.state.core_realtime_scopes
    assert seen["path"] == CORE_REALTIME_PATH
    assert seen["query_string"] == socket.scope["query_string"]
    assert (b"x-probe", b"kept") in seen["headers"]
    assert not served.plugin.registry.records
    assert served.factory.leases == []


# @spec ING-NIMWS-026
async def test_core_v1_realtime_route_is_cores_own(served):
    core = [
        row
        for row in all_routes(served.app)
        if row[0] == "websocket" and row[1] == CORE_REALTIME_ROUTE[1]
    ]
    assert len(core) == 1
    assert core[0][3] is core_realtime_stand_in
    frontend = [
        row
        for row in all_routes(served.app)
        if row[0] == "websocket" and row[1] == REALTIME_PATH
    ]
    assert len(frontend) == 1


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


# @spec ING-NIMWS-012
async def test_init_state_accepts_receive_cap_equal_to_the_envelope(clock):
    options = riva_options(
        ws_event_envelope_max_bytes=2048, ws_receive_max_bytes=2048
    )
    async with serving(clock=clock, options=options, start=False) as served:
        assert served.plugin.config.ws_receive_max_bytes == 2048


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("binary", [False, True])
# @spec ING-NIMWS-012
# @spec ING-ERR-026
async def test_receive_cap_closes_1009(clock, configured, binary):
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


# @spec ING-NIMWS-012
async def test_receive_cap_counts_bytes_not_characters(clock):
    async with serving(clock=clock, options=_cap_options()) as served:
        client = RealtimeClient(served.app, served.clock)
        assert await client.connect()
        assert (await client.next())["type"] == "conversation.created"
        # 2100 characters, 4200 bytes in UTF-8: above the 4096-byte cap
        text = '"' + "é" * 2100 + '"'
        assert len(text) < 4096 < len(text.encode())
        await client.ws.send_text(text)
        before = await client.assert_closes(1009)
        assert [error_code(e) for e in before if e["type"] == "error"] == [
            "request_too_large"
        ]


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
