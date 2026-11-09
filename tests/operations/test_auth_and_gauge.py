"""Core's ``--api-key`` rule and core's WebSocket gauge over the plugin.

The app is built by ``support.app.build_app`` with ``api_key``, which
installs core's ``AuthenticationMiddleware``; the dispatcher sits inside it.
"""

from __future__ import annotations

import pytest
from support.asgi import AsgiWebSocket

TOKEN = "s3cret-token"
OPERATIONAL = (
    "/v1/health/ready",
    "/v1/health/live",
    "/v1/version",
    "/v1/metadata",
)
BOOTSTRAP = "/v1/realtime/transcription_sessions"
REALTIME_URL = "/v1/realtime?intent=transcription"
GAUGE = "vllm:websocket_connections_active"


@pytest.fixture
def test_app(plugins, supported_tasks):
    from support.app import build_app

    return build_app(plugins, supported_tasks=supported_tasks, api_key=TOKEN)


def _client(app, token=None):
    from support.app import http_client

    return http_client(app, api_key=token)


# @spec ING-SHIM-013
@pytest.mark.parametrize("path", OPERATIONAL)
async def test_operational_routes_need_the_bearer_token(
    path, started, test_app
) -> None:
    async with _client(test_app) as anonymous:
        refused = await anonymous.get(path)
    async with _client(test_app, TOKEN) as authorized:
        allowed = await authorized.get(path)
    assert refused.status_code == 401
    assert allowed.status_code == 200


# @spec ING-SHIM-013
async def test_bootstrap_post_needs_the_bearer_token(started, test_app) -> None:
    async with _client(test_app) as anonymous:
        refused = await anonymous.post(BOOTSTRAP, json={})
    async with _client(test_app, TOKEN) as authorized:
        allowed = await authorized.post(BOOTSTRAP, json={})
    assert refused.status_code == 401
    assert allowed.status_code == 200


# @spec ING-SHIM-013
async def test_core_health_is_exempt_and_reports_cores_result(
    started, test_app
) -> None:
    async with _client(test_app) as anonymous:
        response = await anonymous.get("/health")
    assert response.status_code == 200


# @spec ING-SHIM-013
async def test_claimed_websocket_without_token_is_refused(
    started, test_app, fake_factory
) -> None:
    ws = AsgiWebSocket(test_app, REALTIME_URL)
    assert await ws.connect() is False
    assert fake_factory.leases == []


# @spec ING-SHIM-013
async def test_claimed_websocket_with_token_is_served(
    started, test_app
) -> None:
    ws = AsgiWebSocket(
        test_app,
        REALTIME_URL,
        headers=[("authorization", f"Bearer {TOKEN}")],
    )
    assert await ws.connect() is True
    event = await ws.receive_json()
    assert event["type"] == "conversation.created"
    await ws.disconnect()
    await ws.wait_closed()


def _gauge() -> float:
    from prometheus_client import REGISTRY

    value = REGISTRY.get_sample_value(GAUGE)
    return 0.0 if value is None else value


# @spec ING-SHIM-014
async def test_claimed_session_is_counted_by_cores_gauge(
    started, test_app, fake_factory
) -> None:
    from vllm_riva_frontend.nim_ws import bootstrap_session

    before = _gauge()
    ws = AsgiWebSocket(
        test_app,
        REALTIME_URL,
        headers=[("authorization", f"Bearer {TOKEN}")],
    )
    assert await ws.connect()
    assert (await ws.receive_json())["type"] == "conversation.created"
    await ws.send_json(
        {
            "type": "transcription_session.update",
            "session": bootstrap_session(
                started.served_model, fake_factory.default_locale
            ),
        }
    )
    assert (await ws.receive_json())["type"] == "transcription_session.updated"
    assert _gauge() == before + 1
    await ws.send_json({"type": "input_audio_buffer.done"})
    while ws.close_code is None:
        await ws.receive()
    await ws.wait_closed()
    assert ws.close_code == 1000
    assert _gauge() == before
