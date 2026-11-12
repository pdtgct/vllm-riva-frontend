"""The error catalog and its three projections, as units."""

from __future__ import annotations

import ast
import contextlib
import json
from pathlib import Path
from typing import Any

import grpc
import pytest

PACKAGE = Path(__file__).parents[2] / "src" / "vllm_riva_frontend"

CODES = {
    "service_unavailable",
    "busy",
    "admission_wait_timeout",
    "idle_timeout",
    "finalization_timeout",
    "configuration_timeout",
    "request_timeout",
    "malformed_request",
    "request_too_large",
    "session_terminal",
    "protocol_order",
    "invalid_event",
    "invalid_config_field",
    "unsupported_capability",
    "unknown_locale",
    "config_change_rejected",
    "unsupported_format",
    "invalid_audio",
    "buffer_overflow",
    "internal",
}

GRPC_STATUS = {
    "service_unavailable": "UNAVAILABLE",
    "busy": "RESOURCE_EXHAUSTED",
    "request_too_large": "RESOURCE_EXHAUSTED",
    "buffer_overflow": "RESOURCE_EXHAUSTED",
    "idle_timeout": "ABORTED",
    "finalization_timeout": "DEADLINE_EXCEEDED",
    "configuration_timeout": "DEADLINE_EXCEEDED",
    "request_timeout": "DEADLINE_EXCEEDED",
    "malformed_request": "INVALID_ARGUMENT",
    "invalid_event": "INVALID_ARGUMENT",
    "invalid_config_field": "INVALID_ARGUMENT",
    "unknown_locale": "INVALID_ARGUMENT",
    "unsupported_format": "INVALID_ARGUMENT",
    "invalid_audio": "INVALID_ARGUMENT",
    "session_terminal": "FAILED_PRECONDITION",
    "protocol_order": "FAILED_PRECONDITION",
    "unsupported_capability": "UNIMPLEMENTED",
    "internal": "INTERNAL",
    "config_change_rejected": "OK",
    "admission_wait_timeout": "DEADLINE_EXCEEDED",
}

HTTP_STATUS = {
    "malformed_request": 400,
    "invalid_config_field": 400,
    "unknown_locale": 400,
    "request_too_large": 413,
    "service_unavailable": 503,
    "busy": 503,
    "unsupported_capability": 501,
    "internal": 500,
}

REALTIME_CLOSE = {
    "service_unavailable": 1013,
    "busy": 1013,
    "idle_timeout": 1011,
    "finalization_timeout": 1011,
    "configuration_timeout": 1011,
    "request_timeout": 1011,
    "request_too_large": 1009,
    "session_terminal": 1008,
    "protocol_order": 1008,
    "invalid_event": 1008,
    "buffer_overflow": 1011,
}

CONTINUING = (
    "invalid_config_field",
    "unsupported_capability",
    "unknown_locale",
    "config_change_rejected",
)


def _name(status: Any) -> str:
    """The name of a ``grpc.StatusCode`` (what ``grpc_status`` returns)."""
    assert isinstance(status, grpc.StatusCode), status
    return status.name


def _body(rendered: Any) -> dict[str, Any]:
    """``http_body`` returns core's error shape as a plain mapping."""
    assert isinstance(rendered, dict), type(rendered)
    json.dumps(rendered)  # serializable as it stands
    return rendered


@pytest.fixture
def cat():
    from vllm_riva_frontend.errors import catalog

    return catalog()


# @spec ING-ERR-060, ING-ERR-031
def test_catalog_holds_exactly_the_twenty_stable_codes(cat) -> None:
    from vllm_riva_frontend.errors import ErrorCatalog

    assert isinstance(cat, ErrorCatalog)
    assert set(cat.codes) == CODES
    assert len(CODES) == 20


# @spec ING-ERR-031
@pytest.mark.parametrize("code", sorted(GRPC_STATUS))
def test_catalog_projects_each_code_to_its_grpc_status(cat, code) -> None:
    assert _name(cat.grpc_status(code)) == GRPC_STATUS[code]


# @spec ING-ERR-055
@pytest.mark.parametrize("code", sorted(HTTP_STATUS))
def test_catalog_projects_file_route_codes_to_http_status(cat, code) -> None:
    assert cat.http_status(code) == HTTP_STATUS[code]


# @spec ING-ERR-055, ING-ERR-046
def test_request_timeout_projects_to_408(cat) -> None:
    # One status per code: 408, the body-read timeout. The core-execution
    # timeout is the one place that renders 504 (``http_body(504)``); the
    # route tests pin which one each bound produces.
    assert cat.http_status("request_timeout") == 408


# @spec ING-ERR-060, ING-ERR-051
@pytest.mark.parametrize("code", sorted(REALTIME_CLOSE))
def test_catalog_closing_codes_carry_their_realtime_close(cat, code) -> None:
    assert cat.realtime_close(code) == REALTIME_CLOSE[code]


# @spec ING-ERR-033
@pytest.mark.parametrize("code", CONTINUING)
def test_configuration_rejections_do_not_close_the_session(cat, code) -> None:
    assert cat.realtime_close(code) is None


# @spec ING-ERR-060
def test_every_code_has_a_non_silent_grpc_projection(cat) -> None:
    for code in CODES:
        status = _name(cat.grpc_status(code))
        assert status, code
        if code != "config_change_rejected":
            assert status != "OK", code


# @spec ING-ERR-043, ING-ERR-046
def test_capacity_wait_inactivity_finalization_and_meanings_are_distinct(
    cat,
) -> None:
    distinct = [
        "busy",
        "admission_wait_timeout",
        "idle_timeout",
        "finalization_timeout",
        "service_unavailable",
        "configuration_timeout",
        "request_timeout",
        "malformed_request",
        "request_too_large",
    ]
    assert set(distinct) <= set(cat.codes)
    assert len(set(distinct)) == len(distinct)
    # capacity is retryable load, the timeouts are not
    assert _name(cat.grpc_status("busy")) != _name(
        cat.grpc_status("idle_timeout")
    )
    assert _name(cat.grpc_status("idle_timeout")) != _name(
        cat.grpc_status("finalization_timeout")
    )


# @spec ING-ERR-043, ING-ERR-046, ING-ERR-057
def test_session_terminal_and_service_unavailable_remain_distinct(cat) -> None:
    assert "session_terminal" in cat.codes
    assert _name(cat.grpc_status("session_terminal")) == "FAILED_PRECONDITION"
    assert _name(cat.grpc_status("service_unavailable")) == "UNAVAILABLE"
    assert cat.realtime_close("session_terminal") == 1008


def _string_constants(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


# @spec ING-ERR-043
def test_admission_wait_timeout_is_reserved_and_emitted_by_no_surface() -> None:
    import vllm_riva_frontend.errors  # noqa: F401  (the catalog exists)

    users = [
        path.name
        for path in sorted(PACKAGE.glob("*.py"))
        if path.name != "errors.py"
        and "admission_wait_timeout" in _string_constants(path)
    ]
    assert users == []


# @spec ING-ERR-064, ING-ERR-003
def test_grpc_details_lead_with_the_code_and_name_the_fields() -> None:
    from vllm_riva_frontend.errors import FrontendError

    error = FrontendError(
        "invalid_config_field",
        "profanity filtering is not supported",
        fields=("profanity_filter",),
    )
    assert error.grpc_details() == (
        "invalid_config_field: profanity filtering is not supported"
        " (fields: profanity_filter)"
    )


# @spec ING-ERR-064, ING-ERR-003
def test_grpc_details_name_every_offending_field() -> None:
    from vllm_riva_frontend.errors import FrontendError

    error = FrontendError(
        "unsupported_format",
        "off-matrix cell",
        fields=("encoding", "sample_rate_hertz"),
    )
    assert error.grpc_details() == (
        "unsupported_format: off-matrix cell"
        " (fields: encoding, sample_rate_hertz)"
    )


# @spec ING-ERR-064
def test_grpc_details_without_fields_still_lead_with_the_code() -> None:
    from vllm_riva_frontend.errors import FrontendError

    details = FrontendError("internal", "unexpected failure").grpc_details()
    assert details == "internal: unexpected failure"


# @spec ING-ERR-009
def test_http_body_uses_cores_error_shape() -> None:
    from vllm_riva_frontend.errors import FrontendError

    error = FrontendError(
        "invalid_config_field",
        "use_beam_search is not supported",
        fields=("use_beam_search",),
    )
    body = _body(error.http_body(400))
    assert set(body) == {"error"}
    assert set(body["error"]) == {"message", "type", "param", "code"}
    assert body["error"]["message"] == (
        "invalid_config_field: use_beam_search is not supported"
    )
    assert body["error"]["param"] == "use_beam_search"
    assert body["error"]["code"] == 400
    assert body["error"]["type"] == "BadRequestError"


# @spec ING-ERR-009
@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("malformed_request", 400),
        ("request_too_large", 413),
        ("request_timeout", 408),
        ("request_timeout", 504),
        ("unsupported_capability", 501),
        ("service_unavailable", 503),
        ("busy", 503),
        ("internal", 500),
    ],
)
def test_http_body_message_leads_with_the_code(code, status) -> None:
    from vllm_riva_frontend.errors import FrontendError

    body = _body(FrontendError(code, "detail").http_body(status))
    assert body["error"]["message"] == f"{code}: detail"
    assert body["error"]["code"] == status
    assert body["error"]["param"] is None
    assert isinstance(body["error"]["type"], str) and body["error"]["type"]


# @spec ING-ERR-051, ING-ERR-008
def test_busy_carries_its_authority() -> None:
    from vllm_riva_frontend.errors import FrontendError

    error = FrontendError("busy", "ingress cap full", authority="load_shed")
    assert error.authority == "load_shed"
    assert error.code == "busy"
    assert error.grpc_details().startswith("busy: authority: load_shed")


# ----------------------------------------------- uncatalogued refusals
#
# ING-ERR-061 lists the answers that carry no catalog code. Each case below
# drives one item through the app (``support.asgi``). Two items are pinned
# elsewhere: gRPC's native framing error above the message cap
# (``streaming/test_grpc_limits.py``) and core's own errors on the file route
# (``test_file_errors.py``). The status that a listener stop's termination
# gives an RPC is the gRPC framework's; a listener-stop test asserts the
# selected cause and the cleanup instead (ING-LIFE-064,
# ``test_shutdown_rows.py``).

TOKEN = "riva-test-key"
UNAVAILABLE_BODY = {"error": "Service Unavailable"}
UNAUTHORIZED_BODY = {"error": "Unauthorized"}


def _bearer(token: str | None) -> list[tuple[str, str]]:
    return [] if token is None else [("authorization", f"Bearer {token}")]


async def _uncatalogued_app(env, *, api_key=None, init=True):
    """The frontend on core's app; ``init`` runs core's state init."""
    from support.app import build_app, init_state

    app = build_app([env.plugin], api_key=api_key)
    if init:
        await init_state(
            app,
            env.engine,
            models=env.models,
            transcription=env.transcription,
        )
    return app


async def _http(app, path: str, token: str | None = None):
    from support.asgi import asgi_http

    return await asgi_http(app, "GET", path, headers=_bearer(token))


async def _handshake(app, url: str, *, denial: bool, token=None):
    """Connect; an app that ends without a close leaves the socket as is."""
    from support.asgi import AsgiWebSocket

    socket = AsgiWebSocket(
        app, url, headers=_bearer(token), denial_response=denial
    )
    with contextlib.suppress(ConnectionError):
        await socket.connect()
    return socket


def _assert_closed_before_accept(socket) -> None:
    assert not socket.accepted, socket.received
    assert socket.denied is None, socket.denied
    assert socket.close_code is not None, socket.received
    assert all(m["type"] != "websocket.send" for m in socket.received)


async def _http_auth_unresolved(env) -> None:
    from support.app import OPERATIONAL_ROUTES

    app = await _uncatalogued_app(env, init=False)
    response = await _http(app, OPERATIONAL_ROUTES[0][1])
    assert response.status == 503, (response.status, response.error)
    assert response.json() == UNAVAILABLE_BODY


async def _websocket_auth_unresolved_denial(env) -> None:
    from support.realtime import REALTIME_URL

    app = await _uncatalogued_app(env, init=False)
    socket = await _handshake(app, REALTIME_URL, denial=True)
    assert not socket.accepted, socket.received
    assert await socket.read_denial() == (503, UNAVAILABLE_BODY)


async def _websocket_auth_unresolved_close(env) -> None:
    from support.app import init_state
    from support.realtime import REALTIME_URL

    app = await _uncatalogued_app(env, init=False)
    socket = await _handshake(app, REALTIME_URL, denial=False)
    _assert_closed_before_accept(socket)
    # control: once resolved, the same handshake reaches the route, which
    # accepts it (and refuses it as not serving, ING-NIMWS-028)
    await init_state(
        app, env.engine, models=env.models, transcription=env.transcription
    )
    control = await _handshake(app, REALTIME_URL, denial=False)
    assert control.accepted, control.received


async def _http_auth_refused(env) -> None:
    from support.app import OPERATIONAL_ROUTES

    app = await _uncatalogued_app(env, api_key=TOKEN)
    response = await _http(app, OPERATIONAL_ROUTES[0][1])
    assert response.status == 401, (response.status, response.error)
    assert response.json() == UNAUTHORIZED_BODY


async def _websocket_auth_refused_denial(env) -> None:
    from support.realtime import REALTIME_URL

    app = await _uncatalogued_app(env, api_key=TOKEN)
    socket = await _handshake(app, REALTIME_URL, denial=True)
    assert not socket.accepted, socket.received
    assert await socket.read_denial() == (401, UNAUTHORIZED_BODY)


async def _websocket_auth_refused_close(env) -> None:
    from support.realtime import REALTIME_URL

    app = await _uncatalogued_app(env, api_key=TOKEN)
    socket = await _handshake(app, REALTIME_URL, denial=False)
    _assert_closed_before_accept(socket)
    # control: with the token the same handshake reaches the route
    control = await _handshake(app, REALTIME_URL, denial=False, token=TOKEN)
    assert control.accepted, control.received


async def _realtime_query(env) -> None:
    from support.realtime import REALTIME_PATH

    app = await _uncatalogued_app(env)
    url = f"{REALTIME_PATH}?intent=conversation"
    # no body even when the client offers the denial-response extension
    socket = await _handshake(app, url, denial=True)
    _assert_closed_before_accept(socket)
    assert socket.close_code == 1008


async def _http_unknown_path(env) -> None:
    from support.app import PLUGIN_PREFIX

    app = await _uncatalogued_app(env, api_key=TOKEN)
    path = f"{PLUGIN_PREFIX}/no-such-route"
    response = await _http(app, path, TOKEN)
    assert response.status == 404, (response.status, response.error)
    assert response.json() == {"detail": "Not Found"}


async def _websocket_unknown_path(env) -> None:
    from support.app import PLUGIN_PREFIX

    app = await _uncatalogued_app(env, api_key=TOKEN)
    url = f"{PLUGIN_PREFIX}/no-such-socket"
    socket = await _handshake(app, url, denial=False, token=TOKEN)
    _assert_closed_before_accept(socket)


UNCATALOGUED = {
    "http_auth_unresolved": _http_auth_unresolved,
    "websocket_auth_unresolved_denial": _websocket_auth_unresolved_denial,
    "websocket_auth_unresolved_close": _websocket_auth_unresolved_close,
    "http_auth_refused": _http_auth_refused,
    "websocket_auth_refused_denial": _websocket_auth_refused_denial,
    "websocket_auth_refused_close": _websocket_auth_refused_close,
    "realtime_query": _realtime_query,
    "http_unknown_path": _http_unknown_path,
    "websocket_unknown_path": _websocket_unknown_path,
}


@pytest.mark.parametrize("item", sorted(UNCATALOGUED))
# @spec ING-ERR-061
async def test_uncatalogued_refusals_have_their_wire_shape(
    item, plugin, fake_engine, fake_models, fake_transcription
) -> None:
    from types import SimpleNamespace

    env = SimpleNamespace(
        plugin=plugin,
        engine=fake_engine,
        models=fake_models,
        transcription=fake_transcription,
    )
    await UNCATALOGUED[item](env)
