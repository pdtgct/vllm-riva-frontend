"""The error catalog and its three projections, as units."""

from __future__ import annotations

import ast
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


# @spec ING-ERR-042, ING-ERR-031
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


# @spec ING-ERR-042, ING-ERR-051
@pytest.mark.parametrize("code", sorted(REALTIME_CLOSE))
def test_catalog_closing_codes_carry_their_realtime_close(cat, code) -> None:
    assert cat.realtime_close(code) == REALTIME_CLOSE[code]


# @spec ING-ERR-033
@pytest.mark.parametrize("code", CONTINUING)
def test_configuration_rejections_do_not_close_the_session(cat, code) -> None:
    assert cat.realtime_close(code) is None


# @spec ING-ERR-042
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


# @spec ING-ERR-029, ING-ERR-003
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


# @spec ING-ERR-029, ING-ERR-003
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


# @spec ING-ERR-029
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
