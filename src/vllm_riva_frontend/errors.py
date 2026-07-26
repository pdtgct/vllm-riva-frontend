"""Stable compatibility-error catalog."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorProjection:
    """One catalog code projected into each public dialect."""

    grpc_status: str
    nim_event: str
    http_status: int


ERROR_CODES = frozenset(
    {
        "busy",
        "admission_wait_timeout",
        "idle_timeout",
        "finalization_timeout",
        "service_unavailable",
        "configuration_timeout",
        "request_timeout",
        "malformed_request",
        "request_too_large",
        "session_terminal",
        "internal",
        "protocol_order",
        "invalid_config_field",
        "unsupported_capability",
        "unknown_locale",
        "config_change_rejected",
        "unsupported_format",
        "invalid_audio",
        "buffer_overflow",
        "invalid_event",
    }
)


_CATALOG = {
    "busy": ErrorProjection("RESOURCE_EXHAUSTED", "error", 503),
    "admission_wait_timeout": ErrorProjection(
        "DEADLINE_EXCEEDED", "error", 504
    ),
    "idle_timeout": ErrorProjection("ABORTED", "error", 504),
    "finalization_timeout": ErrorProjection("DEADLINE_EXCEEDED", "error", 504),
    "service_unavailable": ErrorProjection("UNAVAILABLE", "error", 503),
    "configuration_timeout": ErrorProjection("DEADLINE_EXCEEDED", "error", 504),
    "request_timeout": ErrorProjection("DEADLINE_EXCEEDED", "error", 504),
    "malformed_request": ErrorProjection("INVALID_ARGUMENT", "error", 400),
    "request_too_large": ErrorProjection("RESOURCE_EXHAUSTED", "error", 413),
    "session_terminal": ErrorProjection("FAILED_PRECONDITION", "error", 409),
    "internal": ErrorProjection(
        "INTERNAL", "conversation.item.input_audio_transcription.failed", 500
    ),
    "protocol_order": ErrorProjection("FAILED_PRECONDITION", "error", 400),
    "invalid_config_field": ErrorProjection("INVALID_ARGUMENT", "error", 400),
    "unsupported_capability": ErrorProjection("UNIMPLEMENTED", "error", 400),
    "unknown_locale": ErrorProjection("INVALID_ARGUMENT", "error", 400),
    "config_change_rejected": ErrorProjection("OK", "error", 400),
    "unsupported_format": ErrorProjection("INVALID_ARGUMENT", "error", 400),
    "invalid_audio": ErrorProjection(
        "INVALID_ARGUMENT",
        "conversation.item.input_audio_transcription.failed",
        400,
    ),
    "buffer_overflow": ErrorProjection("RESOURCE_EXHAUSTED", "error", 400),
    "invalid_event": ErrorProjection("INVALID_ARGUMENT", "error", 400),
}


# @spec ING-ERR-001, ING-ERR-002, ING-ERR-006
def catalog() -> dict[str, ErrorProjection]:
    """Return a copy of the stable code-to-dialect projection catalog."""
    return dict(_CATALOG)
