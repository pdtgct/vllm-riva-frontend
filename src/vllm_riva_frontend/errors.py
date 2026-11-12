"""The error catalog: stable codes and their projections into each dialect.

One catalog of code strings is projected mechanically into Riva gRPC
statuses, Riva Realtime WebSocket events and closes, and core's HTTP error
shape, so a failure means the same thing on every surface. This module also
maps the session seam's reasons and core's file-path failures to codes.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from types import MappingProxyType
from typing import Any

import grpc

#: Seam reasons that also escalate to a requested shutdown.
ESCALATES = frozenset({"engine_dead"})


@dataclass(frozen=True)
class _Row:
    """One catalog row: the gRPC status, HTTP status and Realtime close."""

    grpc_status: grpc.StatusCode
    http_status: int
    realtime_close: int | None
    description: str


# HTTP statuses of codes the file route never produces (n/a in the errors
# leaf) are the nearest equivalent, so every code has one status.
_ROWS: Mapping[str, _Row] = MappingProxyType(
    {
        "service_unavailable": _Row(
            grpc.StatusCode.UNAVAILABLE,
            503,
            1013,
            "the service is not available",
        ),
        "busy": _Row(
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            503,
            1013,
            "the server is at capacity; retry later",
        ),
        "admission_wait_timeout": _Row(
            grpc.StatusCode.DEADLINE_EXCEEDED,
            503,
            None,
            "the admission wait expired",
        ),
        "idle_timeout": _Row(
            grpc.StatusCode.ABORTED,
            408,
            1011,
            "no audio was accepted within the idle bound",
        ),
        "finalization_timeout": _Row(
            grpc.StatusCode.DEADLINE_EXCEEDED,
            504,
            1011,
            "finalization did not complete within its bound",
        ),
        "configuration_timeout": _Row(
            grpc.StatusCode.DEADLINE_EXCEEDED,
            408,
            1011,
            "configuration did not complete within its bound",
        ),
        "request_timeout": _Row(
            grpc.StatusCode.DEADLINE_EXCEEDED,
            408,
            1011,
            "the request exceeded its time bound",
        ),
        "malformed_request": _Row(
            grpc.StatusCode.INVALID_ARGUMENT,
            400,
            None,
            "the request is malformed",
        ),
        "request_too_large": _Row(
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            413,
            1009,
            "the request exceeds a size limit",
        ),
        "session_terminal": _Row(
            grpc.StatusCode.FAILED_PRECONDITION,
            409,
            1008,
            "the session is finalizing and accepts no further message",
        ),
        "protocol_order": _Row(
            grpc.StatusCode.FAILED_PRECONDITION,
            400,
            1008,
            "a message arrived out of protocol order",
        ),
        "invalid_event": _Row(
            grpc.StatusCode.INVALID_ARGUMENT,
            400,
            1008,
            "the event is malformed or unknown",
        ),
        "invalid_config_field": _Row(
            grpc.StatusCode.INVALID_ARGUMENT,
            400,
            None,
            "a configuration field is invalid",
        ),
        "unsupported_capability": _Row(
            grpc.StatusCode.UNIMPLEMENTED,
            501,
            None,
            "the capability is not served",
        ),
        "unknown_locale": _Row(
            grpc.StatusCode.INVALID_ARGUMENT,
            400,
            None,
            "the locale is not served",
        ),
        "config_change_rejected": _Row(
            grpc.StatusCode.OK,
            400,
            None,
            "the field cannot change after configuration",
        ),
        "unsupported_format": _Row(
            grpc.StatusCode.INVALID_ARGUMENT,
            400,
            1008,
            "the audio format is not supported",
        ),
        "invalid_audio": _Row(
            grpc.StatusCode.INVALID_ARGUMENT,
            400,
            1008,
            "the audio cannot be decoded",
        ),
        "buffer_overflow": _Row(
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            413,
            1011,
            "a buffer limit was exceeded",
        ),
        "internal": _Row(
            grpc.StatusCode.INTERNAL,
            500,
            1011,
            "an internal error occurred",
        ),
    }
)


def _row(code: str) -> _Row:
    try:
        return _ROWS[code]
    except KeyError:
        raise ValueError(f"unknown error catalog code {code!r}") from None


class ErrorCatalog:
    """The stable codes and their gRPC, HTTP and Realtime projections."""

    @property
    def codes(self) -> tuple[str, ...]:
        """Every catalog code."""
        return tuple(_ROWS)

    # @spec ING-ERR-031, ING-ERR-060, ING-ERR-043, ING-ERR-046
    def grpc_status(self, code: str) -> grpc.StatusCode:
        """Return the gRPC status of ``code``.

        ``config_change_rejected`` is ``OK``: it is never raised as an abort.
        """
        return _row(code).grpc_status

    # @spec ING-ERR-055
    def http_status(self, code: str) -> int:
        """Return the HTTP status of ``code`` on the file route.

        ``request_timeout`` is 408, the body-read timeout; the core-execution
        timeout passes 504 to ``FrontendError.http_body`` itself.
        """
        return _row(code).http_status

    # @spec ING-ERR-033, ING-ERR-051
    def realtime_close(self, code: str) -> int | None:
        """Return the Realtime close of ``code``'s catalog row, if any.

        ``None`` means the session continues. The phase table, applied by the
        Realtime adapter, overrides this where a phase dictates another close.
        """
        return _row(code).realtime_close

    def description(self, code: str) -> str:
        """Return the default detail text of ``code``."""
        return _row(code).description


_CATALOG = ErrorCatalog()


def catalog() -> ErrorCatalog:
    """Return the error catalog."""
    return _CATALOG


_HTTP_ERROR_TYPES: Mapping[int, str] = MappingProxyType(
    {
        400: "BadRequestError",
        404: "NotFoundError",
        408: "RequestTimeoutError",
        409: "ConflictError",
        413: "RequestTooLargeError",
        422: "UnprocessableEntityError",
        500: "InternalServerError",
        501: "NotImplementedError",
        503: "ServiceUnavailableError",
        504: "GatewayTimeoutError",
    }
)


def _http_error_type(status: int) -> str:
    """Name the error ``type`` the way core names its error responses."""
    known = _HTTP_ERROR_TYPES.get(status)
    if known is not None:
        return known
    try:
        phrase = HTTPStatus(status).phrase
    except ValueError:
        return "Error"
    return "".join(word.capitalize() for word in phrase.split()) + "Error"


class FrontendError(Exception):
    """A failure carrying one catalog code, projected by every surface."""

    # @spec ING-ERR-003, ING-ERR-060
    def __init__(
        self,
        code: str,
        detail: str = "",
        fields: Iterable[str] = (),
        authority: str | None = None,
    ) -> None:
        """Bind the code, its detail, the offending fields and authority.

        ``authority`` is carried by ``busy`` only: ``load_shed`` for the
        frontend's own limits, ``provider`` for an engine refusal.
        """
        _row(code)
        self.code = code
        self.detail = detail or _CATALOG.description(code)
        self.fields: tuple[str, ...] = (
            (fields,) if isinstance(fields, str) else tuple(fields)
        )
        self.authority = authority
        super().__init__(self.grpc_details())

    def _text(self) -> str:
        if self.authority is None:
            return self.detail
        # A busy error leads with its authority on every surface.
        return f"authority: {self.authority}; {self.detail}"

    def message(self) -> str:
        """Return ``<code>: <detail>``, the text every dialect leads with."""
        return f"{self.code}: {self._text()}"

    # @spec ING-ERR-064, ING-ERR-003
    def grpc_details(self) -> str:
        """Return ``<code>: <detail>``, then ``(fields: ...)`` when named."""
        if not self.fields:
            return self.message()
        return f"{self.message()} (fields: {', '.join(self.fields)})"

    # @spec ING-ERR-009
    def http_body(self, status: int) -> dict[str, Any]:
        """Return core's HTTP error shape for this error at ``status``."""
        return {
            "error": {
                "message": self.message(),
                "type": _http_error_type(int(status)),
                "param": ",".join(self.fields) if self.fields else None,
                "code": int(status),
            }
        }

    def __repr__(self) -> str:
        """Return a representation naming the code and the detail."""
        return f"FrontendError({self.code!r}, {self.detail!r})"


# --------------------------------------------------------- the session seam


def seam_error(exc: BaseException) -> tuple[str | None, str | None]:
    """Return the seam's ``session_error`` and ``reason`` of ``exc``.

    Both are read with ``getattr``, so no model-plugin class is named; a
    missing or non-string attribute reads as ``None``.
    """
    session_error = getattr(exc, "session_error", None)
    reason = getattr(exc, "reason", None)
    return (
        session_error if isinstance(session_error, str) else None,
        reason if isinstance(reason, str) else None,
    )


# @spec ING-ERR-063
def _refused(reason: str) -> FrontendError:
    if reason == "locale":
        return FrontendError("unknown_locale", "the locale is not served")
    if reason in ("capacity", "active_limit"):
        return FrontendError(
            "busy",
            f"the engine refused the stream ({reason})",
            authority="provider",
        )
    if reason == "engine_paused":
        return FrontendError("service_unavailable", "the engine is paused")
    if reason == "cadence":
        return FrontendError("internal", "the factory refused the cadence")
    if reason == "request_id_in_use":
        return FrontendError("internal", "the session id is already in use")
    return FrontendError(
        "internal", f"the factory refused the stream: {reason}"
    )


def _failed(reason: str) -> FrontendError:
    if reason == "expired":
        return FrontendError("idle_timeout", "the engine ended an idle stream")
    if reason == "engine_dead":
        return FrontendError("service_unavailable", "the engine is dead")
    if reason == "input_capacity":
        return FrontendError(
            "buffer_overflow", "the session's input limit was exceeded"
        )
    if reason == "output_capacity":
        return FrontendError(
            "buffer_overflow",
            "the session's output (transcript) limit was exceeded",
        )
    if reason == "cancelled":
        # Projected as the owner's selected cause when it has one.
        return FrontendError("internal", "the session was cancelled")
    return FrontendError("internal", f"the session failed: {reason}")


# @spec ING-ERR-008, ING-ERR-013, ING-ERR-014, ING-ERR-010, ING-ERR-035
# @spec ING-ADM-008, ING-ERR-005, ING-ERR-018, ING-ERR-024
def project_seam_error(
    exc: BaseException, *, call: str | None = None
) -> FrontendError:
    """Map a factory or lease exception to its catalog code, reason by reason.

    ``call`` names the lease call that failed; a ``release`` failure is
    ``service_unavailable`` for ``engine_dead`` and ``internal`` otherwise.
    An exception without ``session_error`` or with an unknown reason is
    ``internal``.
    """
    session_error, reason = seam_error(exc)
    if session_error is None or reason is None:
        return FrontendError("internal", f"unexpected {type(exc).__name__}")
    if call == "release":
        if reason == "engine_dead":
            return FrontendError("service_unavailable", "the engine is dead")
        return FrontendError("internal", f"release failed: {reason}")
    if session_error == "refused":
        return _refused(reason)
    if session_error == "failed":
        return _failed(reason)
    return FrontendError("internal", f"unknown session error {session_error}")


# ----------------------------------------------- core's file-path failures

# Core's published names and message texts for its size limits and its
# decoder failures (pinned against core by the errors leaf's tests).
_SIZE_PARAMETERS = frozenset({"audio_filesize_mb"})
_SIZE_MESSAGES = ("Maximum file size exceeded",)
_DECODE_LIMIT_PREFIXES = (
    "Audio exceeds maximum allowed duration of",
    "Audio decode exceeded",
    "Audio would allocate",
)
_DECODE_INVALID_PREFIXES = (
    "Invalid or corrupted audio data",
    "Invalid or unsupported audio file",
    "No audio stream found",
    "No audio found in the input",
    "torchcodec requires an integer sample rate",
)


def _is_core_error(exc: BaseException, module: str, name: str) -> bool:
    """Whether ``exc`` is core's ``module.name``, without importing core."""
    try:
        cls = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError):
        return any(base.__name__ == name for base in type(exc).__mro__)
    return isinstance(exc, cls)


# @spec ING-ERR-052, ING-ERR-053, ING-ERR-039, ING-ERR-040, ING-FILE-081
def classify_core_exception(exc: BaseException) -> FrontendError:
    """Map an exception raised by core's transcription service to a code."""
    if _is_core_error(exc, "vllm.exceptions", "VLLMValidationError"):
        parameter = getattr(exc, "parameter", None)
        message = str(exc.args[0]) if exc.args else str(exc)
        if parameter in _SIZE_PARAMETERS or message.startswith(_SIZE_MESSAGES):
            return FrontendError("request_too_large", message)
        fields = (parameter,) if isinstance(parameter, str) else ()
        return FrontendError("invalid_config_field", message, fields=fields)
    if _is_core_error(exc, "vllm.v1.engine.exceptions", "EngineDeadError"):
        return FrontendError("service_unavailable", "the engine is dead")
    if isinstance(exc, ValueError):
        message = str(exc)
        if message.startswith(_DECODE_LIMIT_PREFIXES):
            return FrontendError("request_too_large", message)
        if message.startswith(_DECODE_INVALID_PREFIXES):
            return FrontendError("invalid_audio", message)
        # No length refusal is projected (ING-FILE-081): a message about the
        # model length is one more unmatched ValueError. Core's own handler
        # answers 400, never a server error.
        return FrontendError("invalid_config_field", message)
    return FrontendError("internal", f"unexpected {type(exc).__name__}")


# @spec ING-ERR-041
def project_error_response(response: Any) -> FrontendError:
    """Map an ``ErrorResponse`` returned by core's service, by HTTP status."""
    error = getattr(response, "error", None)
    status = getattr(error, "code", None)
    message = str(getattr(error, "message", "") or "")
    param = getattr(error, "param", None)
    if not isinstance(status, int):
        return FrontendError("internal", message)
    if status == 413:
        return FrontendError("request_too_large", message)
    if status == 501:
        return FrontendError("unsupported_capability", message)
    if status == 503:
        return FrontendError("service_unavailable", message)
    if 400 <= status < 500:
        field = param if isinstance(param, str) and param else "audio"
        return FrontendError("invalid_config_field", message, fields=(field,))
    return FrontendError("internal", message)
