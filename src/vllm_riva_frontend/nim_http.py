"""Raw-ASGI Speech NIM HTTP transcription without framework upload helpers."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Never, Protocol, cast

from python_multipart import MultipartParser
from python_multipart.exceptions import MultipartParseError

from vllm_riva_frontend.admission import (
    HostAdmission,
    LoadShedGate,
    LoadShedRejected,
    try_acquire_admission,
)
from vllm_riva_frontend.frontend import (
    RiffFormat,
    StreamingAudioFrontend,
    sniff_riff,
    validate_format,
)
from vllm_riva_frontend.lease import DirectLeaseOwner, SessionFactory

_BOUNDARY = re.compile(rb"^[0-9A-Za-z'()+_,\-./:=?]{1,200}$")
_TEXT_FIELDS = frozenset(
    {"language", "model", "response_format", "temperature"}
)


@dataclass(frozen=True)
class MultipartLimits:
    """Finite bounds required before accepting an HTTP request body."""

    encoded_audio_bytes: int
    envelope_bytes: int
    content_type_bytes: int
    boundary_bytes: int
    part_count: int
    part_header_bytes: int
    text_field_bytes: int

    def __post_init__(self) -> None:
        """Reject an unbounded or nonsensical parser configuration."""
        for value in (
            self.encoded_audio_bytes,
            self.envelope_bytes,
            self.content_type_bytes,
            self.boundary_bytes,
            self.part_count,
            self.part_header_bytes,
            self.text_field_bytes,
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError("multipart limits must be positive integers")


@dataclass(frozen=True)
class HttpTranscriptionConfig:
    """The HTTP/lifecycle values consumed by the mounted endpoint."""

    limits: MultipartLimits
    max_riff_header_bytes: int
    max_decoded_duration_seconds: float
    pre_submit_max_samples: int
    request_timeout: float
    finalization_timeout: float
    cleanup_timeout: float

    def __post_init__(self) -> None:
        """Keep every application timeout and decoder bound finite."""
        for value in (
            self.max_riff_header_bytes,
            self.max_decoded_duration_seconds,
            self.request_timeout,
            self.finalization_timeout,
            self.cleanup_timeout,
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("HTTP transcription bounds must be finite")
            if value <= 0:
                raise ValueError("HTTP transcription bounds must be positive")
        if (
            type(self.pre_submit_max_samples) is not int
            or self.pre_submit_max_samples < 1
        ):
            raise ValueError(
                "pre_submit_max_samples must be a positive integer"
            )


class AsgiReceive(Protocol):
    """Minimal raw-ASGI receive shape; deliberately not a framework request."""

    async def __call__(self) -> dict[str, object]:
        """Receive the next ASGI request-body message."""


class AsgiSend(Protocol):
    """Minimal raw-ASGI response-send shape."""

    async def __call__(self, message: dict[str, object]) -> None:
        """Send one ASGI response message."""


class OwnerToken(Protocol):
    """One lifecycle-visible compatibility owner registration."""

    async def release(self) -> None:
        """Release the lifecycle owner after terminal lease cleanup."""


type OwnerRegister = Callable[[], Awaitable[OwnerToken]]
type OwnerFactory = Callable[..., DirectLeaseOwner]


@dataclass(frozen=True)
class HttpFailure:
    """The public status/body selected by one bounded HTTP rejection."""

    status: int
    body: object


@dataclass(frozen=True)
class HttpResponse:
    """A completed response which a raw ASGI mount can write once."""

    status: int
    content_type: bytes
    body: bytes


class _FailureRaised(Exception):
    def __init__(self, failure: HttpFailure) -> None:
        self.failure = failure


class _PreSubmitOverflow(Exception):
    """Signal that a piece cannot enter the bounded owner submission queue."""


class _InvalidAcceptanceCredit(Exception):
    """The provider violated exact-once, exact-piece acceptance credit."""


def _malformed(reason: str) -> HttpFailure:
    return HttpFailure(
        400,
        {
            "error": {
                "message": f"malformed_request: {reason}",
                "type": "BadRequestError",
                "code": 400,
            }
        },
    )


def _multipart_too_large(reason: str) -> HttpFailure:
    return HttpFailure(
        413,
        {
            "error": {
                "message": f"request_too_large: {reason}",
                "type": "RequestTooLargeError",
                "code": 413,
            }
        },
    )


def _detail(code: str, reason: str, *, status: int = 400) -> HttpFailure:
    return HttpFailure(status, {"detail": f"{code}: {reason}"})


def _parse_content_type(
    headers: list[tuple[bytes, bytes]], limits: MultipartLimits
) -> bytes | HttpFailure:
    values = [
        value for name, value in headers if name.lower() == b"content-type"
    ]
    if len(values) != 1:
        return _malformed("exactly one Content-Type is required")
    raw = values[0]
    if len(raw) > limits.content_type_bytes:
        return _multipart_too_large("Content-Type exceeds limit")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        return _malformed("Content-Type must be ASCII")
    pieces = [piece.strip() for piece in text.split(";")]
    if not pieces or pieces[0].lower() != "multipart/form-data":
        return _malformed("Content-Type must be multipart/form-data")
    boundary_values: list[str] = []
    for piece in pieces[1:]:
        key, separator, value = piece.partition("=")
        if not separator or not key:
            return _malformed("malformed Content-Type parameter")
        if key.lower() != "boundary":
            return _malformed("unsupported Content-Type parameter")
        boundary_values.append(value)
    if len(boundary_values) != 1:
        return _malformed("exactly one boundary parameter is required")
    boundary = boundary_values[0]
    if boundary.startswith('"') and boundary.endswith('"'):
        boundary = boundary[1:-1]
    try:
        encoded = boundary.encode("ascii")
    except UnicodeEncodeError:
        return _malformed("multipart boundary must be ASCII")
    if len(encoded) > limits.boundary_bytes:
        return _multipart_too_large("multipart boundary exceeds limit")
    if not _BOUNDARY.fullmatch(encoded):
        return _malformed("invalid multipart boundary")
    return encoded


def _parse_content_length(
    headers: list[tuple[bytes, bytes]], limits: MultipartLimits
) -> int | HttpFailure | None:
    values = [
        value for name, value in headers if name.lower() == b"content-length"
    ]
    if len(values) > 1:
        return _malformed("duplicate Content-Length")
    if not values:
        return None
    try:
        value = values[0].decode("ascii")
    except UnicodeDecodeError:
        return _malformed("invalid Content-Length")
    if not value.isdecimal():
        return _malformed("invalid Content-Length")
    if int(value) > limits.envelope_bytes:
        return _multipart_too_large("multipart envelope exceeds limit")
    return int(value)


def _disposition_name(headers: Mapping[bytes, bytes]) -> str | HttpFailure:
    raw = headers.get(b"content-disposition")
    if raw is None:
        return _malformed("part is missing Content-Disposition")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        return _malformed("part Content-Disposition must be ASCII")
    pieces = [piece.strip() for piece in text.split(";")]
    if not pieces or pieces[0].lower() != "form-data":
        return _malformed("part Content-Disposition must be form-data")
    names = [
        piece[5:].strip('"')
        for piece in pieces[1:]
        if piece.lower().startswith("name=")
    ]
    if len(names) != 1 or not names[0]:
        return _malformed("part requires one name parameter")
    return names[0]


# @spec ING-HTTP-005, ING-HTTP-007, ING-HTTP-010
def classify_request_limit(
    *,
    encoded_audio_bytes: int,
    multipart_envelope_bytes: int,
    limits: MultipartLimits,
) -> HttpFailure | None:
    """Project semantic audio and defensive multipart limits distinctly."""
    if multipart_envelope_bytes > limits.envelope_bytes:
        return _multipart_too_large("multipart envelope exceeds limit")
    if encoded_audio_bytes > limits.encoded_audio_bytes:
        return _detail("request_too_large", "audio too long")
    return None


def _sample_count(samples: object) -> int:
    """Return a finite normalized sample-piece length for admission credit."""
    try:
        count = len(samples)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("normalized samples must have a length") from error
    if type(count) is not int or count < 0:
        raise TypeError("normalized sample length must be a non-negative int")
    return count


def _handoff_pieces(samples: object, *, maximum: int) -> Iterator[object]:
    """Yield arbitrary whole-sample credit pieces, never cadence segments."""
    count = _sample_count(samples)
    for start in range(0, count, maximum):
        yield samples[start : start + maximum]  # type: ignore[index]


async def _bounded_feed(
    owner: DirectLeaseOwner,
    samples: object,
    outstanding: list[int],
    *,
    maximum: int,
) -> list[str]:
    """Submit one piece only when exact accepted-sample credit permits it."""
    count = _sample_count(samples)
    if not count:
        return []
    if outstanding[0] + count > maximum:
        raise _PreSubmitOverflow
    outstanding[0] += count
    credit: int | None = None

    def on_accepted(accepted: int) -> None:
        nonlocal credit
        if type(accepted) is not int or accepted != count or credit is not None:
            raise _InvalidAcceptanceCredit
        credit = accepted
        outstanding[0] -= accepted

    result = await owner.feed(samples, on_accepted=on_accepted)
    if credit != count:
        raise _InvalidAcceptanceCredit
    return result


# @spec ING-HTTP-004, ING-HTTP-005, ING-HTTP-006, ING-HTTP-007, ING-HTTP-011
async def parse_transcription_multipart(
    *,
    receive: AsgiReceive,
    headers: list[tuple[bytes, bytes]],
    limits: MultipartLimits,
) -> dict[str, str | bytes] | HttpFailure:
    """Parse a bounded multipart body directly from raw ASGI receive chunks."""
    boundary = _parse_content_type(headers, limits)
    if isinstance(boundary, HttpFailure):
        return boundary
    content_length = _parse_content_length(headers, limits)
    if isinstance(content_length, HttpFailure):
        return content_length

    fields: dict[str, str | bytes] = {}
    field_names: set[str] = set()
    current_headers: dict[bytes, bytes] = {}
    current_header_name = bytearray()
    current_header_value = bytearray()
    current_name: str | None = None
    current_data = bytearray()
    current_is_file = False
    part_headers = 0
    part_count = 0
    ended = False
    envelope_bytes = 0

    def fail(failure: HttpFailure) -> Never:
        raise _FailureRaised(failure)

    def on_part_begin() -> None:
        nonlocal current_headers, current_header_name, current_header_value
        nonlocal \
            current_name, \
            current_data, \
            current_is_file, \
            part_headers, \
            part_count
        part_count += 1
        if part_count > limits.part_count:
            fail(_multipart_too_large("multipart part count exceeds limit"))
        current_headers = {}
        current_header_name = bytearray()
        current_header_value = bytearray()
        current_name = None
        current_data = bytearray()
        current_is_file = False
        part_headers = 0

    def add_header_bytes(
        data: bytes, start: int, end: int, destination: bytearray
    ) -> None:
        nonlocal part_headers
        portion = data[start:end]
        if part_headers + len(portion) > limits.part_header_bytes:
            fail(_multipart_too_large("multipart part headers exceed limit"))
        part_headers += len(portion)
        destination.extend(portion)

    def on_header_field(data: bytes, start: int, end: int) -> None:
        add_header_bytes(data, start, end, current_header_name)

    def on_header_value(data: bytes, start: int, end: int) -> None:
        add_header_bytes(data, start, end, current_header_value)

    def on_header_end() -> None:
        if not current_header_name:
            fail(_malformed("multipart header name is empty"))
        try:
            name = bytes(current_header_name).lower()
            value = bytes(current_header_value)
        except ValueError:
            fail(_malformed("malformed multipart header"))
        if name in current_headers:
            fail(_malformed("duplicate multipart part header"))
        current_headers[name] = value

    def on_headers_finished() -> None:
        nonlocal current_name, current_is_file
        name = _disposition_name(current_headers)
        if isinstance(name, HttpFailure):
            fail(name)
        if name in field_names:
            fail(_malformed(f"duplicate multipart field: {name}"))
        current_name = name
        current_is_file = name == "file"
        if current_is_file and "file" in fields:
            fail(_malformed("multipart request contains more than one file"))

    def on_part_data(data: bytes, start: int, end: int) -> None:
        portion = data[start:end]
        if current_name is None:
            fail(_malformed("multipart part data preceded headers"))
        limit = (
            limits.encoded_audio_bytes
            if current_is_file
            else limits.text_field_bytes
        )
        if len(current_data) + len(portion) > limit:
            if current_is_file:
                fail(_detail("request_too_large", "audio too long"))
            fail(_multipart_too_large("multipart text field exceeds limit"))
        current_data.extend(portion)

    def on_part_end() -> None:
        if current_name is None:
            fail(_malformed("multipart part has no field name"))
        field_names.add(current_name)
        if current_is_file:
            fields[current_name] = bytes(current_data)
            return
        try:
            fields[current_name] = bytes(current_data).decode("utf-8")
        except UnicodeDecodeError:
            fail(
                _malformed(f"multipart text field is not UTF-8: {current_name}")
            )

    def on_end() -> None:
        nonlocal ended
        ended = True

    parser = MultipartParser(
        boundary,
        {
            "on_part_begin": on_part_begin,
            "on_header_field": on_header_field,
            "on_header_value": on_header_value,
            "on_header_end": on_header_end,
            "on_headers_finished": on_headers_finished,
            "on_part_data": on_part_data,
            "on_part_end": on_part_end,
            "on_end": on_end,
        },
        max_header_size=limits.envelope_bytes,
    )
    try:
        while True:
            message = await receive()
            message_type = message.get("type")
            if message_type == "http.disconnect":
                return _malformed("client disconnected during multipart body")
            if message_type != "http.request":
                return _malformed(
                    "unexpected ASGI message during multipart body"
                )
            body = message.get("body", b"")
            if not isinstance(body, bytes):
                return _malformed("ASGI body is not bytes")
            if envelope_bytes + len(body) > limits.envelope_bytes:
                return _multipart_too_large("multipart envelope exceeds limit")
            envelope_bytes += len(body)
            parser.write(body)
            if not bool(message.get("more_body", False)):
                break
        parser.finalize()
    except _FailureRaised as error:
        return error.failure
    except MultipartParseError:
        return _malformed("invalid multipart framing")
    if not ended:
        return _malformed("truncated multipart body")
    if content_length is not None and envelope_bytes != content_length:
        return _malformed("Content-Length does not match body")
    if "file" not in fields:
        return _malformed("multipart request is missing file")
    return fields


class NimHttpTranscriptionEndpoint:
    """Mountable raw-ASGI endpoint backed directly by one RFC-1 lease owner."""

    def __init__(
        self,
        *,
        factory: SessionFactory,
        load_shed: LoadShedGate,
        config: HttpTranscriptionConfig,
        served_model: str,
        locales: frozenset[str],
        admission: HostAdmission | None = None,
        owner_register: OwnerRegister | None = None,
        owner_factory: OwnerFactory = DirectLeaseOwner,
    ) -> None:
        """Bind the host factory, limits, admission, and served-model facts."""
        self._factory = factory
        self._load_shed = load_shed
        self._config = config
        self._served_model = served_model
        self._locales = locales
        self._admission = admission
        self._owner_register = owner_register
        self._owner_factory = owner_factory

    async def __call__(
        self, scope: Mapping[str, object], receive: AsgiReceive, send: AsgiSend
    ) -> None:
        """Serve only the mounted HTTP request through raw ASGI messages."""
        response = await self.handle(scope=scope, receive=receive)
        await send(
            {
                "type": "http.response.start",
                "status": response.status,
                "headers": [(b"content-type", response.content_type)],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": response.body,
                "more_body": False,
            }
        )

    # @spec ING-HTTP-001, ING-HTTP-008, ING-HTTP-009, ING-HTTP-011
    async def handle(
        self, *, scope: Mapping[str, object], receive: AsgiReceive
    ) -> HttpResponse:
        """Run one bounded request through parse, lease, and response."""
        try:
            return await asyncio.wait_for(
                self._handle(scope, receive), self._config.request_timeout
            )
        except TimeoutError:
            return _failure_response(
                _detail("request_timeout", "HTTP request timed out", status=504)
            )

    async def _handle(
        self, scope: Mapping[str, object], receive: AsgiReceive
    ) -> HttpResponse:
        if scope.get("type") != "http" or scope.get("method") != "POST":
            return _failure_response(
                _malformed("HTTP transcription requires POST")
            )
        admission_lease = try_acquire_admission(self._admission)
        if admission_lease is None:
            return _failure_response(
                _detail(
                    "service_unavailable", "service is not ready", status=503
                )
            )
        try:
            return await self._handle_admitted(scope, receive)
        finally:
            admission_lease.release()

    async def _handle_admitted(
        self, scope: Mapping[str, object], receive: AsgiReceive
    ) -> HttpResponse:
        """Run local admission and inference under one host admission lease."""
        headers = scope.get("headers", [])
        if not isinstance(headers, list) or not all(
            isinstance(header, tuple)
            and len(header) == 2
            and isinstance(header[0], bytes)
            and isinstance(header[1], bytes)
            for header in headers
        ):
            return _failure_response(_malformed("invalid ASGI headers"))
        typed_headers = cast(list[tuple[bytes, bytes]], headers)
        registration = await self._load_shed.register("nim_http_transcription")
        if isinstance(registration, LoadShedRejected):
            return _failure_response(
                _detail(
                    registration.code,
                    f"authority: {registration.authority}",
                    status=503,
                )
            )
        assert registration is not None
        owner: DirectLeaseOwner | None = None
        tracked_owner: OwnerToken | None = None
        try:
            if self._owner_register is not None:
                tracked_owner = await self._owner_register()
            parsed = await parse_transcription_multipart(
                receive=receive,
                headers=typed_headers,
                limits=self._config.limits,
            )
            if isinstance(parsed, HttpFailure):
                return _failure_response(parsed)
            validated = self._validate(parsed)
            if isinstance(validated, HttpFailure):
                return _failure_response(validated)
            audio, locale, riff = validated
            frontend = StreamingAudioFrontend(
                encoding=riff.encoding,
                sample_rate_hz=riff.sample_rate_hz,
            )
            samples = frontend.push(
                audio[riff.data_offset : riff.data_offset + riff.data_bytes]
            )
            tail = frontend.flush()
            owner = self._owner_factory(
                self._factory, cleanup_timeout=self._config.cleanup_timeout
            )
            await owner.open(cadence="1120ms", locale=locale)
            outstanding = [0]
            try:
                for normalized in (samples, tail):
                    for piece in _handoff_pieces(
                        normalized,
                        maximum=self._config.pre_submit_max_samples,
                    ):
                        await _bounded_feed(
                            owner,
                            piece,
                            outstanding,
                            maximum=self._config.pre_submit_max_samples,
                        )
            except _PreSubmitOverflow:
                await owner.cancel()
                return _failure_response(
                    _detail("buffer_overflow", "pre_submit_max_samples")
                )
            try:
                transcript = await asyncio.wait_for(
                    owner.complete(), self._config.finalization_timeout
                )
            except TimeoutError:
                await owner.cancel()
                return _failure_response(
                    _detail(
                        "finalization_timeout",
                        "HTTP transcription finalization timed out",
                        status=504,
                    )
                )
            if parsed.get("response_format", "json") == "text":
                return HttpResponse(
                    200,
                    b"text/plain; charset=utf-8",
                    (transcript or "").encode(),
                )
            return _json_response(200, {"text": transcript or ""})
        except asyncio.CancelledError:
            if owner is not None:
                with suppress(BaseException):
                    await owner.cancel()
            raise
        except Exception:
            if owner is not None:
                with suppress(BaseException):
                    await owner.cancel()
            return _failure_response(
                HttpFailure(
                    500,
                    {
                        "error": {
                            "message": "internal",
                            "type": "InternalError",
                            "code": 500,
                        }
                    },
                )
            )
        finally:
            try:
                if tracked_owner is not None:
                    with suppress(BaseException):
                        await tracked_owner.release()
            finally:
                with suppress(BaseException):
                    await registration.release()

    def _validate(
        self, parsed: Mapping[str, str | bytes]
    ) -> tuple[bytes, str, RiffFormat] | HttpFailure:
        allowed = _TEXT_FIELDS | {"file"}
        unknown = set(parsed) - allowed
        if unknown:
            return _detail(
                "invalid_config_field",
                f"unsupported fields: {', '.join(sorted(unknown))}",
            )
        model = parsed.get("model")
        language = parsed.get("language")
        if model is not None and model != self._served_model:
            return _detail("invalid_config_field", "model")
        if language is not None and (
            language == "auto" or language not in self._locales
        ):
            return _detail("unknown_locale", "language")
        if model is None and language is None:
            return _detail("invalid_config_field", "need model or language")
        response_format = parsed.get("response_format", "json")
        if response_format not in {"json", "text"}:
            return _detail("invalid_config_field", "response_format")
        temperature = parsed.get("temperature")
        if temperature is not None:
            try:
                if not math.isfinite(float(temperature)):
                    raise ValueError
            except ValueError:
                return _detail("invalid_config_field", "temperature")
        audio = parsed["file"]
        assert isinstance(audio, bytes)
        limit_failure = classify_request_limit(
            encoded_audio_bytes=len(audio),
            multipart_envelope_bytes=0,
            limits=self._config.limits,
        )
        if limit_failure is not None:
            return limit_failure
        sniffed = sniff_riff(
            audio[: self._config.max_riff_header_bytes],
            max_header_bytes=self._config.max_riff_header_bytes,
        )
        if sniffed is None or not isinstance(sniffed, RiffFormat):
            return _detail("unsupported_format", "file")
        format_failure = validate_format(
            sniffed.encoding, sniffed.sample_rate_hz, sniffed.channels
        )
        if format_failure is not None:
            return _detail(
                "unsupported_format", ", ".join(format_failure.fields)
            )
        bytes_per_sample = 2 if sniffed.encoding == "LINEAR_PCM" else 1
        available = len(audio) - sniffed.data_offset
        if available < sniffed.data_bytes:
            return _detail("invalid_audio", "truncated RIFF data")
        duration = sniffed.data_bytes / (
            bytes_per_sample * sniffed.sample_rate_hz
        )
        if duration > self._config.max_decoded_duration_seconds:
            return _detail("request_too_large", "audio too long")
        return audio, str(language or "auto"), sniffed


def _json_response(status: int, body: object) -> HttpResponse:
    return HttpResponse(
        status, b"application/json", json.dumps(body).encode("utf-8")
    )


def _failure_response(failure: HttpFailure) -> HttpResponse:
    return _json_response(failure.status, failure.body)
