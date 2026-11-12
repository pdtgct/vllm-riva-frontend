"""Bounded in-memory ``multipart/form-data`` parsing for the file route.

The file route reads its body itself instead of calling ``request.form()``:
core's form parsing may spool a large upload to disk, and the frontend keeps
request audio in memory only, inside configured bounds. Raw ASGI body chunks
feed ``python_multipart.MultipartParser`` callbacks as they arrive; the
encoded body is never held whole. The one ``file`` part goes to an in-memory
buffer that becomes the upload object handed to core without a copy, and
each text part goes to its own bounded buffer.

Every limit is checked before a buffer grows. Filename, boundary and part
content type are untrusted metadata: they are kept as strings and never used
as a path or persisted.
"""

from __future__ import annotations

import asyncio
import io
import string
import typing
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from fastapi import UploadFile
from python_multipart import MultipartParser
from python_multipart.exceptions import FormParserError, MultipartParseError
from starlette.datastructures import Headers
from starlette.requests import ClientDisconnect

from vllm_riva_frontend.errors import FrontendError

__all__ = [
    "MultipartForm",
    "MultipartLimits",
    "make_upload",
    "parse_content_length",
    "parse_content_type",
    "parse_multipart",
]

#: The name of the one part that carries the audio (core's request field).
FILE_FIELD = "file"

_MEDIA_TYPE = "multipart/form-data"
_TOKEN_CHARS = frozenset(
    string.ascii_letters + string.digits + "!#$%&'*+-.^_`|~"
)
# RFC 2046 boundary characters; a space is allowed inside, never last.
_BOUNDARY_CHARS = frozenset(
    string.ascii_letters + string.digits + "'()+_,-./:=? "
)
_WHITESPACE = " \t"


class _Clock(Protocol):
    def monotonic(self) -> float: ...

    async def sleep(self, seconds: float) -> None:
        """Suspend the caller for ``seconds``."""
        ...


#: Starts a task; an owner passes one that keeps it under its record.
Spawn = Callable[[Coroutine[Any, Any, Any]], asyncio.Future[Any]]


@dataclass(frozen=True)
class MultipartLimits:
    """The byte bounds of one file-route request.

    Attributes:
        envelope_max_bytes: The whole request body.
        content_type_max_bytes: The ``Content-Type`` header value.
        boundary_max_bytes: The boundary token.
        max_parts: Parts per request.
        max_header_bytes: The headers of one part, names and values.
        text_field_max_bytes: The value of one text part.
    """

    envelope_max_bytes: int
    content_type_max_bytes: int
    boundary_max_bytes: int
    max_parts: int
    max_header_bytes: int
    text_field_max_bytes: int


class _PartBuffer(io.BytesIO):
    """The in-memory buffer the parser writes the file part into.

    It is an ``io.BytesIO`` (the upload object wraps it as it is), and it
    compares equal to, and converts to, the bytes it holds.
    """

    __hash__ = io.BytesIO.__hash__

    def __bytes__(self) -> bytes:
        return self.getvalue()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (bytes, bytearray, memoryview)):
            return self.getbuffer() == other
        return NotImplemented


@dataclass
class MultipartForm:
    """A parsed file-route body.

    Attributes:
        fields: The text parts as ``(name, value)`` in arrival order; a
            repeated name appears once per part.
        file: The ``file`` part's bytes in an ``io.BytesIO`` positioned at
            0, or ``None`` when no file part arrived.
        filename: The file part's filename (untrusted metadata), if any.
        content_type: The file part's ``Content-Type`` (untrusted), if any.
    """

    fields: list[tuple[str, str]] = field(default_factory=list)
    file: io.BytesIO | None = None
    filename: str | None = None
    content_type: str | None = None


class _MemoryUpload(UploadFile):
    """An upload whose file is an in-memory ``BytesIO``.

    Reads, seeks and closes run on the event loop: the object never touches
    a thread or the filesystem.
    """

    async def read(self, size: int = -1) -> bytes:
        return self.file.read(size)

    async def seek(self, offset: int) -> None:
        self.file.seek(offset)

    async def write(self, data: bytes) -> None:
        if self.size is not None:
            self.size += len(data)
        self.file.write(data)

    async def close(self) -> None:
        self.file.close()


def _memory_upload(
    buffer: io.BytesIO,
    *,
    filename: str | None = None,
    content_type: str | None = None,
) -> UploadFile:
    """Wrap ``buffer`` (positioned at 0) in an in-memory upload object.

    ``size`` is the buffer's length, as core's own form parser reports it,
    so core's upload limit sees the same value on this path as on its own.
    """
    size = buffer.seek(0, io.SEEK_END)
    buffer.seek(0)
    headers = Headers({"content-type": content_type}) if content_type else None
    return _MemoryUpload(
        file=buffer, size=size, filename=filename, headers=headers
    )


def make_upload(form: MultipartForm) -> UploadFile:
    """Return core's upload object over the form's own file buffer.

    The buffer is handed over, not copied.

    Raises:
        FrontendError: ``malformed_request`` when the form has no file part.
    """
    if form.file is None:
        raise FrontendError(
            "malformed_request", "the request has no file part", (FILE_FIELD,)
        )
    return _memory_upload(
        form.file, filename=form.filename, content_type=form.content_type
    )


# ------------------------------------------------------------- headers


def _is_token(text: str) -> bool:
    return bool(text) and all(c in _TOKEN_CHARS for c in text)


def _skip_whitespace(value: str, index: int) -> int:
    while index < len(value) and value[index] in _WHITESPACE:
        index += 1
    return index


def _quoted(value: str, index: int) -> tuple[str, int]:
    """Read the quoted string opening at ``value[index]``.

    Returns the unescaped text and the index after the closing quote.
    """
    chars: list[str] = []
    index += 1
    while index < len(value):
        char = value[index]
        if char == "\\":
            if index + 1 >= len(value):
                break
            chars.append(value[index + 1])
            index += 2
            continue
        if char == '"':
            return "".join(chars), index + 1
        chars.append(char)
        index += 1
    raise ValueError("unterminated quoted string")


def _split_parameters(value: str) -> tuple[str, list[tuple[str, str, bool]]]:
    """Split a header value into its head and its ``;`` parameters.

    Returns ``(head, [(lowercase name, value, quoted), ...])``.

    Raises:
        ValueError: The parameters are not ``name=token`` or
            ``name="quoted string"`` pairs.
    """
    end = value.find(";")
    head = (value if end < 0 else value[:end]).strip(_WHITESPACE)
    params: list[tuple[str, str, bool]] = []
    index = len(value) if end < 0 else end + 1
    while index < len(value):
        index = _skip_whitespace(value, index)
        if index >= len(value):
            break
        if value[index] == ";":
            index += 1
            continue
        equals = index
        while equals < len(value) and value[equals] not in "=;":
            equals += 1
        name = value[index:equals].strip(_WHITESPACE)
        if equals >= len(value) or value[equals] != "=" or not _is_token(name):
            raise ValueError(f"malformed parameter {value[index:equals]!r}")
        index = _skip_whitespace(value, equals + 1)
        if index < len(value) and value[index] == '"':
            text, index = _quoted(value, index)
            index = _skip_whitespace(value, index)
            if index < len(value) and value[index] != ";":
                raise ValueError("text after a quoted parameter value")
            params.append((name.lower(), text, True))
            continue
        stop = value.find(";", index)
        stop = len(value) if stop < 0 else stop
        text = value[index:stop].strip(_WHITESPACE)
        if not _is_token(text):
            raise ValueError(f"malformed parameter value {text!r}")
        params.append((name.lower(), text, False))
        index = stop
    return head, params


def _valid_boundary(boundary: str, quoted: bool) -> bool:
    if not boundary or boundary.endswith(" "):
        return False
    if not quoted and " " in boundary:
        return False
    return all(c in _BOUNDARY_CHARS for c in boundary)


# @spec ING-FILE-005, ING-FILE-039
def parse_content_type(value: str, limits: MultipartLimits) -> bytes:
    """Return the boundary of a ``multipart/form-data`` ``Content-Type``.

    ``value`` is the decoded header value. It must name
    ``multipart/form-data`` with exactly one syntactically valid ASCII
    ``boundary`` parameter; anything else is refused before the body is
    read.

    Raises:
        FrontendError: ``malformed_request`` for another media type, a
            missing, duplicate, empty or malformed boundary, non-ASCII text
            or malformed parameters; ``request_too_large`` when a valid
            value or boundary is above its byte cap.
    """
    if not value.isascii():
        raise FrontendError(
            "malformed_request", "the Content-Type is not ASCII"
        )
    try:
        head, params = _split_parameters(value)
    except ValueError as exc:
        raise FrontendError(
            "malformed_request", f"malformed Content-Type: {exc}"
        ) from None
    if head.lower() != _MEDIA_TYPE:
        raise FrontendError(
            "malformed_request",
            f"the request must be {_MEDIA_TYPE}, not {head or 'empty'}",
        )
    boundaries = [
        (text, quoted) for name, text, quoted in params if name == "boundary"
    ]
    if len(boundaries) != 1:
        raise FrontendError(
            "malformed_request",
            "Content-Type must carry exactly one boundary parameter",
        )
    boundary, quoted = boundaries[0]
    if not _valid_boundary(boundary, quoted):
        raise FrontendError(
            "malformed_request", "the multipart boundary is malformed"
        )
    if len(value) > limits.content_type_max_bytes:
        raise FrontendError(
            "request_too_large",
            f"the Content-Type is above {limits.content_type_max_bytes} bytes",
        )
    if len(boundary) > limits.boundary_max_bytes:
        raise FrontendError(
            "request_too_large",
            "the multipart boundary is above "
            f"{limits.boundary_max_bytes} bytes",
        )
    return boundary.encode("ascii")


def _header_values(
    headers: Mapping[str, str] | Iterable[tuple[bytes, bytes]], name: str
) -> list[str]:
    getlist = getattr(headers, "getlist", None)
    if callable(getlist):
        return [str(v) for v in getlist(name)]
    if isinstance(headers, Mapping):
        found = [v for k, v in headers.items() if str(k).lower() == name]
        return [str(v) for v in found]
    target = name.encode("latin-1")
    return [
        v.decode("latin-1") for k, v in headers if bytes(k).lower() == target
    ]


# @spec ING-FILE-039, ING-HTTP-007
def parse_content_length(
    headers: Mapping[str, str] | Iterable[tuple[bytes, bytes]],
    limits: MultipartLimits,
) -> int | None:
    """Check a present ``Content-Length`` against the envelope bound.

    A missing length (a chunked body) is allowed; the running count of the
    body read enforces the same bound.

    Returns:
        The declared length, or ``None`` when no length was sent.

    Raises:
        FrontendError: ``malformed_request`` for an invalid or conflicting
            length; ``request_too_large`` above ``envelope_max_bytes``.
    """
    values = {v.strip() for v in _header_values(headers, "content-length")}
    if not values:
        return None
    if len(values) != 1:
        raise FrontendError(
            "malformed_request", "conflicting Content-Length headers"
        )
    (text,) = values
    if not text.isdigit():
        raise FrontendError("malformed_request", "invalid Content-Length")
    length = int(text)
    if length > limits.envelope_max_bytes:
        raise FrontendError(
            "request_too_large",
            f"the request body is above {limits.envelope_max_bytes} bytes",
        )
    return length


# --------------------------------------------------------------- parsing


@dataclass
class _Part:
    name: str
    is_file: bool
    text: bytearray = field(default_factory=bytearray)


class _FormBuilder:
    """``MultipartParser`` callbacks that build a ``MultipartForm``.

    Each callback checks its bound before a buffer grows and raises
    ``FrontendError`` out of the parser's ``write``.
    """

    def __init__(self, limits: MultipartLimits) -> None:
        self.limits = limits
        self.form = MultipartForm()
        self.parts = 0
        self.ended = False
        self.file_bytes = 0
        self._part: _Part | None = None
        self._headers: list[tuple[bytes, bytes]] = []
        self._header_bytes = 0
        self._field = bytearray()
        self._value = bytearray()

    def callbacks(self) -> dict[str, Callable[..., None]]:
        return {
            "on_part_begin": self.part_begin,
            "on_header_field": self.header_field,
            "on_header_value": self.header_value,
            "on_header_end": self.header_end,
            "on_headers_finished": self.headers_finished,
            "on_part_data": self.part_data,
            "on_part_end": self.part_end,
            "on_end": self.end,
        }

    def part_begin(self) -> None:
        self.parts += 1
        if self.parts > self.limits.max_parts:
            raise FrontendError(
                "request_too_large",
                f"the request has more than {self.limits.max_parts} parts",
            )
        self._part = None
        self._headers = []
        self._header_bytes = 0
        self._field.clear()
        self._value.clear()

    def _grow_header(
        self, buffer: bytearray, data: bytes, start: int, end: int
    ) -> None:
        self._header_bytes += end - start
        if self._header_bytes > self.limits.max_header_bytes:
            raise FrontendError(
                "request_too_large",
                "the headers of a part are above "
                f"{self.limits.max_header_bytes} bytes",
            )
        buffer += memoryview(data)[start:end]

    def header_field(self, data: bytes, start: int, end: int) -> None:
        self._grow_header(self._field, data, start, end)

    def header_value(self, data: bytes, start: int, end: int) -> None:
        self._grow_header(self._value, data, start, end)

    def header_end(self) -> None:
        self._headers.append((bytes(self._field), bytes(self._value)))
        self._field.clear()
        self._value.clear()

    def headers_finished(self) -> None:
        disposition: str | None = None
        content_type: str | None = None
        for raw_name, raw_value in self._headers:
            try:
                name = raw_name.decode("ascii").strip().lower()
                value = raw_value.decode("utf-8").strip(_WHITESPACE)
            except UnicodeDecodeError:
                raise _malformed("a part header is not valid text") from None
            if name == "content-disposition":
                if disposition is not None:
                    raise _malformed("a part repeats Content-Disposition")
                disposition = value
            elif name == "content-type":
                content_type = value
        if disposition is None:
            raise _malformed("a part has no Content-Disposition")
        part_name, filename = _form_data_name(disposition)
        is_file = part_name == FILE_FIELD
        if is_file:
            if self.form.file is not None:
                raise _malformed("the request has a second file part")
            self.form.file = _PartBuffer()
            self.form.filename = filename
            self.form.content_type = content_type
        self._part = _Part(part_name, is_file)

    def part_data(self, data: bytes, start: int, end: int) -> None:
        part = self._part
        if part is None:
            raise _malformed("part data before the part headers")
        size = end - start
        if part.is_file:
            self.file_bytes += size
            if self.file_bytes > self.limits.envelope_max_bytes:
                raise FrontendError(
                    "request_too_large",
                    "the file part is above "
                    f"{self.limits.envelope_max_bytes} bytes",
                )
            assert self.form.file is not None
            self.form.file.write(memoryview(data)[start:end])
            return
        if len(part.text) + size > self.limits.text_field_max_bytes:
            raise FrontendError(
                "request_too_large",
                f"the text part {part.name!r} is above "
                f"{self.limits.text_field_max_bytes} bytes",
                (part.name,),
            )
        part.text += memoryview(data)[start:end]

    def part_end(self) -> None:
        part = self._part
        if part is None:
            raise _malformed("a part ended before its headers")
        if not part.is_file:
            try:
                value = part.text.decode("utf-8")
            except UnicodeDecodeError:
                raise _malformed(
                    f"the text part {part.name!r} is not UTF-8"
                ) from None
            self.form.fields.append((part.name, value))
        self._part = None

    def end(self) -> None:
        self.ended = True

    def finish(self) -> MultipartForm:
        if not self.ended or self._part is not None:
            raise _malformed("the multipart body is truncated")
        if self.form.file is None:
            raise FrontendError(
                "malformed_request",
                "the request has no file part",
                (FILE_FIELD,),
            )
        self.form.file.seek(0)
        return self.form


def _malformed(detail: str) -> FrontendError:
    return FrontendError("malformed_request", detail)


def _form_data_name(disposition: str) -> tuple[str, str | None]:
    """Return the part name and filename of a ``form-data`` disposition."""
    try:
        head, params = _split_parameters(disposition)
    except ValueError as exc:
        raise _malformed(f"malformed Content-Disposition: {exc}") from None
    if head.lower() != "form-data":
        raise _malformed("a part's Content-Disposition is not form-data")
    seen: dict[str, str] = {}
    for name, value, _ in params:
        if name in seen:
            raise _malformed(f"Content-Disposition repeats {name!r}")
        seen[name] = value
    part_name = seen.get("name")
    if not part_name:
        raise _malformed("a part has no name")
    filename = seen.get("filename", seen.get("filename*"))
    return part_name, filename


def _parser(boundary: bytes, builder: _FormBuilder) -> MultipartParser:
    try:
        return MultipartParser(
            boundary,
            builder.callbacks(),  # type: ignore[arg-type]
            # The parser's own per-line limit sits above the frontend's
            # per-part bound, so the frontend's limit answers first.
            max_header_size=builder.limits.max_header_bytes + 1024,
        )
    except FormParserError:
        raise FrontendError(
            "request_too_large", "the multipart boundary is too long"
        ) from None


# @spec ING-FILE-004, ING-FILE-037, ING-FILE-038, ING-HTTP-007, ING-HTTP-011
# @spec ING-EP-106
async def parse_multipart(
    receive: Callable[[], Awaitable[Mapping[str, Any]]],
    boundary: bytes,
    limits: MultipartLimits,
    *,
    deadline: float,
    clock: _Clock,
    spawn: Spawn | None = None,
) -> MultipartForm:
    """Read and parse the request body from ASGI ``receive``.

    Each chunk goes to the parser as it arrives and is not retained, under
    a running count bounded by ``envelope_max_bytes``; the whole read is
    bounded by the absolute ``deadline`` on ``clock``. Truncated input or a
    disconnect is never a success, and what was buffered is dropped with
    the parser. The read and its timer run as two tasks started by
    ``spawn``, so an owner keeps both under its registry record; a
    cancellation of the caller cancels both and propagates.

    Raises:
        FrontendError: ``malformed_request`` for bad framing, a truncated
            body, a malformed part, a second or a missing file part;
            ``request_too_large`` when the body, part count, part headers,
            a text part or the file part passes its bound;
            ``request_timeout`` when the read outlasts ``deadline``.
        ClientDisconnect: The client disconnected before the body ended.
    """
    start = spawn or asyncio.ensure_future
    reader = start(_read(receive, boundary, limits))
    timer = start(clock.sleep(max(0.0, deadline - clock.monotonic())))
    try:
        await asyncio.wait({reader, timer}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        timer.cancel()
        if not reader.done():
            reader.cancel()
    if not reader.done() or reader.cancelled():
        raise FrontendError(
            "request_timeout",
            "the request body was not received within http_body_read_timeout",
        )
    return typing.cast(MultipartForm, reader.result())


async def _read(
    receive: Callable[[], Awaitable[Mapping[str, Any]]],
    boundary: bytes,
    limits: MultipartLimits,
) -> MultipartForm:
    """The body read of ``parse_multipart``, without its deadline."""
    builder = _FormBuilder(limits)
    parser = _parser(boundary, builder)
    total = 0
    while True:
        message = await receive()
        kind = message.get("type")
        if kind == "http.disconnect":
            raise ClientDisconnect()
        if kind != "http.request":
            continue
        body = message.get("body", b"")
        if body:
            total += len(body)
            if total > limits.envelope_max_bytes:
                raise FrontendError(
                    "request_too_large",
                    "the request body is above "
                    f"{limits.envelope_max_bytes} bytes",
                )
            try:
                parser.write(body)
            except MultipartParseError as exc:
                raise _malformed(f"malformed multipart body: {exc}") from None
        if not message.get("more_body", False):
            break
    parser.finalize()
    return builder.finish()
