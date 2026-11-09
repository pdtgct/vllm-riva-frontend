"""Raw HTTP over ASGI with test-controlled timing, and multipart bodies.

``HttpExchange`` drives one request through the app on the test's loop with
the body delivered chunk by chunk, a stall before a chosen chunk, a client
disconnect, and sends that block or fail: what the file route's ownership
tests need and ``support.asgi.asgi_http`` does not offer. The multipart
builders write ``multipart/form-data`` bodies in code; ``file_request``
combines them for the file route; ``assert_core_error`` reads core's HTTP
error shape (``vllm/entrypoints/serve/engine/protocol.py``).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from support.clock import settle, until

ROUTE = "/v1/audio/transcriptions"
BOUNDARY = b"support-multipart-boundary"
MiB = 1024 * 1024


# ---------------------------------------------------------------- multipart


def multipart_body(
    fields: Iterable[tuple[str, str]] = (),
    *,
    file: bytes | None = b"RIFF-not-really-audio",
    filename: str = "audio.wav",
    file_content_type: str = "audio/wav",
    boundary: bytes = BOUNDARY,
    file_first: bool = True,
) -> bytes:
    """A ``multipart/form-data`` body with text ``fields`` and one file.

    ``file=None`` writes no file part; ``file_first`` puts it before the
    text parts.
    """
    parts: list[bytes] = []

    def text_part(name: str, value: str) -> bytes:
        return (
            b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="' + name.encode() + b'"'
            b"\r\n\r\n" + value.encode() + b"\r\n"
        )

    def file_part(data: bytes) -> bytes:
        return (
            b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="file"; filename="'
            + filename.encode()
            + b'"\r\n'
            b"Content-Type: "
            + file_content_type.encode()
            + b"\r\n\r\n"
            + data
            + b"\r\n"
        )

    if file is not None and file_first:
        parts.append(file_part(file))
    for name, value in fields:
        parts.append(text_part(name, value))
    if file is not None and not file_first:
        parts.append(file_part(file))
    parts.append(b"--" + boundary + b"--\r\n")
    return b"".join(parts)


def content_type(boundary: bytes = BOUNDARY) -> str:
    return "multipart/form-data; boundary=" + boundary.decode()


def chunked(data: bytes, size: int = 64 * 1024) -> list[bytes]:
    return [data[i : i + size] for i in range(0, len(data), size)] or [b""]


# --------------------------------------------------------------- raw client


@dataclass
class HttpExchange:
    """One raw ASGI HTTP request with test-controlled timing.

    * ``chunks`` are delivered one per ``receive``; ``stall_before`` parks
      the ``receive`` for that chunk index until ``resume()``.
    * After the last chunk a ``receive`` parks until ``disconnect()`` (or
      until the response completed) and then reports ``http.disconnect``.
    * ``block_send`` parks the send of that message index (0 is
      ``http.response.start``) until ``unblock_send()``: a stalled reader.
    * ``fail_send`` makes the send of that message index raise ``OSError``.
    """

    app: Any
    headers: list[tuple[str, str]]
    chunks: list[bytes]
    method: str = "POST"
    path: str = ROUTE
    stall_before: int | None = None
    block_send: int | None = None
    fail_send: int | None = None
    scope: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
    receives: int = 0
    chunks_delivered: int = 0
    task: asyncio.Task[Any] | None = None
    error: BaseException | None = None

    def __post_init__(self) -> None:
        self._disconnected = asyncio.Event()
        self._resume = asyncio.Event()
        self._send_gate = asyncio.Event()
        self.send_blocked = asyncio.Event()
        self.scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": self.method,
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "root_path": "",
            "path": self.path,
            "raw_path": self.path.encode(),
            "query_string": b"",
            "headers": [(b"host", b"testserver")]
            + [(k.lower().encode(), v.encode()) for k, v in self.headers],
        }
        self.original_headers = list(self.scope["headers"])

    # -- the ASGI side ------------------------------------------------------

    async def _receive(self) -> dict[str, Any]:
        self.receives += 1
        index = self.chunks_delivered
        if index < len(self.chunks):
            if self.stall_before is not None and index == self.stall_before:
                await self._resume.wait()
            if self._disconnected.is_set():
                return {"type": "http.disconnect"}
            self.chunks_delivered += 1
            return {
                "type": "http.request",
                "body": self.chunks[index],
                "more_body": self.chunks_delivered < len(self.chunks),
            }
        await self._disconnected.wait()
        return {"type": "http.disconnect"}

    async def _send(self, message: dict[str, Any]) -> None:
        index = len(self.messages)
        if self.fail_send is not None and index == self.fail_send:
            raise OSError("client connection lost during send")
        if self.block_send is not None and index == self.block_send:
            self.send_blocked.set()
            await self._send_gate.wait()
        self.messages.append(message)
        if message["type"] == "http.response.body" and not message.get(
            "more_body", False
        ):
            self._disconnected.set()

    async def _run(self) -> None:
        try:
            await self.app(self.scope, self._receive, self._send)
        except BaseException as exc:  # recorded for the test
            self.error = exc

    # -- the test side --------------------------------------------------------

    def start(self) -> HttpExchange:
        if self.task is None:
            self.task = asyncio.ensure_future(self._run())
        return self

    async def finish(self, timeout: float = 5.0) -> HttpExchange:
        self.start()
        assert self.task is not None
        await asyncio.wait_for(asyncio.shield(self.task), timeout)
        return self

    def disconnect(self) -> None:
        self._disconnected.set()

    def resume(self) -> None:
        self._resume.set()

    def unblock_send(self) -> None:
        self._send_gate.set()

    @property
    def done(self) -> bool:
        return self.task is not None and self.task.done()

    @property
    def starts(self) -> list[dict[str, Any]]:
        return [m for m in self.messages if m["type"] == "http.response.start"]

    @property
    def status(self) -> int | None:
        starts = self.starts
        return starts[0]["status"] if starts else None

    @property
    def response_headers(self) -> dict[str, str]:
        starts = self.starts
        if not starts:
            return {}
        return {
            k.decode().lower(): v.decode()
            for k, v in starts[0].get("headers", [])
        }

    @property
    def body(self) -> bytes:
        return b"".join(
            m.get("body", b"")
            for m in self.messages
            if m["type"] == "http.response.body"
        )

    @property
    def completed(self) -> bool:
        """Whether the response body was sent to its end."""
        return any(
            m["type"] == "http.response.body" and not m.get("more_body", False)
            for m in self.messages
        )

    def json(self) -> Any:
        return json.loads(self.body)


def file_request(
    app: Any,
    fields: Iterable[tuple[str, str]] = (),
    *,
    file: bytes | None = b"RIFF-not-really-audio",
    headers: Iterable[tuple[str, str]] | None = None,
    extra_headers: Iterable[tuple[str, str]] = (),
    with_length: bool = True,
    chunk_size: int = 64 * 1024,
    body: bytes | None = None,
    path: str = ROUTE,
    **kwargs: Any,
) -> HttpExchange:
    """An ``HttpExchange`` for one multipart file request (not started)."""
    data = multipart_body(fields, file=file) if body is None else body
    all_headers = (
        list(headers)
        if headers is not None
        else [("content-type", content_type())]
    )
    if with_length:
        all_headers.append(("content-length", str(len(data))))
    all_headers.extend(extra_headers)
    return HttpExchange(
        app, all_headers, chunked(data, chunk_size), path=path, **kwargs
    )


def error_of(exchange: HttpExchange) -> dict[str, Any]:
    """The ``error`` object of core's error shape."""
    payload = exchange.json()
    assert "error" in payload, payload
    return payload["error"]


def assert_core_error(
    exchange: HttpExchange,
    status: int,
    code: str,
    *,
    param: str | None | object = ...,
) -> None:
    """Assert core's HTTP error shape carrying catalog ``code``.

    The shape is ``{"error": {"message": "<code>: <detail>", "type", "param",
    "code": <status>}}``; ``param`` is checked when given.
    """
    assert exchange.status == status, (exchange.status, exchange.body)
    error = error_of(exchange)
    assert set(error) == {"message", "type", "param", "code"}, error
    assert error["code"] == status
    assert error["message"].startswith(f"{code}:"), error["message"]
    if param is not ...:
        assert error["param"] == param, error


__all__ = [
    "BOUNDARY",
    "MiB",
    "ROUTE",
    "HttpExchange",
    "assert_core_error",
    "chunked",
    "content_type",
    "error_of",
    "file_request",
    "multipart_body",
    "settle",
    "until",
]
