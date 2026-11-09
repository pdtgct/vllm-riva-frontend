"""In-loop ASGI drivers.

The plugin's tasks live on the test's event loop, so the app must be driven
on that loop. Starlette's ``TestClient`` runs the app on another loop and is
not used. ``httpx.ASGITransport`` serves ordinary HTTP; these drivers cover
what it cannot: WebSockets, chunked request bodies and client disconnects.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterable, Iterable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

Message = dict[str, Any]


def _headers(items: Iterable[tuple[str, str]]) -> list[tuple[bytes, bytes]]:
    return [(k.lower().encode(), v.encode()) for k, v in items]


def _scope(
    kind: str, path: str, *, query: str, headers: list[tuple[bytes, bytes]]
) -> Message:
    return {
        "type": kind,
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "scheme": "ws" if kind == "websocket" else "http",
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
        "root_path": "",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query.encode(),
        "headers": headers,
    }


class AsgiWebSocket:
    """One WebSocket connection to ``app``, driven from the test's loop.

    ``connect()`` sends ``websocket.connect`` and returns ``True`` when the app
    accepted, ``False`` when it closed or denied the handshake. ``receive()``
    returns the next ``websocket.send`` or ``websocket.close`` message;
    ``send_json`` and ``send_bytes`` deliver client messages (``push_json``
    and ``push_text`` do the same synchronously, for a callback that must
    deliver one without yielding); ``disconnect`` sends
    ``websocket.disconnect``. ``consumed`` counts the messages the app has
    taken through ``receive`` (the connect included), so a test can see how
    far the app read. ``close_code`` is set once the app closed.
    """

    def __init__(
        self,
        app: Any,
        url: str = "/v1/realtime",
        *,
        headers: Iterable[tuple[str, str]] = (),
        subprotocols: Iterable[str] = (),
    ) -> None:
        parts = urlsplit(url)
        self.app = app
        self.scope = _scope(
            "websocket",
            parts.path,
            query=parts.query,
            headers=[(b"host", b"testserver"), *_headers(headers)],
        )
        self.scope["subprotocols"] = list(subprotocols)
        self._to_app: asyncio.Queue[Message] = asyncio.Queue()
        self._from_app: asyncio.Queue[Message | None] = asyncio.Queue()
        self.task: asyncio.Task[None] | None = None
        self.accepted = False
        self.close_code: int | None = None
        self.close_reason: str | None = None
        self.denied: Message | None = None
        self.sent: list[Message] = []
        self.received: list[Message] = []
        self.consumed = 0
        self.app_error: BaseException | None = None

    async def connect(self) -> bool:
        self.task = asyncio.create_task(self._run())
        await self._to_app.put({"type": "websocket.connect"})
        message = await self.receive()
        if message["type"] == "websocket.accept":
            self.accepted = True
            return True
        return False

    async def _run(self) -> None:
        try:
            await self.app(self.scope, self._receive, self._send)
        except BaseException as exc:  # recorded, not swallowed silently
            self.app_error = exc
        finally:
            await self._from_app.put(None)

    async def _receive(self) -> Message:
        message = await self._to_app.get()
        self.consumed += 1
        return message

    async def _send(self, message: Message) -> None:
        self.received.append(message)
        await self._from_app.put(message)

    async def receive(self, timeout: float = 5.0) -> Message:
        """The next message from the app.

        One of ``websocket.accept``, ``websocket.send``, ``websocket.close``
        or an HTTP denial.
        """
        message = await asyncio.wait_for(self._from_app.get(), timeout)
        if message is None:
            if self.app_error is not None:
                raise self.app_error
            raise ConnectionError("the app ended without closing")
        if message["type"] == "websocket.close":
            self.close_code = message.get("code", 1000)
            self.close_reason = message.get("reason")
        elif message["type"] == "websocket.http.response.start":
            self.denied = message
        return message

    async def receive_json(self, timeout: float = 5.0) -> Any:
        message = await self.receive(timeout)
        if message["type"] != "websocket.send":
            raise AssertionError(f"expected a send, got {message['type']}")
        text = message.get("text")
        if text is None:
            text = message["bytes"].decode()
        return json.loads(text)

    async def send_text(self, text: str) -> None:
        message = {"type": "websocket.receive", "text": text}
        self.sent.append(message)
        await self._to_app.put(message)

    async def send_json(self, payload: Any) -> None:
        await self.send_text(json.dumps(payload))

    def push_text(self, text: str) -> None:
        """``send_text`` without yielding to the loop."""
        message = {"type": "websocket.receive", "text": text}
        self.sent.append(message)
        self._to_app.put_nowait(message)

    def push_json(self, payload: Any) -> None:
        """``send_json`` without yielding to the loop."""
        self.push_text(json.dumps(payload))

    async def send_bytes(self, data: bytes) -> None:
        message = {"type": "websocket.receive", "bytes": data}
        self.sent.append(message)
        await self._to_app.put(message)

    async def disconnect(self, code: int = 1000) -> None:
        await self._to_app.put({"type": "websocket.disconnect", "code": code})

    async def wait_closed(self, timeout: float = 5.0) -> None:
        """Wait for the app's handler to return."""
        if self.task is not None:
            await asyncio.wait_for(asyncio.shield(self.task), timeout)

    @property
    def done(self) -> bool:
        return self.task is not None and self.task.done()


class BlockingSendWebSocket(AsgiWebSocket):
    """An ``AsgiWebSocket`` whose app-side sends can be made to stall.

    While ``block`` is set, every ``websocket.send`` and ``websocket.close``
    from the app waits on ``unblocked`` (a stalled reader); ``stalled``
    counts the sends that waited and ``parked`` keeps their messages, in
    order, so a test can tell which send stalled. A send cancelled while
    parked is delivered nowhere.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.block = False
        self.unblocked = asyncio.Event()
        self.stalled = 0
        self.parked: list[Message] = []

    async def _send(self, message: Message) -> None:
        if self.block and message["type"] in (
            "websocket.send",
            "websocket.close",
        ):
            self.stalled += 1
            self.parked.append(message)
            await self.unblocked.wait()
        await super()._send(message)


@dataclass
class AsgiResponse:
    """What a raw HTTP request produced."""

    status: int | None = None
    headers: dict[str, str] = field(default_factory=dict)
    chunks: list[bytes] = field(default_factory=list)
    completed: bool = False
    error: BaseException | None = None

    @property
    def body(self) -> bytes:
        return b"".join(self.chunks)

    def json(self) -> Any:
        return json.loads(self.body)


async def asgi_http(
    app: Any,
    method: str,
    url: str,
    *,
    headers: Iterable[tuple[str, str]] = (),
    body: bytes | Iterable[bytes] | AsyncIterable[bytes] = b"",
    disconnect_after: int | None = None,
    hold_body: asyncio.Event | None = None,
    timeout: float = 10.0,
) -> AsgiResponse:
    """Drive one HTTP request through ``app`` with explicit body chunks.

    ``body`` may be bytes, an iterable of chunks (each delivered as its own
    ``http.request`` message with ``more_body`` set until the last) or an
    async iterable. ``disconnect_after`` sends ``http.disconnect`` instead of
    the chunk at that index (0 disconnects before any body). ``hold_body``
    delays the body until the event is set, so a test can observe the handler
    while it waits for the client.
    """
    parts = urlsplit(url)
    scope = _scope(
        "http",
        parts.path,
        query=parts.query,
        headers=[(b"host", b"testserver"), *_headers(headers)],
    )
    scope["method"] = method.upper()
    if isinstance(body, bytes):
        chunks: list[bytes] = [body]
    elif isinstance(body, AsyncIterable):
        chunks = [chunk async for chunk in body]
    else:
        chunks = list(body)
    response = AsgiResponse()
    index = 0
    disconnected = asyncio.Event()

    async def receive() -> Message:
        nonlocal index
        if hold_body is not None:
            await hold_body.wait()
        if disconnect_after is not None and index >= disconnect_after:
            disconnected.set()
            return {"type": "http.disconnect"}
        if index >= len(chunks):
            # Nothing more to send: a well-behaved client only reports a
            # disconnect from here, so park until the response has been sent.
            await disconnected.wait()
            return {"type": "http.disconnect"}
        chunk = chunks[index]
        index += 1
        return {
            "type": "http.request",
            "body": chunk,
            "more_body": index < len(chunks),
        }

    async def send(message: Message) -> None:
        if message["type"] == "http.response.start":
            response.status = message["status"]
            response.headers = {
                k.decode().lower(): v.decode()
                for k, v in message.get("headers", [])
            }
        elif message["type"] == "http.response.body":
            response.chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                response.completed = True
                disconnected.set()

    try:
        await asyncio.wait_for(app(scope, receive, send), timeout)
    except BaseException as exc:
        response.error = exc
    return response
