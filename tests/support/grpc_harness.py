"""gRPC seams.

Two ways to exercise the Riva servicer:

* **Direct servicer calls** with ``FakeServicerContext`` and
  ``RequestStream``: the streaming, files and errors leaf tests. ``abort``
  raises ``RpcAbort`` carrying the status and details (a real ``grpc.aio``
  context raises ``AbortError``); ``cancel()`` marks the RPC cancelled and
  fires the callbacks registered with ``add_done_callback``, which is how the
  servicer observes a client cancellation; ``time_remaining`` counts down
  to the RPC's absolute deadline on the context's clock (the plugin's
  ``FakeClock`` in leaf tests), as a real context counts down in wall
  time.
* **A real ``grpc.aio`` server** through ``serve`` and ``aio_channel``: the
  vehicle leaf tests of ``start``, ``ready``, ``drain`` and ``stop``. The
  plugin binds the address in its configuration and refuses port 0, so a
  test picks a port with ``free_port()``.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from collections.abc import AsyncIterator, Callable, Iterable
from typing import Any

import grpc
import grpc.aio

from support.clock import RealClock


class RpcAbort(Exception):
    """The fake context's terminal projection of ``context.abort``."""

    def __init__(self, code: grpc.StatusCode, details: str) -> None:
        super().__init__(f"{code.name}: {details}")
        self.code = code
        self.details = details


class FakeServicerContext:
    """The subset of ``grpc.aio.ServicerContext`` the servicer uses.

    ``time_remaining`` (seconds, or ``None`` for no deadline) fixes the RPC's
    absolute deadline at construction: ``clock.monotonic() + time_remaining``
    on ``clock`` (``RealClock`` when none is given). ``time_remaining()``
    then returns what is left of it, never below zero, so time that passes
    on that clock (in transit, validation, a gate wait, execution) is spent
    from the one deadline. ``deadline`` is that absolute value.
    """

    def __init__(
        self,
        *,
        time_remaining: float | None = None,
        clock: Any = None,
        peer: str = "ipv4:127.0.0.1:40000",
        metadata: Iterable[tuple[str, str]] = (),
    ) -> None:
        self.clock = clock if clock is not None else RealClock()
        self.deadline: float | None = None
        self.set_time_remaining(time_remaining)
        self._peer = peer
        self._metadata = tuple(metadata)
        self._cancelled = False
        self._done_callbacks: list[Callable[[Any], Any]] = []
        self.aborts: list[RpcAbort] = []
        self.code: grpc.StatusCode | None = None
        self.details: str | None = None
        self.trailing_metadata: tuple[tuple[str, str], ...] = ()
        self.initial_metadata: tuple[tuple[str, str], ...] = ()

    # -- what the servicer reads -------------------------------------------

    def cancelled(self) -> bool:
        return self._cancelled

    def done(self) -> bool:
        return self._cancelled or self.code is not None

    def time_remaining(self) -> float | None:
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - self.clock.monotonic())

    def peer(self) -> str:
        return self._peer

    def invocation_metadata(self) -> tuple[tuple[str, str], ...]:
        return self._metadata

    def add_done_callback(self, callback: Callable[[Any], Any]) -> None:
        self._done_callbacks.append(callback)
        if self._cancelled:
            callback(self)

    def set_code(self, code: grpc.StatusCode) -> None:
        self.code = code

    def set_details(self, details: str) -> None:
        self.details = details

    def set_trailing_metadata(
        self, metadata: Iterable[tuple[str, str]]
    ) -> None:
        self.trailing_metadata = tuple(metadata)

    async def send_initial_metadata(
        self, metadata: Iterable[tuple[str, str]]
    ) -> None:
        self.initial_metadata = tuple(metadata)

    async def abort(self, code: grpc.StatusCode, details: str = "") -> None:
        abort = RpcAbort(code, details)
        self.aborts.append(abort)
        self.code = code
        self.details = details
        raise abort

    # -- what the test drives ------------------------------------------------

    def cancel(self) -> None:
        """The client cancelled: fire the done callbacks once."""
        if self._cancelled:
            return
        self._cancelled = True
        for callback in list(self._done_callbacks):
            callback(self)

    def set_time_remaining(self, seconds: float | None) -> None:
        """Set the RPC deadline ``seconds`` from now (``None``: no deadline)."""
        self.deadline = (
            None if seconds is None else self.clock.monotonic() + seconds
        )


class RequestStream:
    """A test-driven request iterator for ``StreamingRecognize``.

    ``push(message)`` queues a request; ``close()`` ends the stream (the
    client's half-close); ``fail(exc)`` makes the next pull raise. ``pulls``
    counts the servicer's reads and ``waiting`` is set while the servicer is
    blocked in a read with nothing queued, so a test can see that reads are
    paused while the handler feeds.
    """

    _CLOSE = object()

    def __init__(self, messages: Iterable[Any] = ()) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        for message in messages:
            self._queue.put_nowait(message)
        self.pulls = 0
        self.delivered: list[Any] = []
        self.waiting = asyncio.Event()
        self.closed = False

    def push(self, message: Any) -> None:
        self._queue.put_nowait(message)

    def close(self) -> None:
        self.closed = True
        self._queue.put_nowait(self._CLOSE)

    def fail(self, exc: BaseException) -> None:
        self._queue.put_nowait(exc)

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        self.pulls += 1
        if self._queue.empty():
            self.waiting.set()
        try:
            item = await self._queue.get()
        finally:
            self.waiting.clear()
        if item is self._CLOSE:
            raise StopAsyncIteration
        if isinstance(item, BaseException):
            raise item
        self.delivered.append(item)
        return item


def free_port(host: str = "127.0.0.1") -> int:
    """A TCP port that was free when asked (the design refuses port 0)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


async def serve(
    register: Callable[[grpc.aio.Server], Any],
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    options: Iterable[tuple[str, Any]] = (),
) -> tuple[grpc.aio.Server, str]:
    """Start a real ``grpc.aio`` server; ``register(server)`` adds services.

    Returns the server and its ``host:port`` address. The caller stops it.
    """
    server = grpc.aio.server(options=list(options))
    register(server)
    port = port if port is not None else free_port(host)
    address = f"{host}:{port}"
    bound = server.add_insecure_port(address)
    if bound == 0:
        raise OSError(f"could not bind {address}")
    await server.start()
    return server, address


@contextlib.asynccontextmanager
async def aio_channel(address: str) -> AsyncIterator[grpc.aio.Channel]:
    """An insecure channel to ``address``, closed on exit."""
    channel = grpc.aio.insecure_channel(address)
    try:
        yield channel
    finally:
        await channel.close()


async def wait_reachable(address: str, timeout: float = 5.0) -> None:
    """Wait until a channel to ``address`` is ready."""
    async with aio_channel(address) as channel:
        await asyncio.wait_for(channel.channel_ready(), timeout)
