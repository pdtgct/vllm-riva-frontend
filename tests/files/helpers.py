"""Helpers specific to the files leaf (no frontend imports).

The raw HTTP client, the multipart builders and core's error-shape assertion
are shared (``support.http``), as are the Riva ``Recognize`` builders and
driver (``support.riva``). What stays here observes core's transcription
service in ways only the file ownership tests need: ``GuardedCall`` wraps
``create_transcription`` and ``TrackedStream`` is an event generator that
records pull overlap and the close order. ``SlowWatcherCancel`` makes the
route's disconnect watcher slow to settle when it is cancelled.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterable

from support.fake_engine import FakeEngineClient
from support.fake_transcription import FakeTranscriptionServing
from support.http import HttpExchange

# ---------------------------------------------------------------- core calls


class GuardedCall:
    """Wraps the fake service's ``create_transcription`` to observe it.

    Records whether core's call was cancelled (the owner must never cancel
    it outside ``stop``) and when each call started and ended.
    """

    def __init__(self, service: FakeTranscriptionServing) -> None:
        self.service = service
        self._inner = service.create_transcription
        self.started = 0
        self.returned = 0
        self.cancelled = 0
        service.create_transcription = self  # type: ignore[method-assign]

    async def __call__(self, audio_data, request, raw_request=None):
        self.started += 1
        try:
            result = await self._inner(audio_data, request, raw_request)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        self.returned += 1
        return result


class TrackedStream:
    """An event generator recording pull overlap and the close order.

    ``blocks`` maps an event index to an ``asyncio.Event`` that the pull of
    that index waits for; ``raise_at`` makes that pull raise.
    ``aborts_at_close`` records the engine's abort count when ``aclose`` ran.
    """

    def __init__(
        self,
        events: Iterable[str],
        *,
        engine: FakeEngineClient,
        blocks: dict[int, asyncio.Event] | None = None,
        raise_at: dict[int, BaseException] | None = None,
    ) -> None:
        self.events = list(events)
        self.engine = engine
        self.blocks = dict(blocks or {})
        self.raise_at = dict(raise_at or {})
        self.index = 0
        self.pulls = 0
        self.active = 0
        self.max_active = 0
        self.pull_entered = asyncio.Event()
        self.pulling: int | None = None
        self.closed = False
        self.aborts_at_close: int | None = None
        self.pull_cancelled = 0

    def __aiter__(self) -> TrackedStream:
        return self

    async def __anext__(self) -> str:
        self.pulls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.closed or self.index >= len(self.events):
                raise StopAsyncIteration
            index = self.index
            self.pulling = index
            self.pull_entered.set()
            if index in self.blocks:
                try:
                    await self.blocks[index].wait()
                except asyncio.CancelledError:
                    self.pull_cancelled += 1
                    raise
            if index in self.raise_at:
                raise self.raise_at[index]
            self.index += 1
            return self.events[index]
        finally:
            self.active -= 1
            self.pulling = None

    async def aclose(self) -> None:
        self.closed = True
        self.aborts_at_close = len(self.engine.aborted)


class SlowWatcherCancel:
    """Holds the cancellation of a ``receive`` that waits for the client.

    Wraps an unstarted exchange's ``receive``. After the body, the route's
    disconnect watcher is the only caller waiting in ``receive``; when it is
    cancelled, it does not finish until ``release()`` (or a second
    cancellation), so the watcher settles late. ``cancelled`` is set when
    the cancellation arrives.
    """

    def __init__(self, exchange: HttpExchange) -> None:
        assert exchange.task is None, "wrap the exchange before it starts"
        self._inner = exchange._receive
        self._hold = asyncio.Event()
        self.cancelled = asyncio.Event()
        exchange._receive = self  # type: ignore[method-assign]

    async def __call__(self) -> dict:
        try:
            return await self._inner()
        except asyncio.CancelledError:
            self.cancelled.set()
            await self._hold.wait()
            raise

    def release(self) -> None:
        self._hold.set()


def is_uuid(text: str) -> bool:
    try:
        uuid.UUID(text)
    except ValueError:
        return False
    return True
