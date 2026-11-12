"""The clock seam: every bounded wait of the plugin goes through one clock."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable
from typing import Protocol, TypeVar

T = TypeVar("T")


class Clock(Protocol):
    """What the plugin asks of time.

    Deadlines are absolute ``monotonic()`` values, so a deadline keeps its
    value and identity however many waits it bounds.
    """

    def monotonic(self) -> float:
        """Return the current monotonic time in seconds."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Suspend the caller for ``seconds``."""
        ...

    async def wait_for(self, awaitable: Awaitable[T], deadline: float) -> T:
        """Await ``awaitable`` until the absolute ``deadline``.

        At or after the deadline the awaitable is cancelled and
        ``TimeoutError`` is raised, as ``asyncio.wait_for`` does; an
        awaitable handed over with an expired deadline never starts.
        """
        ...

    def timer(self, deadline: float) -> asyncio.Future[None]:
        """Return a future resolved at the absolute ``deadline``.

        No task runs it: a timer of the event loop (or of the clock)
        resolves it, and cancelling the future cancels that timer. One
        whose deadline has passed is resolved at once.
        """
        ...


class RealClock:
    """``time.monotonic`` and asyncio: the clock the plugin uses in service."""

    def monotonic(self) -> float:
        """Return ``time.monotonic()``."""
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        """Sleep on the event loop; a negative duration does not wait."""
        await asyncio.sleep(max(0.0, seconds))

    async def wait_for(self, awaitable: Awaitable[T], deadline: float) -> T:
        """Await ``awaitable`` until the absolute ``deadline``."""
        remaining = max(0.0, deadline - time.monotonic())
        return await asyncio.wait_for(awaitable, remaining)

    def timer(self, deadline: float) -> asyncio.Future[None]:
        """A future resolved by an event-loop timer at ``deadline``."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            future.set_result(None)
            return future

        def fire() -> None:
            if not future.done():
                future.set_result(None)

        handle = loop.call_later(remaining, fire)
        future.add_done_callback(lambda _: handle.cancel())
        return future
