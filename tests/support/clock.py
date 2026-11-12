"""The clock seam.

The plugin takes one ``Clock`` and bounds every wait through it:
``monotonic()`` for absolute deadlines (``factory.drain``, the shutdown
trigger), ``sleep`` for periodic work (the fault collector) and ``wait_for``
for a bounded await. ``RealClock`` is ``time.monotonic`` and asyncio;
``FakeClock`` keeps virtual time that a test advances explicitly, so a bound
expires when the test says so and never because the machine was slow.

The two agree on ``wait_for``: an awaitable that finishes before the
deadline returns its result; at or after the deadline an awaitable that is
already done returns its result, and anything else is cancelled before it
runs further and the wait raises ``TimeoutError``, as
``asyncio.wait_for(..., 0)`` does. A coroutine handed to ``wait_for`` with
an expired deadline therefore never starts; code that must start a call
(and then cancel it) does so itself. Cancelling the waiter cancels the
awaitable and propagates.

``until`` and ``settle`` are the test's own waits on the loop: ``until``
polls a condition for a bounded stretch of real time (it never moves virtual
time), ``settle`` runs the loop until every ready callback has run, so each
coroutine has reached its next wait (at least ``rounds`` turns, bounded).
``FakeClock.advance`` settles before and after it moves time, so a
coroutine on its way to a timed wait registers that wait before time moves,
however many loop turns it needs. Work on another thread is outside what
``settle`` sees; the implementation layout keeps decode and resample on the
event loop.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeVar

T = TypeVar("T")

#: The most loop turns ``settle`` spends waiting for the loop to go idle.
SETTLE_LIMIT = 2_000


class Clock(Protocol):
    """What the plugin asks of time."""

    def monotonic(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...

    async def wait_for(self, awaitable: Awaitable[T], deadline: float) -> T:
        """Await ``awaitable`` until the absolute ``deadline``.

        Raises ``TimeoutError`` once the deadline has passed, after
        cancelling and awaiting the inner awaitable, like ``asyncio.wait_for``.
        """
        ...

    def timer(self, deadline: float) -> asyncio.Future[None]: ...


class RealClock:
    """``time.monotonic`` and asyncio."""

    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds))

    async def wait_for(self, awaitable: Awaitable[T], deadline: float) -> T:
        remaining = max(0.0, deadline - time.monotonic())
        return await asyncio.wait_for(awaitable, remaining)

    def timer(self, deadline: float) -> asyncio.Future[None]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        handle = loop.call_later(
            max(0.0, deadline - time.monotonic()),
            lambda: future.done() or future.set_result(None),
        )
        future.add_done_callback(lambda _: handle.cancel())
        return future


async def _cancel_for_timeout(task: asyncio.Future[Any]) -> Any:
    """Cancel ``task``, await it and raise ``TimeoutError`` if it ended so.

    A task that finished anyway keeps its outcome, as in
    ``asyncio.wait_for``.
    """
    task.cancel()
    await asyncio.wait({task})
    try:
        return task.result()
    except asyncio.CancelledError as exc:
        raise TimeoutError from exc


class FakeClock:
    """Virtual monotonic time.

    ``monotonic()`` returns the virtual time. ``sleep`` parks the caller until
    ``advance`` moves the time past its wake-up point. ``wait_for`` races the
    awaitable against a park until the absolute deadline. ``advance`` runs the
    loop to idle before and after it moves time, so coroutines scheduled
    before the call register their waits first and woken coroutines run
    before the test continues.
    """

    def __init__(self, start: float = 1_000.0) -> None:
        self._now = start
        self._sleepers: list[tuple[float, int, asyncio.Future[None]]] = []
        self._order = itertools.count()

    def monotonic(self) -> float:
        return self._now

    @property
    def sleepers(self) -> int:
        """How many coroutines are parked in ``sleep`` or ``wait_for``."""
        return len(self._sleepers)

    @property
    def next_wake(self) -> float | None:
        """The earliest wake-up time, or ``None`` when nothing sleeps."""
        return self._sleepers[0][0] if self._sleepers else None

    async def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        await self._park(self._now + seconds)

    def timer(self, deadline: float) -> asyncio.Future[None]:
        """A future resolved when virtual time reaches ``deadline``.

        No task runs it; it counts among ``sleepers`` until it resolves or
        is cancelled.
        """
        future: asyncio.Future[None] = (
            asyncio.get_running_loop().create_future()
        )
        if deadline <= self._now:
            future.set_result(None)
            return future
        entry = (deadline, next(self._order), future)
        heapq.heappush(self._sleepers, entry)

        def forget(_: asyncio.Future[None]) -> None:
            if entry in self._sleepers:
                self._sleepers.remove(entry)
                heapq.heapify(self._sleepers)

        future.add_done_callback(forget)
        return future

    async def _park(self, wake: float) -> None:
        """Wait until virtual time reaches the absolute ``wake``."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        entry = (wake, next(self._order), future)
        heapq.heappush(self._sleepers, entry)
        try:
            await future
        except asyncio.CancelledError:
            if entry in self._sleepers:
                self._sleepers.remove(entry)
                heapq.heapify(self._sleepers)
            raise

    async def wait_for(self, awaitable: Awaitable[T], deadline: float) -> T:
        task: asyncio.Future[Any] = asyncio.ensure_future(awaitable)
        if deadline <= self._now:
            # ``asyncio.wait_for(..., 0)``: a done awaitable keeps its
            # result; anything else is cancelled before it runs further.
            if task.done():
                return task.result()
            return await _cancel_for_timeout(task)
        sleeper = asyncio.ensure_future(self._park(deadline))
        try:
            done, _ = await asyncio.wait(
                {task, sleeper}, return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            task.cancel()
            sleeper.cancel()
            await asyncio.wait({task, sleeper})
            raise
        if task in done:
            sleeper.cancel()
            await asyncio.wait({sleeper})
            return task.result()
        return await _cancel_for_timeout(task)

    async def advance(self, seconds: float, *, settle: int = 10) -> None:
        """Move time forward, wake due sleepers and let the loop run.

        The loop runs to idle first, so coroutines scheduled just before the
        call register their sleeps before time moves; then time moves, due
        sleepers wake and the loop runs to idle again so they make progress.
        ``settle`` is the minimum number of turns of each run.
        """
        await self.settle(settle)
        self._now += seconds
        self._wake_due()
        await self.settle(settle)

    def set(self, now: float) -> None:
        """Jump to an absolute time without yielding to the loop."""
        self._now = now
        self._wake_due()

    async def settle(self, rounds: int = 10) -> None:
        """Run the loop until it is idle (at least ``rounds`` turns)."""
        await settle(rounds)

    def _wake_due(self) -> None:
        while self._sleepers and self._sleepers[0][0] <= self._now:
            _, _, future = heapq.heappop(self._sleepers)
            if not future.done():
                future.set_result(None)


async def until(
    condition: Callable[[], object], *, timeout: float = 2.0, what: str = ""
) -> None:
    """Yield to the loop until ``condition()`` holds (bounded real time)."""

    async def poll() -> None:
        while not condition():
            await asyncio.sleep(0)

    try:
        await asyncio.wait_for(poll(), timeout)
    except TimeoutError:
        raise AssertionError(f"condition not reached: {what}") from None


async def settle(rounds: int = 50) -> None:
    """Run the loop until it is idle: at least ``rounds`` turns.

    After ``rounds`` turns it keeps yielding while other callbacks are ready
    to run, and returns once two consecutive turns find none (or after
    ``SETTLE_LIMIT`` turns, so a busy poller cannot hang a test). Timers and
    callbacks from other threads do not count as ready.
    """
    for _ in range(rounds):
        await asyncio.sleep(0)
    ready = getattr(asyncio.get_running_loop(), "_ready", None)
    if ready is None:
        return
    idle = 0
    for _ in range(SETTLE_LIMIT):
        idle = idle + 1 if not ready else 0
        if idle >= 2:
            return
        await asyncio.sleep(0)
