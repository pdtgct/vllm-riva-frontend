"""Bounded fixture teardowns: a stop that hangs fails the test.

A fixture's teardown awaits the plugin's ``stop`` (or core's
``stop_endpoint_plugins``). ``asyncio.wait_for`` is not a bound there: on
expiry it cancels the stop and then waits for it to end, so a stop that
absorbs the cancellation stalls the run. ``await_stop`` waits with
``asyncio.wait`` instead, which returns at its timeout whatever the task
does, then cancels the stop, gives it the same bound to end and fails the
test (the pattern of the ``started`` fixture in ``tests/conftest.py``).

A stop that ends in time never masks the test's own result: its exception,
if any, is dropped, unless ``propagate`` asks for it to be raised.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

#: Real seconds a teardown gives a stop before it fails the test as a
#: settlement hang; the same bound as ``STOP_BOUND`` in ``tests/conftest.py``
#: (generous: a conforming stop under the fakes ends in well under a second).
STOP_BOUND = 30.0


async def await_stop(
    awaitable: Awaitable[Any],
    *,
    what: str = "plugin stop",
    bound: float = STOP_BOUND,
    advance: Callable[[], Awaitable[Any]] | None = None,
    rounds: int = 0,
    propagate: bool = False,
) -> None:
    """Await a teardown's stop within ``bound`` seconds of real time, or fail.

    ``advance`` (with ``rounds``) is awaited up to ``rounds`` times while the
    stop runs, before the bound applies: a stop whose waits run on a
    ``FakeClock`` is driven forward that way.
    """
    stop = asyncio.ensure_future(awaitable)
    if advance is not None:
        for _ in range(rounds):
            if stop.done():
                break
            await advance()
    done, _ = await asyncio.wait({stop}, timeout=bound)
    if not done:
        stop.cancel()
        with contextlib.suppress(BaseException):
            await asyncio.wait({stop}, timeout=bound)
        pytest.fail(
            f"{what} did not finish within {bound} s of real time:"
            " settlement hung",
            pytrace=False,
        )
    if propagate:
        stop.result()
        return
    with contextlib.suppress(BaseException):
        stop.result()


async def stop_served(
    served: Any, *, what: str = "plugin stop", propagate: bool = False
) -> None:
    """``Served.stop`` for a teardown, bounded by ``await_stop``."""
    from support.app import stop_plugins

    if served.started:
        served.started = False
        await await_stop(
            stop_plugins(served.state), what=what, propagate=propagate
        )
