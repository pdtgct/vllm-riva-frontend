"""The seam conformance suite of ``vllm-riva-frontend``, for tests only.

The module ``vllm_riva_frontend_dev.seam_conformance`` holds the tests: what
the frontend's stream owner relies on in a session factory (the session
design's "Seam conformance" and "Factory obligations" 1 to 8,
``ING-CORE-022`` to ``ING-CORE-034``). This package exports the values an
adapter needs to arrange each case. Importing it imports neither ``pytest``
nor the suite.

Consumer contract:

* ``conftest.py`` calls
  ``pytest.register_assert_rewrite("vllm_riva_frontend_dev.seam_conformance")``
  before the suite is imported.
* A test module star-imports the suite
  (``from vllm_riva_frontend_dev.seam_conformance import *``) and defines a
  ``seam`` fixture returning its adapter. Each test carries its own
  ``asyncio`` mark, so strict and auto mode both run it.
* The suite imports frontend modules (``lease``, ``errors``, ``admission``,
  ``config``, ``lifecycle``) only when its tests run, so collecting it needs
  no vLLM tree and running it needs what the frontend imports, a vLLM tree
  included.

The adapter is the ``seam`` fixture's object. It provides ``factory`` (the
factory under test), ``clock`` (the clock its ``drain`` deadline is measured
on: ``monotonic()``, and ``advance(seconds)`` where time is virtual) and
these hooks. Each hook names an effect ("keep the generation task awaiting
the engine's abort"), never a fake-only mechanism, and acts on a lease the
suite opened through ``factory.open``:

* ``inject_fault(fault)``: make ``fault`` a retained fault, as a cleanup
  task's failed retirement wait would.
* ``slow(lease, name)``: keep the lease call ``name`` (``feed``, ``flush``,
  ``abort``, ``release``, ...) suspended until the returned ``Block`` is
  released; ``slow_open()`` does the same for the next ``open``, after its
  generation task has taken its first step or, with
  ``before_generation=True``, while that task has not started.
* ``refuse_next_open(reason)``: make the next ``open`` an ordinary refusal
  (``session_error`` ``"refused"`` with ``reason``, ``capacity`` for a full
  engine) that starts no session, as the factory's own admission refuses.
* ``slow_abandoned_engine_abort()``: keep the engine abort of the next
  ``open``'s session (one the suite then cancels, so the ``open`` gives it
  up) pending until the returned ``Block`` is released (its generation
  task, once cancelled, awaits it);
  ``abandoned()`` returns that session (the last one a cancelled ``open``
  gave up, ``None`` if none), on which ``generation``, ``engine_calls`` and
  ``owns`` work as for a lease.
* ``slow_cleanup()``: keep the retirement wait of the next session handed to
  the factory's cleanup running until released (``then=Raise(exc)`` makes it
  fail).
* ``slow_engine_abort(lease)``: keep the session's generation task, once
  cancelled, awaiting the engine's abort until released;
  ``fail_engine_abort(lease, exc)`` makes that handler raise ``exc`` (after
  the returned ``Block`` is released).
* ``fail_generation(lease, exc)``: end the generation task with ``exc``.
* ``generation(lease)``: the session's generation task (``None`` when the
  factory has none for it).
* ``engine_calls(lease)``: the engine calls the session's generation task
  made, in order (``generate``, ``abort``).
* ``handed_over(lease)``, ``owns(lease)``: whether the session was handed to
  the factory's cleanup, and whether a cleanup or holding task still holds
  it.
* ``running_tasks()``: the factory's tasks not yet done.
* ``late_cancellations()``: cancellations delivered to a generation task
  after its cancellation handler began, other than by ``teardown_all`` or
  ``teardown()``.
* ``foreign_cancellations()``: cancellations of a factory task that the
  factory did not request (the vehicle must cause none, ``ING-EP-098``).

The exported values:

* ``Return(value)``, ``Raise(exc)`` and ``Block()``: what a hook's
  suspended call does. ``Block.park()`` waits for ``release(then)``, then
  the caller performs ``then``.
* ``SeamError(session_error, reason)`` with its constructors ``refused`` and
  ``failed``: an exception carrying the seam's two attributes.
* ``settle()``: yield to the event loop for a number of turns and then while
  callbacks are ready, within a bound.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__version__ = "0.4.0b2"

__all__ = [
    "Block",
    "Raise",
    "Return",
    "SeamError",
    "__version__",
    "failed",
    "refused",
    "settle",
]

#: The default of ``Return``: "the operation's own default", compared by
#: identity.
_DEFAULT: Any = object()

#: The most loop turns ``settle`` spends waiting for the loop to go idle.
SETTLE_LIMIT = 2_000


class SeamError(Exception):
    """An exception carrying the seam's two attributes."""

    def __init__(
        self, session_error: str, reason: str, message: str = ""
    ) -> None:
        """Carry ``session_error`` and ``reason``; ``message`` is the text."""
        super().__init__(message or f"{session_error}: {reason}")
        self.session_error = session_error
        self.reason = reason


def refused(reason: str, message: str = "") -> SeamError:
    """A refusal: ``session_error`` ``"refused"`` with ``reason``."""
    return SeamError("refused", reason, message)


def failed(reason: str, message: str = "") -> SeamError:
    """A failure: ``session_error`` ``"failed"`` with ``reason``."""
    return SeamError("failed", reason, message)


@dataclass
class Return:
    """Return ``value``; ``Return()`` returns the operation's default."""

    value: Any = _DEFAULT


@dataclass
class Raise:
    """Raise ``exc``.

    ``skip_accept`` makes a scripted ``feed`` raise before it calls
    ``on_accepted`` (the model's ``ValueError`` for bad audio behaves so).
    """

    exc: BaseException
    skip_accept: bool = False


@dataclass
class Block:
    """Park the call until ``release()``; then perform ``then``.

    ``entered`` is set when the call has started waiting. A caller cancelled
    while parked propagates the cancellation (and a ``feed`` or ``flush`` then
    fails the lease with ``cancelled``, as the model plugin's does).

    ``delay_cancel=True`` makes a call that delays its cancellation: while
    parked it absorbs every cancellation (counted in ``cancellations``) and
    keeps waiting; ``release()`` then settles it, as cancelled if it was
    cancelled meanwhile and by ``then`` otherwise.

    ``on_enter`` runs synchronously when the call starts waiting, before
    the loop turns again: a test can cancel the calling task there, so the
    cancellation lands at this very suspension (for example a cancelled
    ``open`` whose generation task has not taken its first step).
    """

    then: Return | Raise = field(default_factory=Return)
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    released: asyncio.Event = field(default_factory=asyncio.Event)
    delay_cancel: bool = False
    cancellations: int = 0
    on_enter: Callable[[], object] | None = None

    def release(self, then: Return | Raise | None = None) -> None:
        """Let the parked call go on; ``then`` replaces what it performs."""
        if then is not None:
            self.then = then
        self.released.set()

    async def park(self) -> None:
        """Wait for ``release()`` as the mode says (see the class doc)."""
        self.entered.set()
        if self.on_enter is not None:
            self.on_enter()
        if not self.delay_cancel:
            await self.released.wait()
            return
        while not self.released.is_set():
            try:
                await self.released.wait()
            except asyncio.CancelledError:
                self.cancellations += 1
                task = asyncio.current_task()
                if task is not None:
                    task.uncancel()
        if self.cancellations:
            raise asyncio.CancelledError


async def settle(rounds: int = 50) -> None:
    """Run the loop until it is idle: at least ``rounds`` turns.

    After ``rounds`` turns it keeps yielding while other callbacks are ready
    to run, and returns once two consecutive turns find none (or after
    ``SETTLE_LIMIT`` turns, so a busy poller cannot hang a test). Timers and
    callbacks from other threads do not count as ready. The ready queue is
    read where the loop exposes it; elsewhere only the ``rounds`` turns run.
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
