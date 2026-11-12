"""The seam conformance suite's hooks over a session factory.

``tests/session/test_seam_conformance.py`` runs against a ``seam`` fixture:
an adapter that provides the factory, the clock its ``drain`` deadline is
measured on and the hooks the suite needs to arrange each obligation of the
session design ("Factory obligations" 1 to 8, ``ING-CORE-022`` to
``ING-CORE-034``). ``FakeSeam`` is that adapter over ``FakeSessionFactory``.
The model plugin's repository overrides the ``seam`` fixture with an adapter
of the same shape over its real factory, on the RFC 1 + RFC 2 build, so a
hook here names an effect ("keep the generation task awaiting the engine's
abort"), never a fake-only mechanism. Every hook acts on a lease the suite
opened through ``factory.open``.

Hooks:

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
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from support.clock import Clock
from support.fake_session import (
    Block,
    FakeLease,
    FakeSessionFactory,
    Raise,
    Return,
    refused,
)

#: Cancellation sources that cancel unconditionally by contract.
UNCONDITIONAL = frozenset({"teardown_all", "teardown", "aclose"})


@dataclass
class FakeSeam:
    """The suite's hooks over ``FakeSessionFactory``."""

    factory: FakeSessionFactory
    #: The clock of ``factory.drain``'s absolute deadline.
    clock: Clock

    def inject_fault(self, fault: BaseException) -> None:
        self.factory.faults.append(fault)

    def slow(self, lease: FakeLease, name: str) -> Block:
        block = Block()
        lease.script(name, block)
        return block

    def slow_open(
        self,
        *,
        before_generation: bool = False,
        on_enter: Callable[[], Any] | None = None,
    ) -> Block:
        """Keep the next ``open`` suspended until the block is released.

        By default its generation task has taken its first step by then;
        ``before_generation=True`` holds that task before its first step for
        as long as ``open`` is suspended, so cancelling the opener cancels a
        generation task that never started.
        """
        block = Block(on_enter=on_enter)
        self.factory.script("open", block)
        self.factory.gate_next_open = before_generation
        return block

    def refuse_next_open(self, reason: str = "capacity") -> None:
        """Make the next ``open`` refuse with ``reason``, starting nothing."""
        self.factory.script("open", Raise(refused(reason)))

    def slow_abandoned_engine_abort(self) -> Block:
        """Keep the engine abort of the next abandoned session pending."""
        block = Block()
        self.factory.next_open_engine_abort = block
        return block

    def abandoned(self) -> FakeLease | None:
        """The session the last cancelled ``open`` gave up, if any."""
        return self.factory.abandoned[-1] if self.factory.abandoned else None

    def slow_cleanup(self, then: Return | Raise | None = None) -> Block:
        """Keep the next handed-over session's retirement wait running."""
        block = Block() if then is None else Block(then=then)
        self.factory.script("cleanup", block)
        return block

    def slow_engine_abort(self, lease: FakeLease) -> Block:
        block = Block()
        lease.script("engine_abort", block)
        return block

    def fail_engine_abort(self, lease: FakeLease, exc: BaseException) -> Block:
        block = Block(then=Raise(exc))
        lease.script("engine_abort", block)
        return block

    def fail_generation(self, lease: FakeLease, exc: BaseException) -> None:
        lease.fail_generation(exc)

    def generation(self, lease: FakeLease) -> asyncio.Task[Any] | None:
        return lease.generation

    def engine_calls(self, lease: FakeLease) -> list[str]:
        return list(lease.engine_calls)

    def handed_over(self, lease: FakeLease) -> bool:
        return lease in self.factory.handed_over

    def owns(self, lease: FakeLease) -> bool:
        return self.factory.owns(lease)

    def running_tasks(self) -> set[asyncio.Task[Any]]:
        return self.factory.running_tasks()

    def late_cancellations(self) -> list[tuple[str, int]]:
        return [
            (source, index)
            for source, index in self.factory.late_cancellations
            if source not in UNCONDITIONAL
        ]

    def foreign_cancellations(self) -> list[tuple[str, int]]:
        return list(self.factory.foreign_cancellations)


async def reference_settlement(
    factory: Any, clock: Clock
) -> tuple[list[BaseException], list[asyncio.Task[Any]]]:
    """The factory settlement steps of ``ING-EP-096``, for support tests.

    ``drain(math.inf)``, ``teardown_all()``, await every returned task
    (not stopping at an exception, reading none), note each task not done
    when ``teardown_all`` returned, ``drain(now)``. Returns the faults both
    drains returned and the tasks found still running. Only
    ``tests/test_support.py`` uses it, to check the fake against the
    obligations; the frontend's tests exercise the plugin's own settlement.
    """
    import math

    faults = list(await factory.drain(math.inf))
    returned = factory.teardown_all()
    still_running = [task for task in returned if not task.done()]
    if returned:
        await asyncio.wait(returned)
    faults.extend(await factory.drain(clock.monotonic()))
    return faults, still_running
