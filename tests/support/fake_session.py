"""A scriptable session factory and lease.

They satisfy the structural protocol of the HLD's "Session-factory seam" and
the model plugin's session design (``SessionFactory``: ``served_cadences``,
``locales``, ``default_locale``, ``open(*, cadence, locale)``,
``drain(deadline) -> list``, ``teardown_all() -> set``; ``SessionLease``:
``feed(samples, *, on_accepted) -> list[str]``, ``update_locale(locale) ->
None``, ``flush() -> str``, ``finish()``, ``abort()``, ``release()`` and the
synchronous ``teardown()``).

Every call is recorded with its arguments in one global order
(``factory.calls``; a lease also keeps its own ``calls``). Each operation can
be scripted per call with ``script(name, *behaviours)``:

* ``Return(value)``: return ``value`` (``Return()`` returns the default);
* ``Raise(exc)``: raise ``exc``;
* ``Block()``: park until the test calls ``release()``, then do ``then``;
  ``Block(delay_cancel=True)`` is a call that delays its cancellation until
  ``release()``.

``lease.after(name, callback)`` runs ``callback()`` synchronously when a call
of ``name`` succeeds, just before it returns, so a test can act at the
instant a call completes (for example deliver client traffic the moment
``release`` succeeds).

Outstanding credit: a lease counts the samples handed to ``feed`` and not
yet acknowledged through ``on_accepted`` (``outstanding``) and keeps the
largest such count (``peak_outstanding``), so a test can check the
pre-submit bound from the lease's side. As the model plugin's session design
requires, ``on_accepted`` always fires before ``feed`` returns successfully:
by default ``feed`` acknowledges at once, before its scripted behaviour;
with ``defer_acceptance`` set, ``feed`` stays pending until the test calls
``acknowledge()``, which fires the callback while ``feed`` is still running
(``running == "feed"``), and only then performs its behaviour and returns.
So a test can observe the caller's credit at the instant of the callback,
with ``feed`` still blocked.

The factory's tasks model the factory obligations of the session design
("Seam conformance", "Factory obligations" 1 to 8), so a settlement test
sees factory work that outlives the owner that handed a session over:

* **Generation task.** ``open`` starts one per session (``generation``)
  before its own scripted wait. Its first step records the engine call
  ``generate``; it then runs until ``flush`` succeeds (normal end),
  ``fail_generation`` ends it with an exception, or it is cancelled. Its
  cancellation handler records ``abort`` as an engine call and awaits the
  lease's scripted ``"engine_abort"`` behaviour (``Return()`` by default; a
  ``Block`` keeps the handler awaiting the engine's abort; a ``Raise`` makes
  the handler raise), then ends cancelled.
* **Who cancels it.** ``abort`` requests its cancellation (unless its
  handler has begun) before its own scripted behaviour, then awaits its end,
  cancelling nothing more when the behaviour or the wait is cancelled or
  raises;
  a cancelled ``open`` requests its cancellation twice with no suspension in
  between and hands a started session to a holding task (one whose
  generation task never started is dropped, untracked: it never runs its
  body and ends at the next loop turn); ``teardown_all`` and ``teardown()``
  cancel unconditionally. No other factory code cancels it (obligation 5).
  Every request is logged (``cancel_log``); a request made after the
  handler began is in ``late_cancellations``, and a cancellation the
  factory did not request (someone else cancelled a factory task) is in
  ``foreign_cancellations``.
* **Cleanup task.** A ``release`` cancelled at any suspension after it
  began (it has no earlier one, obligation 6) hands the session to one
  cleanup task: it runs the scripted ``"cleanup"`` behaviour (the
  retirement wait), then awaits the generation task's end without
  cancelling it (obligation 2), and only then ends, raising a failed
  retirement's exception.
* **Holding task.** A session whose ``release`` finished its retirement
  wait (or failed it) while its generation task still runs, and a started
  session of a cancelled ``open``, is held by one holding task until the
  generation task ends (``SESS-ABORT-002``).
* **Faults.** Retained (``faults``) and returned by ``drain`` exactly once:
  a cleanup or holding task's exception, and a generation task's exception
  that no lease call surfaced (any but one ``fail_generation`` delivered
  while the owner still held the session, which fails the lease with the
  exception's ``reason`` instead). A ``CancelledError`` is never retained.
* **drain(deadline).** Awaits the cleanup and holding tasks that exist
  when it is called until they end or the absolute ``deadline`` on the
  factory's ``clock`` passes (``math.inf`` waits until they end), cancelling
  none (obligation 1), then returns the retained faults and clears them.
  ``active_drains`` and ``max_concurrent_drains`` show whether callers
  serialized their calls.
* **teardown_all().** Closes the factory, cancels every owned task not yet
  done and returns every task it owns (``owned_tasks()``, plus anything a
  test put in ``tasks``). A closed factory starts no task (obligation 7):
  a hand-over after it is recorded in ``refused_after_close`` instead. A
  later ``teardown_all`` returns only ended tasks and raises nothing, and a
  later ``drain`` returns at once (obligation 8).

``events`` records the task lifecycle in one order (``("generation.start",
n)``, ``("generation.handler", n)``, ``("generation.end", n)``,
``("cleanup.start", n)``, ``("cleanup.end", n)``, ``("holding.start", n)``,
``("holding.end", n)``, ``("teardown_all", None)``), and
``owns(lease)`` tells whether a cleanup or holding task still holds a
session. ``aclose()`` cancels whatever is left at the end of a test.
``start_generation=False`` turns generation tasks off (the earlier model:
no task until a cancelled ``release``).

Seam errors are ``SeamError`` instances defined here, carrying
``session_error`` and ``reason``; no model-plugin class is involved.

The lease enforces the model plugin's rules on the caller: overlapping calls
raise ``RuntimeError``, calls after ``teardown_all`` raise
``SeamError("failed", "cancelled")`` and calls not allowed in the lease's
state raise ``SeamError("failed", <reason>)``. Disable with ``strict=False``.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from support.clock import RealClock

_DEFAULT = object()

REFUSED_REASONS = frozenset(
    {
        "capacity",
        "active_limit",
        "engine_paused",
        "request_id_in_use",
        "cadence",
        "locale",
    }
)
FAILED_REASONS = frozenset(
    {
        "rejected_input",
        "abnormal_end",
        "expired",
        "engine_dead",
        "cancelled",
        "output_capacity",
        "input_capacity",
        "state",
    }
)

#: Allowed lease calls per state (model plugin session design, "Lease
#: states"). ``teardown`` is allowed everywhere.
ALLOWED_CALLS: dict[str, frozenset[str]] = {
    "open": frozenset({"feed", "update_locale", "flush", "abort"}),
    "flushed": frozenset({"finish", "release"}),
    "failed": frozenset({"abort", "finish", "release"}),
    "aborted": frozenset({"finish", "release"}),
    "released": frozenset(),
    "torn_down": frozenset(),
}


class SeamError(Exception):
    """An exception carrying the seam's two attributes."""

    def __init__(
        self, session_error: str, reason: str, message: str = ""
    ) -> None:
        super().__init__(message or f"{session_error}: {reason}")
        self.session_error = session_error
        self.reason = reason


def refused(reason: str, message: str = "") -> SeamError:
    return SeamError("refused", reason, message)


def failed(reason: str, message: str = "") -> SeamError:
    return SeamError("failed", reason, message)


# --------------------------------------------------------------- behaviours


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
    on_enter: Callable[[], Any] | None = None

    def release(self, then: Return | Raise | None = None) -> None:
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


Behaviour = Return | Raise | Block


@dataclass
class Call:
    """One recorded call."""

    order: int
    target: str
    name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class _Scripts:
    def __init__(self) -> None:
        self._queues: dict[str, deque[Behaviour]] = {}

    def script(self, name: str, *behaviours: Behaviour) -> None:
        self._queues.setdefault(name, deque()).extend(behaviours)

    def next(self, name: str) -> Behaviour:
        queue = self._queues.get(name)
        if queue:
            return queue.popleft()
        return Return()

    def pending(self, name: str) -> int:
        return len(self._queues.get(name, ()))


async def _perform(behaviour: Behaviour, default: Any) -> Any:
    if isinstance(behaviour, Block):
        await behaviour.park()
        behaviour = behaviour.then
    if isinstance(behaviour, Raise):
        raise behaviour.exc
    return default if behaviour.value is _DEFAULT else behaviour.value


# -------------------------------------------------------------------- lease


class FakeLease:
    """One scriptable lease. See the module docstring."""

    def __init__(
        self,
        factory: FakeSessionFactory,
        index: int,
        *,
        cadence: str,
        locale: str,
        strict: bool = True,
    ) -> None:
        self.factory = factory
        self.index = index
        self.cadence = cadence
        self.locale = locale
        self.strict = strict
        self.state = "open"
        self.failure_reason: str | None = None
        self.calls: list[Call] = []
        self.accepted: list[int] = []
        self.fed: list[np.ndarray] = []
        self.teardown_calls = 0
        #: Keep ``feed`` pending until ``acknowledge()`` (see module doc).
        self.defer_acceptance = False
        self.peak_outstanding = 0
        self._outstanding = 0
        self._deferred: deque[
            tuple[int, Callable[[int], Any], asyncio.Event]
        ] = deque()
        self._after: dict[str, list[Callable[[], Any]]] = {}
        self._scripts = _Scripts()
        self._running: str | None = None
        #: The session's generation task (``None`` when not modelled).
        self.generation: asyncio.Task[Any] | None = None
        #: Engine calls the generation task made (``generate``, ``abort``).
        self.engine_calls: list[str] = []
        self.generation_started = False
        self.handler_began = False
        self._generation_end = asyncio.Event()
        #: Set when the generation task may take its first engine step.
        self._start_gate = asyncio.Event()
        self._start_gate.set()
        self._generation_error: BaseException | None = None
        self._surfaced = False

    @property
    def target(self) -> str:
        return f"lease:{self.index}"

    def script(self, name: str, *behaviours: Behaviour) -> None:
        """Queue behaviours for the next calls of ``name``."""
        self._scripts.script(name, *behaviours)

    def pending(self, name: str) -> int:
        return self._scripts.pending(name)

    def after(self, name: str, callback: Callable[[], Any]) -> None:
        """Run ``callback()`` when the next call of ``name`` succeeds."""
        self._after.setdefault(name, []).append(callback)

    @property
    def outstanding(self) -> int:
        """Samples handed to ``feed`` and not yet acknowledged."""
        return self._outstanding

    @property
    def running(self) -> str | None:
        """The name of the lease call in progress, or ``None``."""
        return self._running

    def acknowledge(self, count: int | None = None) -> int:
        """Fire held ``on_accepted`` callbacks, oldest first.

        Each callback runs synchronously, while its ``feed`` call is still
        pending; that ``feed`` then goes on to its scripted behaviour.
        ``count`` limits how many; returns how many ran.
        """
        ran = 0
        while self._deferred and (count is None or ran < count):
            samples, on_accepted, acknowledged = self._deferred.popleft()
            self._accept(samples, on_accepted)
            acknowledged.set()
            ran += 1
        return ran

    @property
    def fed_samples(self) -> np.ndarray:
        """Every sample ``feed`` received, concatenated (float32)."""
        if not self.fed:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate([np.asarray(piece) for piece in self.fed])

    def names(self) -> list[str]:
        """The names of this lease's calls, in order."""
        return [call.name for call in self.calls]

    def fail_generation(
        self, exc: BaseException, *, surfaced: bool | None = None
    ) -> None:
        """End the generation task with ``exc`` (a generation failure).

        While the owner still holds the session (``surfaced`` defaults to
        "not handed to a cleanup or holding task") the failure is surfaced:
        the lease fails with ``exc.reason`` (``abnormal_end`` when absent)
        and the factory retains nothing. Otherwise the exception is a
        retained fault (obligation 3).
        """
        if surfaced is None:
            surfaced = not self.factory.owns(self) and (
                self not in self.factory.handed_over
            )
        if surfaced:
            self._surfaced = True
            self._fail(getattr(exc, "reason", None) or "abnormal_end")
        self._generation_error = exc
        self._generation_end.set()

    @property
    def generation_done(self) -> bool:
        return self.generation is None or self.generation.done()

    # -- protocol ---------------------------------------------------------

    async def feed(
        self, samples: np.ndarray, *, on_accepted: Callable[[int], Any]
    ) -> list[str]:
        self._record("feed", (samples,), {"on_accepted": on_accepted})
        behaviour = self._scripts.next("feed")
        async with self._call("feed"):
            if not (isinstance(behaviour, Raise) and behaviour.skip_accept):
                piece = np.asarray(samples)
                self.fed.append(piece)
                count = int(piece.shape[0])
                self._outstanding += count
                self.peak_outstanding = max(
                    self.peak_outstanding, self._outstanding
                )
                if self.defer_acceptance:
                    await self._await_acknowledgement(count, on_accepted)
                else:
                    self._accept(count, on_accepted)
            try:
                result = await _perform(behaviour, [])
            except asyncio.CancelledError:
                self._fail("cancelled")
                raise
            except SeamError as exc:
                if exc.session_error == "failed":
                    self._fail(exc.reason)
                raise
            self._succeeded("feed")
            return result

    async def update_locale(self, locale: str) -> None:
        self._record("update_locale", (locale,), {})
        behaviour = self._scripts.next("update_locale")
        async with self._call("update_locale"):
            try:
                await _perform(behaviour, None)
            except SeamError as exc:
                if exc.session_error == "failed":
                    self._fail(exc.reason)
                raise
            self.locale = locale
            self._succeeded("update_locale")

    async def flush(self) -> str:
        self._record("flush", (), {})
        behaviour = self._scripts.next("flush")
        async with self._call("flush"):
            try:
                text = await _perform(behaviour, self.factory.final_text)
            except asyncio.CancelledError:
                self._fail("cancelled")
                raise
            except SeamError as exc:
                if exc.session_error == "failed":
                    self._fail(exc.reason)
                raise
            self.state = "flushed"
            self._generation_end.set()
            self._succeeded("flush")
            return text

    async def finish(self) -> None:
        self._record("finish", (), {})
        behaviour = self._scripts.next("finish")
        if self.state == "released":
            return None
        async with self._call("finish"):
            result = await _perform(behaviour, None)
            self._succeeded("finish")
            return result

    async def abort(self) -> None:
        self._record("abort", (), {})
        behaviour = self._scripts.next("abort")
        async with self._call("abort"):
            # As the model's lease does, ``abort`` requests the generation
            # task's cancellation and marks the lease ``aborted`` before any
            # suspension, so the cancellation stands even when the scripted
            # behaviour is cancelled or raises, and a caller whose bound
            # cancels ``abort`` can still ``release``.
            generation = self.generation
            if (
                generation is not None
                and not generation.done()
                and not self.handler_began
            ):
                self.factory._cancel(generation, "abort", self)
            self.state = "aborted"
            result = await _perform(behaviour, None)
            if generation is not None and not generation.done():
                # A cancelled wait leaves the generation task running and
                # cancels it no further (obligation 5).
                await asyncio.wait({generation})
            self._succeeded("abort")
            return result

    async def release(self) -> None:
        self._record("release", (), {})
        behaviour = self._scripts.next("release")
        if self.state == "released":
            return None
        async with self._call("release"):
            # Hand-over coverage is in place before the first suspension
            # (obligation 6): ``_call`` does not suspend.
            try:
                result = await _perform(behaviour, None)
            except asyncio.CancelledError:
                # A cancelled release hands the session to the factory.
                self.factory._hand_over(self)
                raise
            except BaseException:
                # A failed retirement wait: a holding task keeps the session
                # until its generation task ends; the failure is surfaced.
                self.factory._hold(self, "failed_retirement")
                raise
            self.state = "released"
            self.factory.released.append(self)
            if not self.generation_done:
                self.factory._hold(self, "generation_running")
            self._succeeded("release")
            return result

    def teardown(self) -> None:
        self._record("teardown", (), {})
        self.teardown_calls += 1
        self.state = "torn_down"
        if self.generation is not None and not self.generation.done():
            self.factory._cancel(self.generation, "teardown", self)

    # -- internals ----------------------------------------------------------

    def _record(self, name: str, args: tuple, kwargs: dict) -> None:
        call = self.factory._record(self.target, name, args, kwargs)
        self.calls.append(call)

    def _accept(self, count: int, on_accepted: Callable[[int], Any]) -> None:
        self._outstanding -= count
        on_accepted(count)
        self.accepted.append(count)

    async def _await_acknowledgement(
        self, count: int, on_accepted: Callable[[int], Any]
    ) -> None:
        """Keep ``feed`` pending until ``acknowledge()`` fires its callback.

        Cancelled first, the samples were never accepted: the entry is
        dropped, its count leaves ``outstanding`` and the lease is broken
        (``cancelled``), as a cancelled ``feed`` breaks the model's lease.
        """
        acknowledged = asyncio.Event()
        entry = (count, on_accepted, acknowledged)
        self._deferred.append(entry)
        try:
            await acknowledged.wait()
        except asyncio.CancelledError:
            if entry in self._deferred:
                self._deferred.remove(entry)
                self._outstanding -= count
            self._fail("cancelled")
            raise

    def _succeeded(self, name: str) -> None:
        for callback in self._after.pop(name, []):
            callback()

    def _fail(self, reason: str) -> None:
        if self.state in ("open", "failed"):
            self.state = "failed"
            self.failure_reason = reason

    def _call(self, name: str) -> _CallGuard:
        return _CallGuard(self, name)

    def _check(self, name: str) -> None:
        if not self.strict:
            return
        if self._running is not None:
            raise RuntimeError(
                f"lease call {name} while {self._running} is running"
            )
        if self.factory.closed:
            raise failed("cancelled", "factory torn down")
        allowed = ALLOWED_CALLS[self.state]
        if name not in allowed:
            reason = (
                self.failure_reason
                if self.state == "failed" and self.failure_reason
                else "state"
            )
            raise failed(reason, f"{name} not allowed in state {self.state}")


class _CallGuard:
    def __init__(self, lease: FakeLease, name: str) -> None:
        self.lease = lease
        self.name = name

    async def __aenter__(self) -> None:
        self.lease._check(self.name)
        self.lease._running = self.name

    async def __aexit__(self, *exc: object) -> None:
        self.lease._running = None


# ------------------------------------------------------------------ factory


class FakeEntryPoint:
    """What ``importlib.metadata.entry_points(group=...)`` yields."""

    def __init__(self, name: str, constructor: Callable[..., Any]) -> None:
        self.name = name
        self.group = "vllm_riva_frontend.session_factories"
        self.value = f"tests.support.fake_session:{name}"
        self._constructor = constructor

    def load(self) -> Callable[..., Any]:
        return self._constructor


def fake_entry_points(
    **constructors: Callable[..., Any],
) -> Callable[..., list[FakeEntryPoint]]:
    """A stand-in for ``importlib.metadata.entry_points``.

    ``fake_entry_points(nemotron=lambda engine, model_config: factory)``
    returns a callable; ``callable(group=GROUP)`` yields one entry point per
    keyword, in order. Any other group yields nothing.
    """
    points = [
        FakeEntryPoint(name, constructor)
        for name, constructor in constructors.items()
    ]

    def entry_points(**params: Any) -> list[FakeEntryPoint]:
        group = params.get("group")
        if group not in (None, "vllm_riva_frontend.session_factories"):
            return []
        return list(points)

    return entry_points


class FakeSessionFactory:
    """A scriptable, recording session factory. See the module docstring."""

    def __init__(
        self,
        *,
        served_cadences: Iterable[str] = ("560ms", "1120ms"),
        locales: Iterable[str] = ("en-US", "de-DE", "auto"),
        default_locale: str | None = "auto",
        final_text: str = "",
        strict: bool = True,
        validate: bool = True,
        clock: Any = None,
        start_generation: bool = True,
    ) -> None:
        self.clock = clock if clock is not None else RealClock()
        self.served_cadences: frozenset[str] = frozenset(served_cadences)
        self.locales: frozenset[str] = frozenset(locales)
        self.default_locale = default_locale
        self.final_text = final_text
        self.strict = strict
        self.validate = validate
        self.start_generation = start_generation
        #: When set, the next ``open`` holds its generation task before its
        #: first engine step until the scripted wait of ``open`` ends, so a
        #: cancelled ``open`` finds a task that never started.
        self.gate_next_open = False
        #: When set, the engine-abort behaviour of the next ``open``'s
        #: session (the generation task's cancellation handler), so a test
        #: can keep the engine abort of a session a cancelled ``open`` gives
        #: up pending.
        self.next_open_engine_abort: Behaviour | None = None
        #: Every session a cancelled or failed ``open`` gave up, in order.
        self.abandoned: list[FakeLease] = []
        self.calls: list[Call] = []
        self.leases: list[FakeLease] = []
        self.released: list[FakeLease] = []
        self.handed_over: list[FakeLease] = []
        #: Faults the next ``drain`` returns (a test may append to it).
        self.faults: list[BaseException] = []
        self.drains: list[float] = []
        self.teardown_calls = 0
        self.closed = False
        #: Extra tasks ``teardown_all`` cancels and returns (a test adds).
        self.tasks: set[asyncio.Task[Any]] = set()
        #: The generation task of each session ``open`` started.
        self.generation: dict[FakeLease, asyncio.Task[Any]] = {}
        #: The cleanup task of each session handed to the factory.
        self.cleanup: dict[FakeLease, asyncio.Task[Any]] = {}
        #: The holding task of each session held until generation ends.
        self.holding: dict[FakeLease, asyncio.Task[Any]] = {}
        #: Why each held session is held (``failed_retirement``,
        #: ``generation_running``, ``cancelled_open``).
        self.held_because: dict[FakeLease, str] = {}
        #: Hand-overs or holds requested after ``teardown_all``.
        self.refused_after_close: list[tuple[str, FakeLease]] = []
        #: ``(source, kind, lease index)`` per cancellation the factory
        #: requested; ``late_cancellations`` holds those made after the
        #: generation task's handler began; ``foreign_cancellations`` the
        #: ``(kind, lease index)`` of cancellations nobody here requested.
        self.cancel_log: list[tuple[str, str, int]] = []
        self.late_cancellations: list[tuple[str, int]] = []
        self.foreign_cancellations: list[tuple[str, int]] = []
        self.events: list[tuple[str, int | None]] = []
        #: Every task ``teardown_all`` returned, over all its calls.
        self.returned: set[asyncio.Task[Any]] = set()
        self.active_drains = 0
        self.max_concurrent_drains = 0
        self._requested: dict[asyncio.Task[Any], int] = {}
        self._collected: set[asyncio.Future[Any]] = set()
        self._scripts = _Scripts()
        self._order = 0

    def constructor(self, engine_client: Any, model_config: Any) -> Any:
        """An entry-point constructor that accepts this factory's model.

        Records the arguments it was called with on ``constructed_with``
        and starts nothing (``ING-CORE-031``).
        """
        self.constructed_with = (engine_client, model_config)
        return self

    def script(self, name: str, *behaviours: Behaviour) -> None:
        """Queue behaviours for the next calls of ``name``.

        ``name`` is ``open``, ``drain`` or ``cleanup`` (the retirement wait
        of the next session handed to the factory's cleanup).
        """
        self._scripts.script(name, *behaviours)

    def names(self, target: str | None = None) -> list[str]:
        """Call names in global order, optionally for one target.

        ``target`` is ``"factory"`` or ``"lease:<n>"``.
        """
        return [
            call.name
            for call in self.calls
            if target is None or call.target == target
        ]

    def owned_tasks(self) -> set[asyncio.Task[Any]]:
        """Every generation, cleanup and holding task, done or not."""
        return (
            set(self.generation.values())
            | set(self.cleanup.values())
            | set(self.holding.values())
        )

    def running_tasks(self) -> set[asyncio.Task[Any]]:
        """The owned tasks (and ``tasks``) not yet done."""
        return {t for t in self.owned_tasks() | self.tasks if not t.done()}

    def owns(self, lease: FakeLease) -> bool:
        """Whether a cleanup or holding task still holds ``lease``."""
        for table in (self.cleanup, self.holding):
            task = table.get(lease)
            if task is not None and not task.done():
                return True
        return False

    async def aclose(self) -> None:
        """Cancel and await what is still running (test teardown only)."""
        running = self.running_tasks()
        for task in running:
            self._cancel(task, "aclose", None)
        if running:
            await asyncio.wait(running)

    # -- protocol ---------------------------------------------------------

    async def open(self, *, cadence: str, locale: str) -> FakeLease:
        self._record(
            "factory", "open", (), {"cadence": cadence, "locale": locale}
        )
        if self.closed:
            raise failed("cancelled", "factory torn down")
        if self.validate:
            if cadence not in self.served_cadences:
                raise refused("cadence", f"cadence {cadence!r} not served")
            if locale not in self.locales:
                raise refused("locale", f"locale {locale!r} unknown")
        behaviour = self._scripts.next("open")
        if isinstance(behaviour, Raise):
            # A refusal before any session exists starts nothing.
            raise behaviour.exc
        lease = FakeLease(
            self,
            len(self.leases),
            cadence=cadence,
            locale=locale,
            strict=self.strict,
        )
        if self.next_open_engine_abort is not None:
            lease.script("engine_abort", self.next_open_engine_abort)
            self.next_open_engine_abort = None
        if self.start_generation:
            if self.gate_next_open:
                self.gate_next_open = False
                lease._start_gate.clear()
            self._start_generation(lease)
        try:
            result = await _perform(behaviour, lease)
        except BaseException:
            self._abandon_open(lease)
            raise
        lease._start_gate.set()
        if isinstance(result, FakeLease):
            self.leases.append(result)
        return result

    async def drain(self, deadline: float) -> list[BaseException]:
        self._record("factory", "drain", (deadline,), {})
        self.drains.append(deadline)
        self.active_drains += 1
        self.max_concurrent_drains = max(
            self.max_concurrent_drains, self.active_drains
        )
        try:
            behaviour = self._scripts.next("drain")
            pending = {
                task
                for table in (self.cleanup, self.holding)
                for task in table.values()
                if not task.done()
            }
            if pending:
                # ``asyncio.wait`` leaves the tasks running when it is
                # cancelled or times out, so the wait cancels no task.
                if deadline == math.inf:
                    await asyncio.wait(pending)
                else:
                    with contextlib.suppress(TimeoutError):
                        await self.clock.wait_for(
                            asyncio.wait(pending), deadline
                        )
            for lease, task in list(self.generation.items()):
                if task.done():
                    self._generation_fault(lease, task)
            for table in (self.cleanup, self.holding):
                for task in list(table.values()):
                    if task.done():
                        self._collect(task)
            faults = list(self.faults)
            self.faults.clear()
            return await _perform(behaviour, faults)
        finally:
            self.active_drains -= 1

    def teardown_all(self) -> set[asyncio.Task[Any]]:
        self._record("factory", "teardown_all", (), {})
        self.events.append(("teardown_all", None))
        self.teardown_calls += 1
        self.closed = True
        tasks = self.owned_tasks() | set(self.tasks)
        for task in tasks:
            if not task.done():
                self._cancel(task, "teardown_all", None)
        self.returned |= tasks
        return tasks

    # -- internals ----------------------------------------------------------

    def _lease_of(self, task: asyncio.Task[Any]) -> tuple[str, int]:
        for kind, table in (
            ("generation", self.generation),
            ("cleanup", self.cleanup),
            ("holding", self.holding),
        ):
            for lease, owned in table.items():
                if owned is task:
                    return kind, lease.index
        return "extra", -1

    def _cancel(
        self, task: asyncio.Task[Any], source: str, lease: FakeLease | None
    ) -> None:
        """Request ``task``'s cancellation, logged with its source."""
        kind, index = self._lease_of(task)
        self.cancel_log.append((source, kind, index))
        if kind == "generation" and lease is None:
            lease = next(
                (le for le, t in self.generation.items() if t is task), None
            )
        if kind == "generation" and lease is not None and lease.handler_began:
            self.late_cancellations.append((source, index))
        self._requested[task] = self._requested.get(task, 0) + 1
        task.cancel()

    def _check_foreign(self, kind: str, index: int) -> None:
        """Record cancellations of the current task nobody here requested."""
        task = asyncio.current_task()
        if task is None:
            return
        foreign = task.cancelling() - self._requested.get(task, 0)
        if foreign > 0:
            self.foreign_cancellations.append((kind, index))
            self._requested[task] = task.cancelling()

    def _start_generation(self, lease: FakeLease) -> None:
        task = asyncio.ensure_future(self._generation_body(lease))
        lease.generation = task
        self.generation[lease] = task
        task.add_done_callback(lambda t: self._generation_done(lease, t))

    async def _generation_body(self, lease: FakeLease) -> None:
        if not lease._start_gate.is_set():
            # Not yet started: a cancellation here ends the task at once,
            # before its body (no engine call, no handler).
            await lease._start_gate.wait()
        lease.generation_started = True
        self.events.append(("generation.start", lease.index))
        lease.engine_calls.append("generate")
        try:
            await lease._generation_end.wait()
        except asyncio.CancelledError:
            self._check_foreign("generation", lease.index)
            lease.handler_began = True
            self.events.append(("generation.handler", lease.index))
            lease.engine_calls.append("abort")
            behaviour = lease._scripts.next("engine_abort")
            try:
                await _perform(behaviour, None)
            except asyncio.CancelledError:
                # A cancellation while awaiting the engine's abort.
                self._check_foreign("generation", lease.index)
                raise
            raise
        if lease._generation_error is not None:
            raise lease._generation_error

    def _generation_done(self, lease: FakeLease, task: asyncio.Task) -> None:
        self.events.append(("generation.end", lease.index))
        self._generation_fault(lease, task)

    def _generation_fault(self, lease: FakeLease, task: asyncio.Task) -> None:
        """Retain an unsurfaced generation exception, once."""
        if task in self._collected:
            return
        self._collected.add(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None and not (
            lease._surfaced and error is lease._generation_error
        ):
            self.faults.append(error)

    def _abandon_open(self, lease: FakeLease) -> None:
        """A cancelled or failed ``open`` gives its session up.

        The generation task is cancelled twice with no suspension between
        the requests (the wait's cancellation and the hand-over); a task
        that already took its first step is covered by a holding task, one
        that has not is dropped and never runs its body.
        """
        self.abandoned.append(lease)
        generation = lease.generation
        if generation is None or generation.done():
            return
        self._cancel(generation, "open", lease)
        self._cancel(generation, "open.hand_over", lease)
        if lease.generation_started:
            self._hold(lease, "cancelled_open")
        else:
            del self.generation[lease]

    def _hand_over(self, lease: FakeLease) -> None:
        if self.closed:
            self.refused_after_close.append(("cleanup", lease))
            return
        self.handed_over.append(lease)
        behaviour = self._scripts.next("cleanup")
        task = asyncio.ensure_future(self._cleanup_body(lease, behaviour))
        task.add_done_callback(self._collect)
        self.cleanup[lease] = task

    async def _cleanup_body(self, lease: FakeLease, behaviour: Any) -> None:
        self.events.append(("cleanup.start", lease.index))
        try:
            failure: BaseException | None = None
            try:
                await _perform(behaviour, None)
            except asyncio.CancelledError:
                self._check_foreign("cleanup", lease.index)
                raise
            except Exception as exc:
                failure = exc
            await self._await_generation(lease, "cleanup")
            if failure is not None:
                raise failure
        finally:
            self.events.append(("cleanup.end", lease.index))

    def _hold(self, lease: FakeLease, why: str) -> None:
        if lease.generation_done:
            return
        if self.closed:
            self.refused_after_close.append(("holding", lease))
            return
        self.held_because[lease] = why
        task = asyncio.ensure_future(self._holding_body(lease))
        task.add_done_callback(self._collect)
        self.holding[lease] = task

    async def _holding_body(self, lease: FakeLease) -> None:
        self.events.append(("holding.start", lease.index))
        try:
            await self._await_generation(lease, "holding")
        finally:
            self.events.append(("holding.end", lease.index))

    async def _await_generation(self, lease: FakeLease, kind: str) -> None:
        """Wait for the generation task's end, never cancelling it."""
        generation = lease.generation
        if generation is None or generation.done():
            return
        try:
            await asyncio.wait({generation})
        except asyncio.CancelledError:
            self._check_foreign(kind, lease.index)
            raise

    def _collect(self, task: asyncio.Future[Any]) -> None:
        """Retain a finished cleanup or holding task's exception, once."""
        if task in self._collected:
            return
        if not task.done():
            return
        self._collected.add(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self.faults.append(error)

    def _record(
        self, target: str, name: str, args: tuple, kwargs: dict
    ) -> Call:
        call = Call(self._order, target, name, args, dict(kwargs))
        self._order += 1
        self.calls.append(call)
        return call
