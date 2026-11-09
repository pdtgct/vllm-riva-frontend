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

The factory's cleanup: a ``release`` cancelled after it began hands the
session to the factory, which runs a cleanup task for it (its behaviour is
scripted as ``"cleanup"``; ``Return()`` by default, so it ends at once).
A cleanup task that fails records its exception as a fault (a cancelled one
records none). ``drain(deadline)`` awaits the pending cleanup tasks until
the absolute ``deadline`` on the factory's ``clock`` (``RealClock`` by
default, as the model plugin's factory uses ``time.monotonic()``), keeps any
still running and returns the recorded faults, each exactly once, clearing
them. ``owns(lease)`` tells whether a session's cleanup is still running.

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
    """

    then: Return | Raise = field(default_factory=Return)
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    released: asyncio.Event = field(default_factory=asyncio.Event)
    delay_cancel: bool = False
    cancellations: int = 0

    def release(self, then: Return | Raise | None = None) -> None:
        if then is not None:
            self.then = then
        self.released.set()

    async def park(self) -> None:
        """Wait for ``release()`` as the mode says (see the class doc)."""
        self.entered.set()
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
            # The model's lease is ``aborted`` before its wait, so a caller
            # whose bound cancels ``abort`` can still ``release``.
            self.state = "aborted"
            result = await _perform(behaviour, None)
            self._succeeded("abort")
            return result

    async def release(self) -> None:
        self._record("release", (), {})
        behaviour = self._scripts.next("release")
        if self.state == "released":
            return None
        async with self._call("release"):
            try:
                result = await _perform(behaviour, None)
            except asyncio.CancelledError:
                # A cancelled release hands the session to the factory.
                self.factory._hand_over(self)
                raise
            self.state = "released"
            self.factory.released.append(self)
            self._succeeded("release")
            return result

    def teardown(self) -> None:
        self._record("teardown", (), {})
        self.teardown_calls += 1
        self.state = "torn_down"

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
    """A scriptable, recording session factory."""

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
    ) -> None:
        self.clock = clock if clock is not None else RealClock()
        self.served_cadences: frozenset[str] = frozenset(served_cadences)
        self.locales: frozenset[str] = frozenset(locales)
        self.default_locale = default_locale
        self.final_text = final_text
        self.strict = strict
        self.validate = validate
        self.calls: list[Call] = []
        self.leases: list[FakeLease] = []
        self.released: list[FakeLease] = []
        self.handed_over: list[FakeLease] = []
        #: Faults the next ``drain`` returns (a test appends to it).
        self.faults: list[BaseException] = []
        self.drains: list[float] = []
        self.teardown_calls = 0
        self.closed = False
        #: Tasks ``teardown_all`` cancels and returns (a test adds to it).
        self.tasks: set[asyncio.Task[Any]] = set()
        #: The cleanup task of each session handed to the factory.
        self.cleanup: dict[FakeLease, asyncio.Task[Any]] = {}
        self._collected: set[asyncio.Future[Any]] = set()
        self._scripts = _Scripts()
        self._order = 0

    def constructor(self, engine_client: Any, model_config: Any) -> Any:
        """An entry-point constructor that accepts this factory's model.

        Records the arguments it was called with on ``constructed_with``.
        """
        self.constructed_with = (engine_client, model_config)
        return self

    def script(self, name: str, *behaviours: Behaviour) -> None:
        """Queue behaviours for the next calls of ``name``.

        ``name`` is ``open``, ``drain`` or ``cleanup`` (the cleanup task of
        the next session handed to the factory).
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
        lease = FakeLease(
            self,
            len(self.leases),
            cadence=cadence,
            locale=locale,
            strict=self.strict,
        )
        result = await _perform(behaviour, lease)
        if isinstance(result, FakeLease):
            self.leases.append(result)
        return result

    def owns(self, lease: FakeLease) -> bool:
        """Whether the factory's cleanup of ``lease`` is still running."""
        task = self.cleanup.get(lease)
        return task is not None and not task.done()

    async def drain(self, deadline: float) -> list[BaseException]:
        self._record("factory", "drain", (deadline,), {})
        self.drains.append(deadline)
        behaviour = self._scripts.next("drain")
        pending = {task for task in self.cleanup.values() if not task.done()}
        if pending:
            # ``asyncio.wait`` leaves the tasks running when it is cancelled,
            # so a cleanup still running at the deadline stays owned here.
            with contextlib.suppress(TimeoutError):
                await self.clock.wait_for(asyncio.wait(pending), deadline)
        for task in list(self.cleanup.values()):
            if task.done():
                self._collect(task)
        faults = list(self.faults)
        self.faults.clear()
        return await _perform(behaviour, faults)

    def teardown_all(self) -> set[asyncio.Task[Any]]:
        self._record("factory", "teardown_all", (), {})
        self.teardown_calls += 1
        self.closed = True
        tasks = set(self.tasks)
        tasks.update(t for t in self.cleanup.values() if not t.done())
        for task in tasks:
            task.cancel()
        return tasks

    # -- internals ----------------------------------------------------------

    def _hand_over(self, lease: FakeLease) -> None:
        self.handed_over.append(lease)
        behaviour = self._scripts.next("cleanup")
        task = asyncio.ensure_future(_perform(behaviour, None))
        task.add_done_callback(self._collect)
        self.cleanup[lease] = task

    def _collect(self, task: asyncio.Future[Any]) -> None:
        """Record a finished cleanup task's exception as a fault, once."""
        if task in self._collected:
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
