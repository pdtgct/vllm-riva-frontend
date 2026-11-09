"""The session seam and the per-stream owner that drives it.

The model plugin publishes a session factory under the entry-point group
``vllm_riva_frontend.session_factories``; the frontend reaches it only
through the structural protocols below and reads seam errors only through
their ``session_error`` and ``reason`` attributes. ``StreamOwner`` is the one
serialized owner of a live stream on either streaming surface: it makes
every lease call, bounds every call and every wait for client input through
the plugin's clock, selects exactly one terminal cause and runs the
state-aware cleanup.

How a surface drives an owner:

1. Register the owner task, then build ``StreamOwner``; the
   preconfiguration bound starts at construction and the owner attaches
   itself to its record as the shutdown observer.
2. Bound each wait for client input: either through ``receive(awaitable)``,
   which applies the owner's current bound and, after the shutdown trigger,
   reads nothing and raises ``ShutdownObserved``; or by reading ``bounds``
   and handing an expiry to ``fail``.
3. ``open``, ``attach_frontend``, then ``feed`` each decoded slice and
   ``update_locale`` as the dialect asks.
4. End the stream with ``finalize()`` (the final transcript, only after
   ``finish`` and ``release`` succeeded) or ``fail(cause)``.

Every owner method that ends the stream raises ``FrontendError`` carrying
the one selected cause, so the surface projects that cause and nothing
else. ``finalization`` tells a surface whether a final belongs to a
client-selected or a shutdown finalization.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import inspect
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np
import numpy.typing as npt

from .errors import FrontendError, project_seam_error, seam_error

T = TypeVar("T")

#: The entry-point group this package owns for session factories.
SESSION_FACTORIES_GROUP = "vllm_riva_frontend.session_factories"

# @spec ING-CORE-014
#: The cadence every streaming surface opens at.
STREAM_CADENCE = "560ms"


# ------------------------------------------------------------------ protocol


# @spec ING-CORE-013
@runtime_checkable
class SessionLease(Protocol):
    """One engine stream the model plugin's factory opened."""

    async def feed(
        self,
        samples: npt.NDArray[np.float32],
        *,
        on_accepted: Callable[[int], object],
    ) -> list[str]:
        """Hand over one piece; returns the changed cumulative transcripts."""

    async def update_locale(self, locale: str) -> None:
        """Apply a locale to every unit packed after the call."""

    async def flush(self) -> str:
        """Send the final tail; returns the final cumulative transcript."""

    async def finish(self) -> None:
        """End the engine stream normally."""

    async def abort(self) -> None:
        """End the engine stream abnormally."""

    async def release(self) -> None:
        """Return the slot; returns after it retired."""

    def teardown(self) -> None:
        """Synchronous, engine-free teardown for a stopped plugin."""


# @spec ING-CORE-013
@runtime_checkable
class SessionFactory(Protocol):
    """What the frontend uses of the model plugin's session factory."""

    served_cadences: frozenset[str]
    locales: frozenset[str]
    default_locale: str | None

    async def open(self, *, cadence: str, locale: str) -> SessionLease:
        """Open one lease; may raise a refusal or a failure."""

    async def drain(self, deadline: float) -> list[BaseException]:
        """Await orphan cleanup until ``deadline``; return faults once."""

    def teardown_all(self) -> set[asyncio.Task[Any]]:
        """Terminal close; returns the cancelled tasks."""


# ----------------------------------------------------------------- discovery


def _startup_error(failures: list[str]) -> Exception:
    # Imported here: lifecycle imports this module.
    from .lifecycle import StartupError

    return StartupError(failures)


def _close_factory(factory: object) -> None:
    """Tear down a factory that will not be used, so it leaves no task."""
    teardown = getattr(factory, "teardown_all", None)
    if callable(teardown):
        with contextlib.suppress(Exception):
            teardown()


# @spec ING-CORE-012, ING-CORE-017, ING-CORE-014
def discover_session_factory(
    engine_client: Any,
    *,
    selected: str | None,
    entry_points: Callable[..., Iterable[Any]],
) -> object:
    """Find the one session factory that serves the engine's model.

    Every constructor in the group is called with ``(engine_client,
    engine_client.model_config)`` unchanged; ``None`` means it does not serve
    the model, and a raise fails startup with its message. Exactly one must
    accept, unless ``selected`` names an entry point, whose constructor alone
    is then used. The chosen factory must serve ``STREAM_CADENCE``. Every
    failure is a ``StartupError`` naming the entry points; a factory built
    and then not used is torn down.
    """
    points = list(entry_points(group=SESSION_FACTORIES_GROUP))
    names = ", ".join(point.name for point in points) or "none"
    model_config = engine_client.model_config
    if selected is not None:
        chosen = [point for point in points if point.name == selected]
        if not chosen:
            raise _startup_error(
                [
                    f"riva_frontend.session_factory names {selected!r}, which "
                    f"is not an entry point of {SESSION_FACTORIES_GROUP} "
                    f"(entry points: {names})"
                ]
            )
        points = chosen

    accepted: list[tuple[str, object]] = []
    failures: list[str] = []
    first_error: BaseException | None = None
    for point in points:
        try:
            factory = point.load()(engine_client, model_config)
        except Exception as error:
            failures.append(
                f"session factory {point.name!r} failed to construct: {error}"
            )
            first_error = first_error or error
            continue
        if factory is not None:
            accepted.append((point.name, factory))

    if failures:
        for _, factory in accepted:
            _close_factory(factory)
        raise _startup_error(failures) from first_error
    if len(accepted) != 1:
        for _, factory in accepted:
            _close_factory(factory)
        if selected is not None:
            problem = f"session factory {selected!r} does not serve this model"
        elif not accepted:
            problem = (
                f"no session factory in {SESSION_FACTORIES_GROUP} serves "
                f"this model (entry points: {names})"
            )
        else:
            serving = ", ".join(name for name, _ in accepted)
            problem = (
                f"several session factories serve this model: {serving} "
                f"(entry points: {names}); set "
                "riva_frontend.session_factory to select one"
            )
        raise _startup_error([problem])

    name, factory = accepted[0]
    cadences = getattr(factory, "served_cadences", frozenset())
    if STREAM_CADENCE not in cadences:
        _close_factory(factory)
        raise _startup_error(
            [
                f"session factory {name!r} does not serve the streaming "
                f"cadence {STREAM_CADENCE} (it serves: "
                f"{', '.join(sorted(cadences)) or 'none'})"
            ]
        )
    return factory


# @spec ING-CORE-015
def resolve_locale(
    client_locale: str | None, factory: Any, *, field: str
) -> str:
    """The stream's locale: the client's, validated, or the default.

    ``field`` is the dialect's locale field, named in a refusal: a locale
    outside the factory's ``locales`` is ``unknown_locale``; naming none while
    the factory has no ``default_locale`` is ``invalid_config_field``.
    """
    if client_locale:
        if client_locale in factory.locales:
            return client_locale
        raise FrontendError(
            "unknown_locale",
            detail=(
                f"locale {client_locale!r} is not served; served locales: "
                f"{', '.join(sorted(factory.locales))}"
            ),
            fields=(field,),
        )
    default = factory.default_locale
    if default is None:
        raise FrontendError(
            "invalid_config_field",
            detail="the model has no default locale, so a locale is required",
            fields=(field,),
        )
    return str(default)


# ------------------------------------------------------------ owner shapes


class OwnerState(enum.StrEnum):
    """The stream owner's states (session design, The stream owner)."""

    awaiting_configuration = "awaiting_configuration"
    configured = "configured"
    streaming = "streaming"
    finalizing = "finalizing"
    terminal = "terminal"


class LeaseState(enum.StrEnum):
    """The owner's view of its lease's state."""

    open = "open"
    flushed = "flushed"
    failed = "failed"
    aborted = "aborted"
    released = "released"
    torn_down = "torn_down"


#: Phases that are deadlines; ``Cause.deadline`` is set for these.
DEADLINE_PHASES = frozenset(
    {
        "preconfiguration",
        "first_audio",
        "idle",
        "lease_call",
        "duration",
        "finalization",
        "cleanup",
        "rpc_deadline",
        "drain",
    }
)
#: Phases of causes that are not deadlines.
EVENT_PHASES = frozenset(
    {"transport", "protocol", "input", "engine", "factory", "shutdown"}
)

#: The catalog code a deadline phase selects when it expires first.
# @spec ING-ERR-017, ING-ERR-018, ING-ERR-019, ING-ERR-020, ING-ERR-048
_EXPIRY_CODES = {
    "preconfiguration": "configuration_timeout",
    "first_audio": "idle_timeout",
    "idle": "idle_timeout",
    "lease_call": "service_unavailable",
    "duration": "request_timeout",
    "finalization": "finalization_timeout",
    "cleanup": "internal",
    "rpc_deadline": "request_timeout",
    "drain": "service_unavailable",
}
_EXPIRY_DETAILS = {
    "preconfiguration": "configuration and open exceeded the "
    "preconfiguration bound",
    "first_audio": "no audio was accepted within the first-audio bound",
    "idle": "no audio was accepted within the idle bound",
    "lease_call": "a session call exceeded its bound",
    "duration": "the session exceeded max_session_duration",
    "finalization": "finalization exceeded its bound",
    "cleanup": "session cleanup exceeded its bound",
    "rpc_deadline": "the RPC deadline passed",
    "drain": "the server is shutting down",
}


@dataclass(frozen=True)
class Cause:
    """The one terminal cause an owner selected.

    ``phase`` is a deadline phase (``DEADLINE_PHASES``) or an event phase
    (``EVENT_PHASES``); ``deadline`` is the absolute ``clock.monotonic()``
    deadline for a deadline phase and ``None`` otherwise. ``error`` keeps a
    projected seam error (its detail, fields and authority) when the cause
    came from one.
    """

    code: str
    phase: str
    deadline: float | None = None
    error: FrontendError | None = field(default=None, compare=False, repr=False)


@dataclass
class Timer:
    """One owner bound.

    ``deadline`` is the running absolute deadline (``None`` while disarmed
    or suspended); ``suspended`` is the remaining time of a client-idle
    timer held while a lease call is outstanding.
    """

    name: str
    seconds: float | None
    deadline: float | None = None
    suspended: float | None = None

    @property
    def armed(self) -> bool:
        """Whether the timer runs or is held suspended."""
        return self.deadline is not None or self.suspended is not None

    def arm(self, now: float) -> None:
        """Start the full bound from ``now`` (no-op for an unset bound)."""
        self.suspended = None
        self.deadline = None if self.seconds is None else now + self.seconds

    def disarm(self) -> None:
        """Stop the timer."""
        self.deadline = None
        self.suspended = None

    def suspend(self, now: float) -> None:
        """Hold the remaining time while a lease call is outstanding."""
        if self.deadline is not None:
            self.suspended = max(0.0, self.deadline - now)
            self.deadline = None

    def resume(self, now: float) -> None:
        """Run again from ``now`` with the time it kept."""
        if self.suspended is not None:
            self.deadline = now + self.suspended
            self.suspended = None


# @spec ING-LIFE-057
@dataclass
class Bounds:
    """The owner's timers (session design, Bounds)."""

    preconfiguration: Timer
    first_audio: Timer
    idle: Timer
    lease_call: Timer
    duration: Timer
    finalization: Timer
    cleanup: Timer

    @classmethod
    def from_config(cls, config: Any) -> Bounds:
        """The bounds a ``FrontendConfig`` sets (``None`` for an unset one)."""
        return cls(
            preconfiguration=Timer(
                "preconfiguration", config.preconfiguration_timeout
            ),
            first_audio=Timer("first_audio", config.first_audio_timeout),
            idle=Timer("idle", config.session_idle_timeout),
            lease_call=Timer("lease_call", config.lease_call_timeout),
            duration=Timer("duration", config.max_session_duration),
            finalization=Timer(
                "finalization", config.session_finalization_timeout
            ),
            cleanup=Timer("cleanup", config.session_cleanup_timeout),
        )

    def timers(self) -> tuple[Timer, ...]:
        """Every timer, in the layout's order."""
        return (
            self.preconfiguration,
            self.first_audio,
            self.idle,
            self.lease_call,
            self.duration,
            self.finalization,
            self.cleanup,
        )


class ShutdownObserved(Exception):
    """``receive`` after the shutdown trigger: read nothing, finalize.

    A surface's main loop answers it with ``finalize()`` (the shutdown
    finalization); a receive observer of a running client-selected
    finalization simply stops.
    """

    def __init__(self, trigger: Any) -> None:
        """Carry the trigger the owner observed."""
        super().__init__("the server is shutting down")
        self.trigger = trigger


def _trigger_of(error: BaseException) -> Any:
    """The shutdown trigger a cancellation carries as its message, if any."""
    try:
        from .lifecycle import ShutdownTrigger
    except ImportError:  # pragma: no cover - lifecycle always defines it
        ShutdownTrigger = None
    for arg in error.args:
        if ShutdownTrigger is not None and isinstance(arg, ShutdownTrigger):
            return arg
        if type(arg).__name__ == "ShutdownTrigger":
            return arg
    return None


def _uncancel() -> None:
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        task.uncancel()


def _close(awaitable: object) -> None:
    """Close an unawaited coroutine so it never warns."""
    if inspect.iscoroutine(awaitable):
        awaitable.close()


# -------------------------------------------------------------------- owner


# @spec ING-LIFE-058, ING-LIFE-056, ING-CORE-009, ING-CORE-010, ING-CORE-018
class StreamOwner:
    """The one serialized owner of one live stream.

    It makes every lease call itself, one at a time, each as a child task of
    its registry record so a call can be cancelled without cancelling the
    surface. Exactly one terminal cause is selected; cleanup runs once, in
    one shielded task, as the lease's state allows, and it unregisters the
    record when it ends.
    """

    def __init__(
        self,
        *,
        lease_opener: Callable[..., Awaitable[Any]],
        surface: str,
        config: Any,
        clock: Any,
        registry: Any,
        record: Any,
        catalog: Any,
    ) -> None:
        """Bind one registered stream; its preconfiguration bound starts."""
        self._opener = lease_opener
        self.surface = surface
        self._config = config
        self._clock = clock
        self._registry = registry
        self._record = record
        self._catalog = catalog
        self.bounds = Bounds.from_config(config)
        self.bounds.preconfiguration.arm(clock.monotonic())
        #: The lease-method names called, in order.
        self.calls: list[str] = []
        #: Failures with reason ``engine_dead`` this owner met (escalations).
        self.escalations: list[BaseException] = []
        #: Called with each escalation, when set.
        self.on_escalation: Callable[[BaseException], object] | None = None
        #: Secondary failures recorded during cleanup.
        self.cleanup_errors: list[BaseException] = []
        self._state = OwnerState.awaiting_configuration
        self._lease: Any = None
        self._lease_state: LeaseState | None = None
        self._cause: Cause | None = None
        self._frontend: Any = None
        self._outstanding = 0
        self._call: asyncio.Future[Any] | None = None
        self._call_name: str | None = None
        self._cleanup: asyncio.Future[None] | None = None
        self._changed = asyncio.Event()
        self._receiving: set[asyncio.Future[Any]] = set()
        self._trigger: Any = None
        self._shutdown = False
        self._finalization: str | None = None
        self._transcript: str | None = None
        self._rpc_deadline: float | None = None
        self._torn = False
        self._clients_suspended = False
        self._handed_over = False
        # The record names this owner as its shutdown observer, so the
        # vehicle can raise the trigger on it.
        attach = getattr(registry, "attach_owner", None)
        if callable(attach):
            attach(record, self)
        else:
            with contextlib.suppress(AttributeError, TypeError):
                record.owner = self

    # -- observable state ---------------------------------------------------

    @property
    def state(self) -> OwnerState:
        """``awaiting_configuration`` ... ``terminal``."""
        return self._state

    @property
    def lease_state(self) -> LeaseState | None:
        """The owner's view of the lease; ``None`` before ``open`` returns."""
        return self._lease_state

    @property
    def cause(self) -> Cause | None:
        """The one selected terminal cause, or ``None``."""
        return self._cause

    @property
    def frontend(self) -> Any:
        """The stream's audio frontend, ``None`` until attached."""
        return self._frontend

    @property
    def outstanding_samples(self) -> int:
        """Samples handed to ``feed`` whose ``on_accepted`` has not fired."""
        return self._outstanding

    @property
    def finalization(self) -> str | None:
        """``"client"`` or ``"shutdown"`` once finalization was accepted."""
        return self._finalization

    @property
    def drain_deadline(self) -> float | None:
        """The drain deadline ``D`` once the trigger was observed."""
        return None if self._trigger is None else self._trigger.deadline

    @property
    def shutdown_observed(self) -> bool:
        """Whether the shutdown trigger reached this owner."""
        return self._trigger is not None

    @property
    def handed_over(self) -> bool:
        """Whether a cancelled ``release`` left the session to the factory."""
        return self._handed_over

    @property
    def rpc_deadline(self) -> float | None:
        """The RPC's absolute deadline, when the surface has one."""
        return self._rpc_deadline

    @rpc_deadline.setter
    def rpc_deadline(self, deadline: float | None) -> None:
        self._rpc_deadline = deadline
        self._notify()

    def attach_frontend(self, frontend: Any) -> None:
        """Attach the stream's audio frontend once its cell is known."""
        self._frontend = frontend

    # -- surface operations -------------------------------------------------

    # @spec ING-CORE-011, ING-LIFE-022, ING-LIFE-032, ING-LIFE-036
    async def open(self, cadence: str, locale: str) -> None:
        """Open the lease under the preconfiguration bound.

        On success the owner is ``configured`` and the first-audio and
        duration bounds start. A refusal or failure ends the stream with its
        projected code; a bound that expires cancels ``open``, and a lease
        the owner never received is the factory's to clean up.
        """
        await self._guarded(self._open(cadence, locale))

    # @spec ING-FE-007, ING-FE-015, ING-LIFE-040, ING-LIFE-041
    # @spec ING-LIFE-042, ING-LIFE-021
    async def feed(self, samples: npt.ArrayLike) -> list[str]:
        """Hand one decoded piece to the lease; returns changed hypotheses.

        The piece's credit is taken when ``feed`` is called and released
        once, by that call's ``on_accepted``, which also resets the idle
        bound; client-idle timing is suspended while the call is outstanding
        and the call runs under ``lease_call_timeout``.
        """
        return await self._guarded(self._feed(samples))

    # @spec ING-LIFE-007, ING-LIFE-042
    async def update_locale(self, locale: str) -> None:
        """Run a validated locale change through the serialized lease.

        A refusal leaves the stream and the prior locale unchanged; any other
        failure, or ``lease_call_timeout``, ends the stream.
        """
        await self._guarded(self._update_locale(locale))

    # @spec ING-LIFE-002, ING-LIFE-003, ING-LIFE-017, ING-LIFE-025, ING-FE-009
    # @spec ING-LIFE-027, ING-LIFE-028, ING-LIFE-029, ING-LIFE-044
    # @spec ING-ERR-023
    async def finalize(self) -> str:
        """Run the one finalization sequence and return the final transcript.

        The audio frontend's drained tail is fed, then ``flush`` (always,
        zero audio included; a non-string result is a failed ``flush``),
        then ``finish`` and ``release``. The transcript is returned only
        after both succeeded and no cause was selected meanwhile; otherwise
        the selected cause is raised as ``FrontendError``.
        """
        return await self._guarded(self._finalize())

    # @spec ING-LIFE-026, ING-LIFE-035, ING-LIFE-036, ING-LIFE-038
    # @spec ING-ERR-014
    async def fail(self, cause: Cause) -> None:
        """Select ``cause`` (unless one is already selected) and clean up.

        The current ``open``, ``feed``, ``update_locale``, ``flush`` or
        ``finish`` is cancelled and settles before the shielded cleanup runs;
        a running ``release`` is never cancelled. Returns after cleanup.
        """
        if self._is_torn_down():
            self.teardown()
            return
        self._select(cause)
        await self._join_cleanup()

    # @spec ING-LIFE-057, ING-LIFE-022, ING-LIFE-040, ING-LIFE-043
    # @spec ING-ERR-056, ING-LIFE-034, ING-LIFE-052
    async def receive(self, awaitable: Awaitable[T]) -> T:
        """Wait for one client message under the owner's current bound.

        Before ``open`` the bound is the preconfiguration bound; after it,
        the first-audio or idle bound and the duration bound; during a
        client-selected finalization (the receive observer), the
        finalization bound; and always the RPC deadline. Expiry selects the
        earliest deadline's cause, cleans up and raises ``FrontendError``.
        After the shutdown trigger nothing is read: the pending read is
        cancelled and ``ShutdownObserved`` is raised (an owner with no lease
        ends with the shutdown code instead). Transport exceptions propagate.
        """
        return await self._guarded(self._receive(awaitable))

    # @spec ING-LIFE-053, ING-LIFE-054, ING-LIFE-055, ING-LIFE-031
    # @spec ING-LIFE-048, ING-ERR-059
    def observe_shutdown(self, trigger: Any) -> None:
        """Observe the drain's shutdown trigger (deadline ``D``).

        An owner with neither a client-selected finalization nor an abnormal
        cause selects provisional shutdown mode: any later failure or
        deadline takes the shutdown abort row. A pending client read is
        cancelled (never a lease call); an outstanding call settles under its
        own deadline with the wait clipped to ``D``.
        """
        if self._trigger is not None or self._state is OwnerState.terminal:
            return
        self._trigger = trigger
        if self._cause is None and self._finalization is None:
            self._shutdown = True
        for task in list(self._receiving):
            task.cancel()
        self._notify()

    # @spec ING-EP-043
    def teardown(self) -> None:
        """The stopped plugin's path: the lease's synchronous ``teardown``.

        No ``abort`` or ``release`` follows; in-flight calls and cleanup are
        cancelled and the owner ends.
        """
        self._torn = True
        lease = self._lease
        if lease is not None and self._lease_state not in (
            LeaseState.released,
            LeaseState.torn_down,
        ):
            self.calls.append("teardown")
            self._lease_state = LeaseState.torn_down
            with contextlib.suppress(Exception):
                lease.teardown()
        current = asyncio.current_task()
        for task in (self._call, self._cleanup):
            if task is not None and task is not current and not task.done():
                task.cancel()
        self._end()

    # -- operation bodies ---------------------------------------------------

    async def _open(self, cadence: str, locale: str) -> None:
        await self._check_live()
        if self._lease is not None or self._call is not None:
            raise RuntimeError("a stream owner opens exactly one lease")
        if self._trigger is not None:
            await self._end_with(Cause("service_unavailable", "shutdown"))
        task = await self._invoke(
            "open", "open", lambda: self._opener(cadence=cadence, locale=locale)
        )
        if self._cause is None and task.cancelled():
            self._select(Cause("service_unavailable", "shutdown"))
        elif self._cause is None and task.exception() is not None:
            self._select(self._failure(task.exception(), "factory"))
        if self._cause is not None:
            await self._join_cleanup()
            raise self._error()
        now = self._clock.monotonic()
        self._state = OwnerState.configured
        self.bounds.preconfiguration.disarm()
        self.bounds.first_audio.arm(now)
        self.bounds.duration.arm(now)
        self._notify()

    async def _feed(self, samples: npt.ArrayLike) -> list[str]:
        await self._check_streaming()
        pieces = np.asarray(samples, dtype=np.float32)
        if pieces.ndim != 1:
            pieces = pieces.reshape(-1)
        limit = int(self._config.pre_submit_max_samples)
        transcripts: list[str] = []
        for start in range(0, pieces.size, limit):
            transcripts.extend(
                await self._feed_piece(pieces[start : start + limit], "call")
            )
        return transcripts

    async def _update_locale(self, locale: str) -> None:
        await self._check_streaming()
        task = await self._lease_call("update_locale", "call", locale)
        if self._cause is None and not task.cancelled():
            error = task.exception()
            if error is not None and seam_error(error)[0] == "refused":
                raise project_seam_error(error)
        await self._raise_on_failure(task)

    async def _finalize(self) -> str:
        if self._transcript is not None and self._cause is None:
            return self._transcript
        await self._check_live()
        if self._lease is None:
            code = "service_unavailable" if self._trigger else "protocol_order"
            await self._end_with(
                Cause(code, "shutdown" if self._trigger else "protocol")
            )
        if self._finalization is None:
            self._begin_finalization()

        if self._frontend is not None:
            tail = np.asarray(self._frontend.drain(), dtype=np.float32)
            if tail.size:
                await self._feed_piece(tail, "final")

        task = await self._lease_call("flush", "final")
        await self._raise_on_failure(task)
        text = task.result()
        if not isinstance(text, str):
            await self._end_with(
                Cause(
                    "internal",
                    "engine",
                    error=FrontendError(
                        "internal",
                        detail="the session's flush returned no transcript",
                    ),
                )
            )

        task = await self._lease_call("finish", "final")
        await self._raise_on_failure(task)

        # Terminal cleanup goes on to ``release`` only; the final is
        # exposed only after it succeeded.
        await self._join_cleanup()
        if self._cause is not None:
            raise self._error()
        if self._lease_state is not LeaseState.released:
            self._select(Cause("internal", "cleanup"))
            raise self._error()
        self._transcript = text
        return text

    async def _receive(self, awaitable: Awaitable[T]) -> T:
        if self._cause is not None:
            _close(awaitable)
            await self._join_cleanup()
            raise self._error()
        if self._state is OwnerState.terminal:
            _close(awaitable)
            raise RuntimeError("the stream has ended; nothing is read")
        if self._trigger is not None:
            _close(awaitable)
            await self._shutdown_receive()
        task: asyncio.Future[T] = self._child(awaitable)
        self._receiving.add(task)
        try:
            while not task.done():
                due = self._due("input")
                await self._until({task}, None if due is None else due[0])
                if task.done():
                    break
                if self._cause is not None or self._trigger is not None:
                    break
                if due is not None and self._clock.monotonic() >= due[0]:
                    self._select(self._expired(due))
                    break
        except BaseException:
            task.cancel()
            raise
        finally:
            self._receiving.discard(task)
        if not task.done():
            task.cancel()
            await asyncio.wait({task})
        if self._cause is not None:
            await self._join_cleanup()
            raise self._error()
        if self._trigger is not None:
            if not task.cancelled():
                task.exception()  # unread traffic is not classified
            await self._shutdown_receive()
        message = task.result()
        if self._state is OwnerState.configured:
            self._state = OwnerState.streaming
        return message

    async def _shutdown_receive(self) -> Any:
        if self._lease is None:
            await self._end_with(Cause("service_unavailable", "shutdown"))
        raise ShutdownObserved(self._trigger)

    # -- lease calls ----------------------------------------------------------

    async def _feed_piece(
        self, piece: npt.NDArray[np.float32], kind: str
    ) -> list[str]:
        count = int(piece.size)
        if count == 0:
            return []
        self._outstanding += count
        credited = True

        def on_accepted(accepted: int) -> None:
            nonlocal credited
            if credited:
                credited = False
                self._outstanding -= count
                self._audio_accepted()

        try:
            task = await self._lease_call(
                "feed", kind, piece, on_accepted=on_accepted
            )
        finally:
            if credited:
                credited = False
                self._outstanding -= count
        await self._raise_on_failure(task)
        result = task.result()
        return [str(text) for text in result] if result else []

    async def _lease_call(
        self, name: str, kind: str, *args: Any, **kwargs: Any
    ) -> asyncio.Future[Any]:
        """Run one lease call under ``kind``'s bounds; returns its task."""
        method = getattr(self._lease, name)
        client_call = kind == "call"
        if client_call:
            now = self._clock.monotonic()
            self.bounds.lease_call.arm(now)
            self._suspend_client_timers(now)
        try:
            return await self._invoke(
                name, kind, lambda: method(*args, **kwargs)
            )
        finally:
            if client_call:
                self.bounds.lease_call.disarm()
                if self._state not in (
                    OwnerState.finalizing,
                    OwnerState.terminal,
                ):
                    self._resume_client_timers(self._clock.monotonic())

    async def _invoke(
        self, name: str, kind: str, start: Callable[[], Awaitable[Any]]
    ) -> asyncio.Future[Any]:
        """Run ``start()`` as the owner's one in-flight call and settle it."""
        if name != "open":
            self.calls.append(name)
        task = self._spawn(name, start())
        self._call, self._call_name = task, name
        await self._settle(task, kind)
        return task

    def _spawn(self, name: str, call: Awaitable[Any]) -> asyncio.Future[Any]:
        """Start one call as a child task of the record.

        The call's outcome updates the owner's view of the lease inside the
        task, at the instant it completes, so whoever sees the task done
        sees the lease state it left.
        """
        recorded = False

        async def tracked() -> Any:
            nonlocal recorded
            try:
                result = await call
            except BaseException as error:
                recorded = True
                self._outcome(name, None, error)
                raise
            recorded = True
            self._outcome(name, result, None)
            return result

        def settled(task: asyncio.Future[Any]) -> None:
            if self._call is task:
                self._call = self._call_name = None
            if not recorded:
                # Cancelled before it started: the call never ran.
                _close(call)
                self._outcome(name, None, asyncio.CancelledError())

        task = self._child(tracked())
        task.add_done_callback(settled)
        return task

    async def _settle(self, task: asyncio.Future[Any], kind: str) -> None:
        """Wait for the in-flight call under the deadlines of ``kind``.

        The first deadline to expire selects its cause and the cleanup task
        cancels the call; a cause selected elsewhere does the same. The
        shutdown trigger only clips the wait to ``D``.
        """
        while not task.done():
            # Once a cause stands, the cleanup task cancels the call; only
            # its settlement is awaited.
            due = None if self._cause is not None else self._due(kind)
            await self._until({task}, None if due is None else due[0])
            if task.done():
                return
            if (
                self._cause is None
                and due is not None
                and self._clock.monotonic() >= due[0]
            ):
                self._select(self._expired(due))
            if self._cause is not None:
                self._start_cleanup()

    def _outcome(
        self, name: str, result: Any, error: BaseException | None
    ) -> None:
        """Track the lease's state from one call's outcome."""
        if self._call is asyncio.current_task():
            self._call = self._call_name = None
        cancelled = isinstance(error, asyncio.CancelledError)
        if error is not None and seam_error(error)[1] == "engine_dead":
            self._escalate(error)
        if name == "open":
            if error is None:
                self._lease = result
                self._lease_state = LeaseState.open
            return
        state = self._lease_state
        if state is LeaseState.torn_down:
            return
        if name in ("feed", "update_locale"):
            refused = error is not None and seam_error(error)[0] == "refused"
            if error is not None and not refused and state is LeaseState.open:
                self._lease_state = LeaseState.failed
        elif name == "flush":
            if error is None:
                self._lease_state = LeaseState.flushed
            elif state is LeaseState.open:
                self._lease_state = LeaseState.failed
        elif name == "abort":
            self._lease_state = LeaseState.aborted
        elif name == "release":
            if error is None:
                self._lease_state = LeaseState.released
            elif cancelled:
                self._handed_over = True

    async def _raise_on_failure(self, task: asyncio.Future[Any]) -> None:
        """Turn a failed call, or a cause selected meanwhile, into the end."""
        if self._cause is None:
            if task.cancelled():
                # Only the server cancels a call the owner did not.
                self._select(Cause("service_unavailable", "shutdown"))
            elif task.exception() is not None:
                self._select(self._failure(task.exception(), "engine"))
        if self._cause is not None:
            await self._join_cleanup()
            raise self._error()

    # -- cause and cleanup ----------------------------------------------------

    def _select(self, cause: Cause) -> bool:
        """Record ``cause`` unless a cause stands; an abnormal cause is final.

        In provisional shutdown mode every cause but a transport failure
        becomes the shutdown abort row (``service_unavailable``, phase
        ``shutdown``): the session is being shut down. A cause selected after
        the stream ended normally is recorded too; it suppresses a final whose
        send has not begun.
        """
        if self._cause is not None:
            return False
        if self._shutdown and cause.phase not in (
            "transport",
            "shutdown",
            "drain",
        ):
            cause = Cause("service_unavailable", "shutdown")
        self._cause = cause
        self._notify()
        return True

    async def _end_with(self, cause: Cause) -> None:
        """Select ``cause``, clean up and raise the selected cause."""
        self._select(cause)
        await self._join_cleanup()
        raise self._error()

    def _start_cleanup(self) -> asyncio.Future[None]:
        if self._cleanup is None:
            self._cleanup = self._child(self._terminate())
        return self._cleanup

    async def _join_cleanup(self) -> None:
        """Start the cleanup if needed and wait for it, shielded."""
        cleanup = self._start_cleanup()
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError as error:
                if cleanup.done():
                    break
                trigger = _trigger_of(error)
                if trigger is None:
                    raise
                self.observe_shutdown(trigger)
                _uncancel()

    # @spec ING-LIFE-037, ING-LIFE-020, ING-LIFE-027, ING-ERR-005
    # @spec ING-ERR-035
    async def _terminate(self) -> None:
        """The one terminal cleanup, as the lease's state allows.

        The current call is cancelled and settles first. An open or failed
        lease takes ``abort`` then ``release``; a flushed or aborted one
        ``release`` only; a torn-down plugin only the synchronous
        ``teardown``. ``abort`` and ``release`` share one cleanup budget
        (clipped to ``D`` while draining). A call the budget cannot wait for
        is still started and runs to its first suspension before it is
        cancelled, so a cancelled ``release`` hands the session to the
        factory's cleanup.
        """
        try:
            await self._quiesce()
            if self._is_torn_down():
                self.teardown()
                return
            if self._lease is None or self._lease_state in (
                LeaseState.released,
                LeaseState.torn_down,
            ):
                return
            self.bounds.cleanup.arm(self._clock.monotonic())
            if self._lease_state in (LeaseState.open, LeaseState.failed):
                await self._cleanup_call("abort")
            await self._cleanup_call("release")
            if self._lease_state is not LeaseState.released:
                self._select(Cause("internal", "cleanup"))
        except asyncio.CancelledError:
            if self._is_torn_down():
                self.teardown()
            raise
        finally:
            self._end()

    async def _quiesce(self) -> None:
        """Cancel the in-flight call and wait until it settled."""
        while True:
            task = self._call
            if task is None or task.done():
                return
            task.cancel()
            await self._shielded(asyncio.wait({task}))

    async def _shielded(self, awaitable: Awaitable[Any]) -> None:
        """Await one cleanup step, shielded from cancellation.

        Only a stopped plugin's cancellation interrupts cleanup; any other
        is absorbed, and the step's own awaitable is closed or cancelled.
        """
        step = self._child(awaitable)
        while not step.done():
            try:
                await asyncio.wait({step})
            except asyncio.CancelledError:
                if self._is_torn_down():
                    step.cancel()
                    raise
                _uncancel()

    async def _cleanup_call(self, name: str) -> None:
        self.calls.append(name)
        task = self._spawn(name, getattr(self._lease, name)())
        self._call, self._call_name = task, name
        # Let the call start and run to its first suspension, whatever the
        # budget left: a spent budget must still hand the session over.
        await self._shielded(asyncio.sleep(0))
        while not task.done():
            deadline = self._cleanup_deadline()
            if deadline is not None and self._clock.monotonic() >= deadline:
                break
            await self._shielded(self._until({task}, deadline))
        if not task.done():
            task.cancel()
            await self._shielded(asyncio.wait({task}))
        error = None if task.cancelled() else task.exception()
        if error is not None:
            self.cleanup_errors.append(error)
            if name == "release":
                projected = project_seam_error(error, call="release")
                self._select(Cause(projected.code, "engine", error=projected))

    def _cleanup_deadline(self) -> float | None:
        deadline = self.bounds.cleanup.deadline
        if self._trigger is not None:
            limit = self._trigger.deadline
            deadline = limit if deadline is None else min(deadline, limit)
        return deadline

    def _end(self) -> None:
        if self._state is OwnerState.terminal:
            return
        self._state = OwnerState.terminal
        for timer in self.bounds.timers():
            timer.disarm()
        for task in list(self._receiving):
            task.cancel()
        with contextlib.suppress(KeyError, ValueError):
            self._registry.unregister(self._record)
        self._notify()

    def _is_torn_down(self) -> bool:
        return (
            self._torn
            or bool(getattr(self._registry, "torn_down", False))
            or bool(getattr(self._record, "torn_down", False))
        )

    def _escalate(self, error: BaseException) -> None:
        self.escalations.append(error)
        hooks = (
            self.on_escalation,
            getattr(self._registry, "escalate", None),
        )
        for hook in hooks:
            if callable(hook):
                with contextlib.suppress(Exception):
                    hook(error)
                return

    def _failure(self, error: BaseException | None, phase: str) -> Cause:
        """The cause a failed seam call selects (errors leaf mapping)."""
        if error is None:
            return Cause("internal", phase)
        projected = project_seam_error(error)
        return Cause(projected.code, phase, error=projected)

    def _expired(self, due: tuple[float, str]) -> Cause:
        deadline, phase = due
        return Cause(_EXPIRY_CODES[phase], phase, deadline)

    def _error(self) -> FrontendError:
        cause = self._cause
        if cause is None:
            return FrontendError("internal", detail="the stream ended")
        if cause.error is not None and cause.error.code == cause.code:
            return cause.error
        detail = _EXPIRY_DETAILS.get(
            cause.phase, f"the stream ended ({cause.phase})"
        )
        return FrontendError(cause.code, detail=detail)

    # -- timing ---------------------------------------------------------------

    def _due(self, kind: str) -> tuple[float, str] | None:
        """The earliest deadline of a wait of ``kind``, with its phase.

        Ties go to the phase timer over the RPC deadline and ``D``.
        """
        b = self.bounds
        timers: tuple[Timer, ...]
        if kind == "open":
            timers = (b.preconfiguration,)
        elif kind == "call":
            timers = (b.lease_call, b.duration)
        elif kind == "final":
            timers = (b.finalization,)
        elif self._state is OwnerState.awaiting_configuration:
            timers = (b.preconfiguration,)
        elif self._state is OwnerState.finalizing:
            timers = (b.finalization,)
        else:
            timers = (b.first_audio, b.idle, b.duration)
        candidates = [
            (timer.deadline, 0, timer.name)
            for timer in timers
            if timer.deadline is not None
        ]
        if self._rpc_deadline is not None:
            candidates.append((self._rpc_deadline, 1, "rpc_deadline"))
        if self._trigger is not None:
            candidates.append((self._trigger.deadline, 2, "drain"))
        if not candidates:
            return None
        deadline, _, phase = min(candidates)
        return deadline, phase

    async def _until(
        self, futures: set[asyncio.Future[Any]], deadline: float | None
    ) -> None:
        """Wait for one of ``futures``, a state change, or ``deadline``.

        The deadline is a ``clock.sleep`` in a child task, so the wait
        creates no task outside the owner's record. A cancellation carrying
        the shutdown trigger is observed, not propagated.
        """
        helpers = [self._child(self._changed.wait())]
        if deadline is not None:
            remaining = max(0.0, deadline - self._clock.monotonic())
            helpers.append(self._child(self._clock.sleep(remaining)))
        try:
            await asyncio.wait(
                {*futures, *helpers}, return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError as error:
            trigger = _trigger_of(error)
            if trigger is None:
                raise
            self.observe_shutdown(trigger)
            _uncancel()
        finally:
            for helper in helpers:
                helper.cancel()

    def _child(self, awaitable: Awaitable[T]) -> asyncio.Future[T]:
        """A task kept under the owner's record until it ends."""
        task = asyncio.ensure_future(awaitable)
        children = getattr(self._record, "children", None)
        if isinstance(children, set):
            children.add(task)
            task.add_done_callback(children.discard)
        return task

    def _notify(self) -> None:
        self._changed.set()
        self._changed = asyncio.Event()

    def _begin_finalization(self) -> None:
        now = self._clock.monotonic()
        self._finalization = "shutdown" if self._shutdown else "client"
        self._state = OwnerState.finalizing
        b = self.bounds
        for timer in (b.first_audio, b.idle, b.duration, b.lease_call):
            timer.disarm()
        b.finalization.arm(now)
        self._notify()

    def _suspend_client_timers(self, now: float) -> None:
        self.bounds.first_audio.suspend(now)
        self.bounds.idle.suspend(now)
        self._clients_suspended = True

    def _resume_client_timers(self, now: float) -> None:
        self.bounds.first_audio.resume(now)
        self.bounds.idle.resume(now)
        self._clients_suspended = False

    def _audio_accepted(self) -> None:
        """Accepted audio replaces first-audio timing and resets idle."""
        b = self.bounds
        b.first_audio.disarm()
        if self._state in (OwnerState.finalizing, OwnerState.terminal):
            return
        if b.idle.seconds is None:
            return
        if self._clients_suspended:
            b.idle.deadline = None
            b.idle.suspended = b.idle.seconds
        else:
            b.idle.arm(self._clock.monotonic())

    # -- guards ---------------------------------------------------------------

    async def _check_live(self) -> None:
        if self._cause is not None:
            await self._join_cleanup()
            raise self._error()
        if self._state is OwnerState.terminal:
            raise RuntimeError("the stream has ended")

    async def _check_streaming(self) -> None:
        await self._check_live()
        if self._lease is None:
            raise RuntimeError("the stream has no open lease")
        if self._state is OwnerState.finalizing:
            await self._end_with(Cause("session_terminal", "protocol"))
        if self._state is OwnerState.configured:
            self._state = OwnerState.streaming

    async def _guarded(self, operation: Awaitable[T]) -> T:
        """Run one owner operation; the owner's own cancellation ends it.

        A cancellation the owner did not ask for is the server's: the stream
        ends with ``service_unavailable`` and the shielded cleanup, or, on a
        stopped plugin, with the lease's synchronous ``teardown``.
        """
        try:
            return await operation
        except asyncio.CancelledError:
            if self._is_torn_down():
                self.teardown()
                raise
            self._select(Cause("service_unavailable", "shutdown"))
            try:
                await self._join_cleanup()
            except asyncio.CancelledError:
                if self._is_torn_down():
                    self.teardown()
                raise
            # The cancellation became the stream's cause; it is consumed.
            _uncancel()
            raise self._error() from None
