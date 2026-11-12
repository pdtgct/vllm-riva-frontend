"""The endpoint plugin vLLM core sees: startup, owners, shutdown and faults.

``RivaFrontendPlugin`` implements the lifecycle core defines for endpoint
plugins (``attach_router``, ``init_state``, ``start``, ``ready``, ``drain``,
``stop``) and adds none. It owns the frontend's route set under
``/plugins/riva/v1/`` and its path-scoped authentication, the order of the
startup checks, the registry of transport owner tasks, the ingress cap, the
relation between its bounds and the engine's stream deadlines, the drain,
the settlement and ``stop``, and fault escalation.

The surface modules (``grpc``, ``nim_ws``, ``files``) and the session seam
(``lease``) are imported inside the lifecycle methods: they read this
module's registry types, so a module-level import would be circular. vLLM
itself is imported only inside the lifecycle methods (ING-EP-108).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import importlib.metadata
import inspect
import logging
import math
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from grpc_health.v1 import health_pb2

from vllm_riva_frontend import routes
from vllm_riva_frontend.admission import FILE_KINDS, IngressCap, OwnerKind
from vllm_riva_frontend.auth import (
    AuthState,
    PathScopedAuthMiddleware,
    configured_keys,
    decide_tokens,
)
from vllm_riva_frontend.clock import Clock, RealClock
from vllm_riva_frontend.config import (
    PLUGIN_KEY,
    Capacity,
    ConfigError,
    DeploymentProvenance,
    FrontendConfig,
    StartupError,
    StreamDeadlines,
    build_deployment_provenance,
    check_bounds,
    derive_file_caps,
    parse_config,
    read_capacity,
    read_stream_deadlines,
)
from vllm_riva_frontend.errors import ErrorCatalog, catalog
from vllm_riva_frontend.operational import operational_endpoints

__all__ = [
    "FaultCollector",
    "ListenerMonitor",
    "OwnerRecord",
    "OwnerRegistry",
    "RivaFrontendPlugin",
    "ShutdownTrigger",
    "StartupError",
    "engine_reserve",
    "settle_point",
]

logger = logging.getLogger(__name__)

#: Framing allowance of the multipart body above core's upload limit.
UPLOAD_FRAMING_BYTES = 262_144
#: The fault collector's period; a protocol constant, not an option.
COLLECTOR_PERIOD = 1.0
#: How long one listener-termination wait lasts before it is renewed.
_MONITOR_POLL = 3600.0

#: The owner kinds the shutdown trigger finalizes; files are cancelled.
STREAM_KINDS = frozenset({OwnerKind.stream, OwnerKind.realtime})

_SERVING = health_pb2.HealthCheckResponse.SERVING
_NOT_SERVING = health_pb2.HealthCheckResponse.NOT_SERVING


# ----------------------------------------------------------- the formulas


# @spec ING-EP-066
def engine_reserve(timeout: float) -> float:
    """Return the time left to core to shut the engine down after ``drain``.

    The larger of 5 s and a tenth of the shutdown budget.
    """
    return max(5.0, timeout / 10)


# @spec ING-EP-036
def settle_point(now: float, deadline: float) -> float:
    """Return when owners stop finishing on their own: shortly before ``D``.

    ``D`` less the smaller of 5 s and a quarter of the time left to it.
    """
    return deadline - min(5.0, max(0.0, deadline - now) / 4)


@dataclass(frozen=True)
class ShutdownTrigger:
    """What ``drain`` raises for each stream owner at the settle point.

    ``deadline`` is ``D``, an absolute ``clock.monotonic()`` value: the
    owner finalizes its stream by it, or aborts.
    """

    deadline: float


def _log_late_factory_drain(error: Exception) -> None:
    """Log a late ``factory.drain(D)`` failure that is not propagated."""
    logger.error(
        "%s: factory.drain(D) failed",
        PLUGIN_KEY,
        exc_info=(type(error), error, error.__traceback__),
    )


async def _run_to_end(awaitable: Awaitable[Any]) -> bool:
    """Await ``awaitable`` to its end through any number of cancellations.

    The work runs as its own task, so a cancellation of the caller never
    reaches it. Returns whether the caller was cancelled meanwhile (the
    caller re-raises that cancellation afterwards); the work's own
    exception propagates.
    """
    work = asyncio.ensure_future(awaitable)
    interrupted = False
    while not work.done():
        try:
            await asyncio.wait({work})
        except asyncio.CancelledError:
            interrupted = True
    work.result()
    return interrupted


# -------------------------------------------------------- owner registry


@dataclass(eq=False)
class OwnerRecord:
    """One registered transport owner: its task and what it holds.

    ``task`` is the handler. ``continuation`` is a cleanup that outlived the
    answer (a file request's, or a stream owner's waiting for a cancelled
    call); ``children`` are tasks the owner adds directly, the stream's
    sends included. ``owner`` is the object that observes the shutdown
    trigger (a stream's ``StreamOwner``), when one is attached.

    A streaming surface sets ``held_by_handler``: the record then stays
    registered until its handler, its continuation and every send in
    ``unsettled_sends`` (sends cancelled at their bound that have not
    settled, also in ``children``) have ended, and the handler's end
    (``surfaces.release_owner``) removes it, not the end of the owner's
    cleanup. The settlement awaits all of them (ING-EP-093).

    ``cancelled_by_plugin`` is set, just before the cancellation, whenever
    the plugin itself cancels ``task`` (the settle point of ``drain``, the
    vehicle's settlement, ``stop``), so the owner tells the plugin's
    cancellation from its transport's by that mark alone (ING-FILE-089).
    """

    kind: OwnerKind
    task: asyncio.Task[Any]
    continuation: asyncio.Task[Any] | None = None
    children: set[asyncio.Task[Any]] = field(default_factory=set)
    owner: Any = None
    held_by_handler: bool = False
    unsettled_sends: set[asyncio.Task[Any]] = field(default_factory=set)
    cancelled_by_plugin: bool = False

    def cancel_by_plugin(self) -> None:
        """Cancel ``task`` as the plugin, marking the cancellation as such."""
        self.cancelled_by_plugin = True
        self.task.cancel()


def _shutdown_observer(record: OwnerRecord) -> Any:
    """The object on ``record`` that observes the shutdown trigger."""
    if callable(getattr(record.owner, "observe_shutdown", None)):
        return record.owner
    for value in vars(record).values():
        if callable(getattr(value, "observe_shutdown", None)):
            return value
    return None


class OwnerRegistry:
    """The registry of transport owner tasks; one record per owner.

    Besides the records, it holds the sends without an owner (a refusal's
    send its handler abandoned, ING-EP-095) until each ends, and carries
    what an owner needs from the plugin without holding the plugin:
    ``torn_down`` (``stop`` began), ``drain_deadline`` (``D``, only while a
    ``drain`` call runs its graceful work), ``send_bounds_passed`` (every
    send bound counts as passed, ING-EP-110), ``shutdown_trigger`` once
    raised, and ``escalate`` for a failure that escalates.
    """

    # @spec ING-EP-106
    def __init__(self) -> None:
        """Create an empty registry, open to sends without an owner."""
        self._records: dict[int, OwnerRecord] = {}
        self._waiters: set[asyncio.Future[None]] = set()
        self._unowned: set[asyncio.Future[Any]] = set()
        self._sends_closed = False
        self.torn_down = False
        self.drain_deadline: float | None = None
        self.send_bounds_passed = False
        self.shutdown_trigger: ShutdownTrigger | None = None
        self.on_fault: Callable[[BaseException], None] | None = None

    @property
    def records(self) -> Mapping[int, OwnerRecord]:
        """The live records, read-only."""
        return MappingProxyType(self._records)

    # @spec ING-EP-106, ING-ADM-013
    def register(
        self, kind: OwnerKind | str, task: asyncio.Task[Any]
    ) -> OwnerRecord:
        """Register ``task`` as an owner of ``kind`` and return its record."""
        record = OwnerRecord(OwnerKind(kind), task)
        self._records[id(record)] = record
        self._notify()
        return record

    def unregister(self, record: OwnerRecord) -> None:
        """Remove ``record``; removing it again does nothing."""
        if self._records.get(id(record)) is record:
            del self._records[id(record)]
            self._notify()

    def attach_continuation(
        self, record: OwnerRecord, task: asyncio.Task[Any]
    ) -> None:
        """Keep ``record`` under ``task``, its cleanup continuation."""
        record.continuation = task
        self._notify()

    def attach_owner(self, record: OwnerRecord, owner: Any) -> None:
        """Attach the object that observes the shutdown trigger.

        If the trigger has already been raised, it is delivered at once.
        """
        record.owner = owner
        if self.shutdown_trigger is not None:
            self._deliver(record, self.shutdown_trigger)

    def escalate(self, error: BaseException) -> None:
        """Report a failure that escalates (an ``engine_dead`` reason)."""
        if self.on_fault is not None:
            self.on_fault(error)

    # ------------------------------------------- sends without an owner

    @property
    def unowned_sends(self) -> frozenset[asyncio.Future[Any]]:
        """The registered sends without an owner that have not ended."""
        return frozenset(self._unowned)

    @property
    def sends_closed(self) -> bool:
        """Whether the settlement closed the registry to such sends."""
        return self._sends_closed

    # @spec ING-EP-094, ING-EP-095
    def adopt_send(self, task: asyncio.Future[Any]) -> bool:
        """Register an abandoned refusal send; ``False`` once closed.

        Synchronous: the open check and the registration have no
        suspension between them, so every abandoned send is either here
        before ``close_sends`` or kept by its own handler after it. The
        send takes no ingress-cap count and leaves when it ends.
        """
        if self._sends_closed:
            return False
        if not task.done():
            self._unowned.add(task)
            task.add_done_callback(self._send_ended)
            self._notify()
        return True

    # @spec ING-EP-094, ING-EP-093
    def close_sends(self) -> None:
        """Close the registry to sends without an owner (synchronous)."""
        self._sends_closed = True

    def _send_ended(self, task: asyncio.Future[Any]) -> None:
        self._unowned.discard(task)
        self._notify()

    # ------------------------------------------------------ settlement

    def cancel_owners(self) -> list[asyncio.Task[Any]]:
        """Cancel each owner task once; its cleanup continues.

        A task already being cancelled (a file request cancelled at the
        settle point, say) is not cancelled again, so its cleanup is not
        cut short.
        """
        current = asyncio.current_task()
        records = list(self._records.values())
        for record in records:
            task = record.task
            if (
                task is not current
                and not task.done()
                and not task.cancelling()
            ):
                record.cancel_by_plugin()
        return [record.task for record in records]

    def discard_ended(self) -> None:
        """Remove every record whose tasks have all ended (settlement end)."""
        for record in list(self._records.values()):
            held = (
                record.task,
                record.continuation,
                *record.children,
                *record.unsettled_sends,
            )
            if all(task is None or task.done() for task in held):
                self.unregister(record)

    # @spec ING-EP-037, ING-EP-073
    def deliver_shutdown(self, trigger: ShutdownTrigger) -> None:
        """Raise the shutdown trigger for every owner.

        A stream owner observes it and runs its own shutdown finalization; a
        stream task with no observer yet and every file request are
        cancelled and take their cancellation path.
        """
        self.shutdown_trigger = trigger
        for record in list(self._records.values()):
            if record.task.done():
                continue
            if record.kind in FILE_KINDS:
                record.cancel_by_plugin()
            else:
                self._deliver(record, trigger)

    def _deliver(self, record: OwnerRecord, trigger: ShutdownTrigger) -> None:
        if record.task.done():
            return
        observer = _shutdown_observer(record)
        if observer is None:
            record.cancel_by_plugin()
            return
        try:
            outcome = observer.observe_shutdown(trigger)
        except Exception:
            logger.exception("%s: an owner refused the trigger", PLUGIN_KEY)
            record.cancel_by_plugin()
            return
        if inspect.isawaitable(outcome):
            # Kept under the owner's record, so it is no detached task.
            record.children.add(asyncio.ensure_future(outcome))

    def pending(self) -> set[asyncio.Task[Any]]:
        """The owner tasks and continuations that have not finished."""
        tasks: set[asyncio.Task[Any]] = set()
        for record in self._records.values():
            tasks.add(record.task)
            if record.continuation is not None:
                tasks.add(record.continuation)
        return {task for task in tasks if not task.done()}

    def outstanding(self) -> set[asyncio.Future[Any]]:
        """Everything the settlement awaits that has not ended.

        Every owner task, cleanup continuation, child (a stream's sends
        included), unsettled send and send without an owner.
        """
        tasks: set[asyncio.Future[Any]] = set(self._unowned)
        for record in self._records.values():
            tasks.add(record.task)
            if record.continuation is not None:
                tasks.add(record.continuation)
            tasks.update(record.children)
            tasks.update(record.unsettled_sends)
        return {task for task in tasks if not task.done()}

    def settled(self, kinds: Iterable[OwnerKind] | None = None) -> bool:
        """Whether every owner (of ``kinds``) and continuation has finished."""
        wanted = None if kinds is None else frozenset(kinds)
        for record in self._records.values():
            if wanted is not None and record.kind not in wanted:
                continue
            if not record.task.done():
                return False
            if (
                record.continuation is not None
                and not record.continuation.done()
            ):
                return False
        return True

    def changed(self) -> asyncio.Future[None]:
        """Return a future resolved at the registry's next change."""
        future: asyncio.Future[None] = (
            asyncio.get_running_loop().create_future()
        )
        self._waiters.add(future)
        future.add_done_callback(self._waiters.discard)
        return future

    def _notify(self) -> None:
        for future in list(self._waiters):
            if not future.done():
                future.set_result(None)


# ----------------------------------------------------- supervision tasks


class FaultCollector:
    """Collects the factory's retained faults while the plugin serves.

    Every ``period`` it calls ``factory.drain`` with an immediate deadline,
    under the lock every other ``factory.drain`` call of the plugin takes,
    so two callers never take the same faults. ``request_stop`` ends it by
    a signal it awaits, never by cancelling it (ING-EP-100).
    """

    def __init__(
        self,
        factory: Any,
        period: float = COLLECTOR_PERIOD,
        on_fault: Callable[[BaseException], None] | None = None,
        *,
        clock: Clock | None = None,
        lock: asyncio.Lock | None = None,
    ) -> None:
        """Bind the factory, the period, the fault handler and the clock."""
        self._factory = factory
        self._period = period
        self._on_fault = on_fault
        self._clock = clock if clock is not None else RealClock()
        self._lock = lock if lock is not None else asyncio.Lock()
        self._stopping = False
        self._stop: asyncio.Future[None] | None = None

    # @spec ING-EP-067, ING-EP-100
    async def run(self) -> None:
        """Collect every period until asked to stop.

        Between two calls it waits on the period's timer (a timer of the
        clock, no task) and on the stop signal; a call in progress is never
        interrupted by the signal.
        """
        stop = self._stop_signal()
        while not self._stopping:
            timer = self._clock.timer(self._clock.monotonic() + self._period)
            try:
                await asyncio.wait(
                    {timer, stop}, return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                timer.cancel()
            if self._stopping:
                return
            await self._collect()

    def _stop_signal(self) -> asyncio.Future[None]:
        if self._stop is None:
            self._stop = asyncio.get_running_loop().create_future()
            if self._stopping:
                self._stop.set_result(None)
        return self._stop

    async def _collect(self) -> None:
        try:
            async with self._lock:
                faults = await self._factory.drain(self._clock.monotonic())
        except Exception as error:  # a failing drain is a fault itself
            faults = [error]
        for fault in faults or ():
            if self._on_fault is not None:
                self._on_fault(fault)

    # @spec ING-EP-100
    def request_stop(self) -> None:
        """Signal the collector to end; its task is never cancelled.

        A collection call in progress completes and its faults are
        recorded; the wait between two calls ends on the signal, and the
        collector then ends normally.
        """
        self._stopping = True
        if self._stop is not None and not self._stop.done():
            self._stop.set_result(None)


class ListenerMonitor:
    """Watches the gRPC server and reports its exit."""

    def __init__(
        self,
        server: Any,
        on_exit: Callable[[BaseException | None], None],
        expected: Callable[[], bool] | None = None,
    ) -> None:
        """Bind the server, the exit handler and the expected-exit flag.

        After an exit the plugin caused (``expected()`` true) the monitor
        stays until the plugin cancels it, last in ``stop`` (ING-EP-079).
        """
        self._server = server
        self._on_exit = on_exit
        self._expected = expected

    # @spec ING-EP-016
    async def run(self) -> None:
        """Wait for the server's termination, then report it."""
        try:
            # With a timeout the server shields its termination future;
            # without one, cancelling this monitor would cancel that future
            # and break the server's own ``stop``.
            while await self._server.wait_for_termination(
                timeout=_MONITOR_POLL
            ):
                pass
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._on_exit(error)
            return
        self._on_exit(None)
        if self._expected is not None and self._expected():
            await asyncio.get_running_loop().create_future()


# ------------------------------------------------------------ the plugin


# @spec ING-VEH-026
def _served_identity(state: Any) -> str:
    """Call the serving-model registry once for the served identity."""
    registry = getattr(state, "openai_serving_models", None)
    if registry is None:
        raise StartupError(
            [f"{PLUGIN_KEY}: state.openai_serving_models is missing"]
        )
    accessor = getattr(registry, "model_name", None)
    if not callable(accessor):
        raise StartupError(
            [f"{PLUGIN_KEY}: openai_serving_models.model_name is not callable"]
        )
    try:
        name = accessor()
    except Exception as error:
        raise StartupError(
            [f"{PLUGIN_KEY}: the served-model lookup failed: {error!r}"]
        ) from error
    if not isinstance(name, str) or not name:
        raise StartupError(
            [f"{PLUGIN_KEY}: the served-model registry returned {name!r}"]
        )
    return name


# @spec ING-EP-028, ING-EP-029, ING-EP-030
def _engine_capacity(vllm_config: Any) -> Capacity:
    """Refuse data parallelism, then read slots and the active limit."""
    failures: list[str] = []
    parallel = getattr(
        getattr(vllm_config, "parallel_config", None), "data_parallel_size", 1
    )
    if parallel not in (None, 1):
        failures.append(
            "vllm_config.parallel_config.data_parallel_size is "
            f"{parallel!r}: {PLUGIN_KEY} needs data parallel size 1, as the "
            "slot count and active limit are per engine"
        )
    capacity: Capacity | None = None
    try:
        capacity = read_capacity(vllm_config)
    except StartupError as error:
        failures.extend(error.failures)
    if failures:
        raise StartupError(failures)
    assert capacity is not None
    return capacity


def _core_decode_limit(service: Any, environment_limit: float) -> float:
    """Core's decode-duration limit, as the transcription service holds it."""
    limit = getattr(service, "max_audio_decode_duration_s", None)
    if isinstance(limit, (int, float)) and not isinstance(limit, bool):
        return float(limit)
    return float(environment_limit)


# @spec ING-EP-081
def _api_server_failures(args: Any) -> list[str]:
    """Fail unless ``args.api_server_count`` counts one API server process.

    ``None`` or an absent attribute (core's single-process mode) and 1
    count as one; 0 (headless) and anything above 1 fail, since every API
    server process would bind the gRPC port.
    """
    count = getattr(args, "api_server_count", None)
    if count is None or (type(count) is int and count == 1):
        return []
    return [
        f"{PLUGIN_KEY} needs exactly one API server process, but "
        f"args.api_server_count is {count!r} (--api-server-count): every "
        "API server process would bind the gRPC listener's port, and 0 "
        "runs no API server"
    ]


def _report_settlement(
    faults: Iterable[BaseException],
    last_resort: Iterable[asyncio.Future[Any]],
    where: str,
) -> None:
    """Log the factory settlement's faults, as returned, and last resorts."""
    for fault in faults:
        logger.error(
            "%s: session cleanup fault at %s: %r", PLUGIN_KEY, where, fault
        )
    for task in last_resort:
        logger.error(
            "%s: last-resort fault at %s: factory task %r was still running "
            "when teardown_all returned",
            PLUGIN_KEY,
            where,
            task,
        )


def _unreachable_app(*_: Any) -> Any:
    """The inner app of the verifier, which only ``verify_token`` uses."""
    raise RuntimeError("the frontend's token verifier is not an ASGI app")


# @spec ING-EP-020, ING-VEH-005, ING-VEH-008
class RivaFrontendPlugin:
    """The Riva frontend as one vLLM endpoint plugin.

    Public attributes after ``init_state``: ``config``, ``served_model``,
    ``deadlines``, ``capacity``, ``files_enabled``, ``factory``,
    ``registry``, ``ingress``, ``file_gate``, ``catalog``, ``servicer``,
    ``provenance`` (settable: every surface reads this one object),
    ``transcription`` (core's service or ``None``), ``state`` and
    ``engine_client``. After ``start``: ``server``, ``health``,
    ``fault_collector`` and ``listener_monitor``. Throughout: ``clock``,
    ``auth``, ``draining``, ``drain_deadline`` (``D``, only while a ``drain``
    call runs its graceful work), ``listener_stopping``,
    ``send_bounds_passed``, ``torn_down`` and ``fault``.
    """

    name = "riva_frontend"
    required_tasks: tuple[str, ...] = ("realtime",)

    # @spec ING-EP-062
    def __init__(
        self,
        *,
        clock: Clock | None = None,
        entry_points: Callable[..., Iterable[Any]] | None = None,
    ) -> None:
        """Bind the clock and the session-factory discovery seam."""
        self.clock: Clock = clock if clock is not None else RealClock()
        self._entry_points: Callable[..., Iterable[Any]] = (
            entry_points
            if entry_points is not None
            else importlib.metadata.entry_points
        )
        self._app: Any = None
        self._routes: tuple[routes.FrontendRoute, ...] = ()
        self.auth = AuthState()
        self.state: Any = None
        self.engine_client: Any = None
        self.config: FrontendConfig | None = None
        self.served_model: str | None = None
        self.deadlines: StreamDeadlines | None = None
        self.capacity: Capacity | None = None
        self.files_enabled = False
        self.transcription: Any = None
        self.provenance: DeploymentProvenance | None = None
        self.factory: Any = None
        self.registry: OwnerRegistry | None = None
        self.ingress: IngressCap | None = None
        self.file_gate: Any = None
        self.catalog: ErrorCatalog | None = None
        self.servicer: Any = None
        self.server: Any = None
        self.health: Any = None
        self.fault_collector: asyncio.Task[None] | None = None
        self.listener_monitor: asyncio.Task[None] | None = None
        self.draining = False
        self.drain_deadline: float | None = None
        self.torn_down = False
        self.fault: BaseException | None = None
        self._collector: FaultCollector | None = None
        self._serving = False
        self._listener_closing = False
        self._listener_stopped = False
        self._shutdown_requested = False
        # Serializes every factory.drain call of the plugin (ING-EP-067).
        self._factory_lock = asyncio.Lock()
        # Futures of the sends waiting for their bounds to change.
        self._drain_waiters: set[asyncio.Future[None]] = set()
        # A ``factory.drain(D)`` call still running when ``D`` passed.
        self._factory_drain: asyncio.Future[None] | None = None

    @property
    def admission_open(self) -> bool:
        """Whether the ingress cap admits owners."""
        return self.ingress is not None and self.ingress.is_open

    @property
    def listener_stopping(self) -> bool:
        """Whether the plugin began stopping the gRPC listener.

        An RPC terminated after this is the plugin's cancellation of its
        owner, never a client disconnect (ING-LIFE-064).
        """
        return self._listener_closing

    @property
    def send_bounds_passed(self) -> bool:
        """Whether every send bound counts as passed (ING-EP-110).

        True once the plugin is torn down and, for the rest of a ``drain``
        call (cleared when it returns), once its settlement began or it
        skipped its graceful steps; ``D`` passing is read from
        ``drain_deadline`` and the clock.
        """
        registry = self.registry
        return self.torn_down or (
            registry is not None and registry.send_bounds_passed
        )

    # @spec ING-LIFE-061
    def drain_announced(self) -> asyncio.Future[None]:
        """Return a future resolved at the next change of the send bounds.

        The changes are ``drain`` setting ``D`` and the send bounds passing
        (the settlement beginning, graceful steps skipped, or ``stop``).
        While ``D`` is set the future stays pending until the bounds pass,
        so a send started during graceful work is woken when the
        settlement begins early; once the bounds have passed no change is
        left and the future is already resolved. Every send in flight
        waits on one, so it re-clips its bound at once (ING-LIFE-061,
        ING-EP-110).
        """
        future: asyncio.Future[None] = (
            asyncio.get_running_loop().create_future()
        )
        if self.send_bounds_passed:
            future.set_result(None)
            return future
        self._drain_waiters.add(future)
        future.add_done_callback(self._drain_waiters.discard)
        return future

    def _announce_drain(self) -> None:
        for future in list(self._drain_waiters):
            if not future.done():
                future.set_result(None)

    # ------------------------------------------------------ attach_router

    # @spec ING-EP-063, ING-EP-091, ING-SHIM-023, ING-SHIM-020, ING-EP-061
    # @spec ING-FILE-075, ING-EP-083, ING-FILE-076
    def attach_router(self, app: Any) -> None:
        """Add the frontend's route set and its path-scoped authentication.

        Every route is new and under ``/plugins/riva/v1/``: the four
        operational routes, the Realtime bootstrap and WebSocket routes and,
        only beside core's ``POST /v1/audio/transcriptions``, the file
        route. Each is checked against core's table first; a collision
        fails startup. The middleware is added with ``app.add_middleware``,
        so core's middleware, added after the plugin hooks, wraps it.
        """
        from vllm_riva_frontend.files import file_route
        from vllm_riva_frontend.nim_ws import RealtimeRoute, bootstrap_endpoint

        frontend = [
            routes.http_route("GET", path, endpoint)
            for path, endpoint in operational_endpoints(self)
        ]
        frontend.append(
            routes.http_route(
                "POST", routes.BOOTSTRAP_PATH, bootstrap_endpoint(self)
            )
        )
        frontend.append(
            routes.websocket_route(routes.REALTIME_PATH, RealtimeRoute(self))
        )
        if routes.core_serves_files(app):
            frontend.append(
                routes.http_route(
                    "POST", routes.FILE_ROUTE_PATH, file_route(self)
                )
            )
        failures = routes.collisions(app, frontend)
        if failures:
            raise StartupError(failures)
        routes.add_routes(app, frontend)
        app.add_middleware(PathScopedAuthMiddleware, auth=self.auth)
        self._routes = tuple(frontend)
        self._app = app

    # ---------------------------------------------------------- init_state

    # @spec ING-EP-068, ING-EP-107, ING-EP-069, ING-EP-025, ING-EP-021
    # @spec ING-EP-027, ING-ADM-014, ING-VEH-026, ING-EP-097
    async def init_state(
        self, engine_client: Any, state: Any, args: Any
    ) -> None:
        """Run the startup checks in order, then build the plugin's parts.

        Order: configuration, served identity, engine stream deadlines,
        capacity, complete-file service, API server process count,
        authentication, the build (including the authentication verifier)
        and the route-table check, and the session factory last. Each step
        raises one ``StartupError`` naming everything that failed within
        it. Every factory a constructor returns is settled when discovery
        discards it or when ``init_state`` fails. No listener is bound and
        no task outlives this call.
        """
        import vllm.envs as envs
        from vllm.entrypoints.serve.middleware.authenticate import (
            AuthenticationMiddleware,
        )

        self.state = state
        if engine_client is None:
            raise StartupError([f"{PLUGIN_KEY} needs an engine client"])
        self.engine_client = engine_client
        vllm_config = engine_client.vllm_config
        additional = getattr(vllm_config, "additional_config", None)
        if not isinstance(additional, Mapping):
            additional = {}

        config = parse_config(additional)
        self.served_model = _served_identity(state)

        deadlines = read_stream_deadlines(additional)
        bounds = check_bounds(config, deadlines)
        if bounds:
            raise StartupError(bounds)
        self.deadlines = deadlines

        self.capacity = _engine_capacity(vllm_config)

        service = getattr(state, "openai_serving_transcription", None)
        config = self._size_file_service(
            config,
            service,
            upload_limit=int(
                envs.VLLM_MAX_AUDIO_CLIP_FILESIZE_MB * 1024 * 1024
            ),
            decode_limit=float(envs.VLLM_MAX_AUDIO_DECODE_DURATION_S),
        )
        self.config = config
        self.transcription = service
        self.provenance = build_deployment_provenance(config)

        servers = _api_server_failures(args)
        if servers:
            raise StartupError(servers)

        tokens = self._decide_authentication(
            args, envs.VLLM_API_KEY, AuthenticationMiddleware
        )

        self._build(config)
        self.auth.resolve(
            None
            if tokens is None
            else AuthenticationMiddleware(_unreachable_app, tokens=tokens)
        )
        shadowed = routes.shadowed(self._app, self._routes)
        if shadowed:
            raise StartupError(shadowed)

        await self._discover_factory(engine_client, config)

        if getattr(vllm_config, "shutdown_timeout", None) == 0:
            logger.warning(
                "%s: --shutdown-timeout is 0, so no drain will run; "
                "shutdown is stop alone, which settles open streams",
                PLUGIN_KEY,
            )

    # @spec ING-FILE-006, ING-FILE-009, ING-FILE-015, ING-FILE-016
    # @spec ING-FILE-018, ING-FILE-046, ING-FILE-054, ING-FILE-081
    def _size_file_service(
        self,
        config: FrontendConfig,
        service: Any,
        *,
        upload_limit: int,
        decode_limit: float,
    ) -> FrontendConfig:
        """Size the file gate and sub-cap and check core's file limits.

        No relation to ``max_model_len`` is checked (ING-FILE-081).
        """
        if service is None:
            self.files_enabled = False
            return dataclasses.replace(
                config, max_concurrent_files=0, max_file_owners=0
            )
        assert self.capacity is not None
        failures: list[str] = []
        concurrent, owners, enabled = 0, 0, False
        try:
            concurrent, owners, enabled = derive_file_caps(
                config, self.capacity, True
            )
        except ConfigError as error:
            failures.extend(error.failures)
        envelope = config.http_multipart_envelope_max_bytes
        if envelope < upload_limit + UPLOAD_FRAMING_BYTES:
            failures.append(
                f"{PLUGIN_KEY}.http_multipart_envelope_max_bytes ({envelope}) "
                f"must be at least core's upload limit ({upload_limit} bytes, "
                f"VLLM_MAX_AUDIO_CLIP_FILESIZE_MB) plus "
                f"{UPLOAD_FRAMING_BYTES} bytes of framing"
            )
        limit = _core_decode_limit(service, decode_limit)
        duration = config.unary_max_decoded_duration_seconds
        if duration > limit:
            failures.append(
                f"{PLUGIN_KEY}.unary_max_decoded_duration_seconds ({duration}) "
                f"must be at most core's decode-duration limit ({limit} "
                "s, VLLM_MAX_AUDIO_DECODE_DURATION_S)"
            )
        if failures:
            raise StartupError(failures)
        if not enabled:
            logger.warning(
                "%s: complete files are disabled: the smaller of the "
                "state-slot count (%d) and the active-request limit (%d) is "
                "below 2 and max_concurrent_files is not configured",
                PLUGIN_KEY,
                self.capacity.slots,
                self.capacity.active_limit,
            )
        self._warn_inherited_sampling(service)
        self.files_enabled = enabled
        return dataclasses.replace(
            config, max_concurrent_files=concurrent, max_file_owners=owners
        )

    @staticmethod
    def _warn_inherited_sampling(service: Any) -> None:
        """Warn when the served model's generation settings set a guarded key.

        Complete files decode greedily, so an inherited sampling setting is
        ignored; the operator's configuration is not inspected per request.
        """
        from vllm_riva_frontend.files import FORWARDED_SAMPLING_FIELDS

        settings = getattr(service, "default_sampling_params", None)
        if not isinstance(settings, Mapping):
            return
        guarded = sorted(set(settings) & set(FORWARDED_SAMPLING_FIELDS))
        if guarded:
            logger.warning(
                "%s: the served model's generation settings set %s; complete "
                "files decode greedily and ignore them",
                PLUGIN_KEY,
                ", ".join(guarded),
            )

    # @spec ING-EP-085, ING-EP-086
    def _decide_authentication(
        self, args: Any, environment_key: str | None, middleware: type
    ) -> list[str] | None:
        """Decide the namespace's tokens from core's two signals.

        The configured key follows core's selection rule; the registrations
        are the app's registrations of core's ``AuthenticationMiddleware``.
        The tokens are never logged.
        """
        if self._app is None:
            raise StartupError(
                [f"{PLUGIN_KEY}: init_state ran before attach_router"]
            )
        keys = configured_keys(getattr(args, "api_key", None), environment_key)
        registrations = [
            dict(entry.kwargs)
            for entry in getattr(self._app, "user_middleware", ())
            if entry.cls is middleware
        ]
        return decide_tokens(bool(keys), registrations)

    def _build(self, config: FrontendConfig) -> None:
        """Build the registry, ingress cap, file gate and error catalog."""
        registry = OwnerRegistry()
        registry.on_fault = self.escalate
        self.registry = registry
        assert config.max_file_owners is not None
        self.ingress = IngressCap(
            max_sessions=config.load_shed_max_sessions,
            max_file_owners=config.max_file_owners,
            serving=self._core_serving,
        )
        self.file_gate = None
        if self.files_enabled:
            from vllm_riva_frontend.files import FileGate

            assert config.max_concurrent_files is not None
            self.file_gate = FileGate(
                config.max_concurrent_files,
                clock=self.clock,
                serving=self._core_serving,
            )
        self.catalog = catalog()

    def _core_serving(self) -> bool:
        """Whether core has not set ``state.draining``."""
        return not getattr(self.state, "draining", False)

    # @spec ING-EP-069, ING-EP-064, ING-EP-076
    async def _discover_factory(
        self, engine_client: Any, config: FrontendConfig
    ) -> None:
        """Discover the one session factory, owning every one constructed.

        Each factory discovery discards gets the factory settlement; if
        discovery or a later check fails, so does every factory constructed
        and not yet settled, before the error propagates. A cancellation
        never cuts a settlement short: it is held until the settlement in
        progress ends, then the selected factory is settled too and the
        cancellation re-raised, so a cancelled ``init_state`` leaves no
        factory behind.
        """
        from vllm_riva_frontend.lease import discover_session_factory

        discarded: list[Any] = []
        try:
            factory = discover_session_factory(
                engine_client,
                selected=config.session_factory,
                entry_points=self._entry_points,
                discard=discarded.append,
            )
        except StartupError as error:
            await self._settle_failed(discarded, error)
            raise
        except Exception as error:
            failure = StartupError(
                [f"{PLUGIN_KEY}: session factory discovery failed: {error}"]
            )
            await self._settle_failed(discarded, failure)
            raise failure from error
        if await self._settle_factories(discarded, "factory discovery"):
            await self._settle_factories([factory], "a cancelled startup")
            raise asyncio.CancelledError
        try:
            self._check_cadence(factory)
            from vllm_riva_frontend.grpc import RivaServicer

            self.factory = factory
            self.servicer = RivaServicer(self)
        except BaseException as error:
            self.factory = None
            self.servicer = None
            await self._settle_failed([factory], error)
            raise

    async def _settle_failed(
        self, factories: list[Any], failure: BaseException
    ) -> None:
        """Settle ``factories`` for a failed startup; then the caller raises.

        A cancellation that arrived during the settlement replaces the
        failure as what propagates.
        """
        if await self._settle_factories(factories, "a failed startup"):
            if isinstance(failure, asyncio.CancelledError):
                raise failure
            raise asyncio.CancelledError from failure

    async def _settle_factories(self, factories: list[Any], where: str) -> bool:
        """Run the factory settlement on each factory, to its end.

        Returns whether the caller was cancelled meanwhile; the caller
        re-raises that cancellation once its own settlement is complete.
        """
        interrupted = False
        for factory in factories:
            if await _run_to_end(self._settle_and_report(factory, where)):
                interrupted = True
        return interrupted

    async def _settle_and_report(self, factory: Any, where: str) -> None:
        faults, last_resort = await self._factory_settlement(factory)
        _report_settlement(faults, last_resort, where)

    # @spec ING-CORE-014
    def _check_cadence(self, factory: Any) -> None:
        from vllm_riva_frontend.lease import STREAM_CADENCE

        served = getattr(factory, "served_cadences", ())
        if STREAM_CADENCE not in served:
            raise StartupError(
                [
                    f"{PLUGIN_KEY}: the session factory does not serve the "
                    f"streaming cadence {STREAM_CADENCE} (it serves "
                    f"{sorted(served)})"
                ]
            )

    # --------------------------------------------------------------- start

    # @spec ING-EP-005, ING-EP-080, ING-EP-064, ING-EP-055, ING-VEH-002
    async def start(self, state: Any) -> None:
        """Bind and start the gRPC listener, then open admission last.

        Health stays ``NOT_SERVING`` until admission opens, and stays so
        when core's ``state.draining`` is already set then. A failure or a
        cancellation rolls back what exists and re-raises.
        """
        if self.servicer is None or self.config is None:
            raise StartupError([f"{PLUGIN_KEY}: start before init_state"])
        from vllm_riva_frontend.grpc import build_server

        config = self.config
        try:
            self.server, self.health = build_server(config, self.servicer)
            await self._set_health(_NOT_SERVING)
            try:
                bound = self.server.add_insecure_port(config.grpc_bind)
            except RuntimeError as error:
                raise StartupError(
                    [f"{PLUGIN_KEY}.grpc_bind: cannot bind {config.grpc_bind}"]
                ) from error
            if not bound:
                raise StartupError(
                    [f"{PLUGIN_KEY}.grpc_bind: cannot bind {config.grpc_bind}"]
                )
            await self.server.start()
            assert self.registry is not None and self.ingress is not None
            collector = FaultCollector(
                self.factory,
                COLLECTOR_PERIOD,
                self.escalate,
                clock=self.clock,
                lock=self._factory_lock,
            )
            self._collector = collector
            self.fault_collector = asyncio.ensure_future(collector.run())
            monitor = ListenerMonitor(
                self.server,
                self._on_listener_exit,
                lambda: self._listener_closing,
            )
            self.listener_monitor = asyncio.ensure_future(monitor.run())
            self.ingress.open()
            self._serving = True
            if not getattr(state, "draining", False):
                await self._set_health(_SERVING)
        except BaseException as failure:
            interrupted = await _run_to_end(self._roll_back_start())
            if interrupted and not isinstance(failure, asyncio.CancelledError):
                raise asyncio.CancelledError from failure
            raise

    # @spec ING-EP-064, ING-EP-100, ING-EP-097
    async def _roll_back_start(self) -> None:
        """Undo a failed ``start``: no listener exit here is a fault."""
        self._serving = False
        self._listener_closing = True
        if self.ingress is not None:
            self.ingress.close()
        supervisors = [
            task
            for task in (self.fault_collector, self.listener_monitor)
            if task is not None
        ]
        for task in supervisors:
            task.cancel()
        if self.server is not None:
            try:
                await self.server.stop(None)
            except Exception:
                logger.exception("%s: listener stop failed", PLUGIN_KEY)
            else:
                self._listener_stopped = True
        if supervisors:
            await asyncio.wait(supervisors)
        if self.factory is not None:
            await self._settle_and_report(self.factory, "a rolled-back start")

    async def _set_health(self, status: int) -> None:
        """Set the server's and the Riva service's health to ``status``."""
        if self.health is None:
            return
        from vllm_riva_frontend.grpc import RIVA_SERVICE_NAME

        for service in ("", RIVA_SERVICE_NAME):
            outcome = self.health.set(service, status)
            if inspect.isawaitable(outcome):
                await outcome

    def _on_listener_exit(self, error: BaseException | None) -> None:
        """A listener exit not caused by drain, stop or rollback is a fault."""
        if self._listener_closing:
            return
        self.escalate(
            error
            if error is not None
            else RuntimeError("the Riva gRPC listener exited unexpectedly")
        )

    # --------------------------------------------------------------- ready

    # @spec ING-EP-065, ING-EP-082
    def ready(self, state: Any) -> bool:
        """Whether the listener serves, nothing drains and no fault is recorded.

        Reads attributes only, core's ``state.draining`` among them.
        """
        return (
            self._serving
            and not self.draining
            and not self.torn_down
            and self.fault is None
            and not getattr(state, "draining", False)
        )

    # -------------------------------------------------------------- faults

    def _close_admission(self) -> None:
        if self.ingress is not None:
            self.ingress.close()
        if self.file_gate is not None:
            self.file_gate.close()

    # @spec ING-EP-111, ING-EP-044, ING-EP-045, ING-EP-101
    def escalate(self, error: BaseException) -> None:
        """Record a fault, close admission, then ask core for a restart.

        The triggers are an ``engine_dead`` failure on the stream path, a
        fault the collector returns and an unexpected listener exit. The
        fault makes ``ready`` false; admission and the file gate close
        before ``state.request_shutdown()`` is called, once. gRPC health
        and core's ``state.draining`` are left as they are.
        """
        if self.fault is None:
            self.fault = error
        logger.error(
            "%s: fault, requesting an orderly restart: %r", PLUGIN_KEY, error
        )
        self._close_admission()
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        request = getattr(self.state, "request_shutdown", None)
        if callable(request):
            request()
        else:
            logger.error("%s: state.request_shutdown is missing", PLUGIN_KEY)

    # --------------------------------------------------------------- drain

    # @spec ING-EP-066, ING-EP-036, ING-EP-037, ING-EP-109, ING-EP-073
    # @spec ING-EP-074, ING-EP-075, ING-EP-076, ING-EP-046, ING-EP-082
    async def drain(self, state: Any, timeout: float) -> None:
        """Finish or end every owner by ``D``, then settle everything.

        ``D`` is ``now + timeout - engine_reserve(timeout)``; every wait of
        the graceful work is bounded by it. The listener keeps answering
        ``NOT_SERVING`` until the settle point, where stream owners get the
        shutdown trigger and file requests are cancelled; then the listener
        stops with the time left to ``D`` as grace, the collector ends and
        ``factory.drain(D)`` runs. When ``timeout`` is within the reserve the
        graceful steps are skipped. The settlement always follows, through
        any cancellation, and only then is a failure or cancellation
        re-raised. A ``factory.drain(D)`` failing by ``D`` is a failing
        step; one still running at ``D`` is awaited by the settlement, and
        its late failure (an exception or its cancellation) is raised after
        the settlement completes, unless ``drain`` was cancelled or an
        earlier step failed (ING-EP-076). The cancellation of ``drain``
        takes precedence, then the earliest failure; a late exception that
        is not the one raised is logged.
        """
        del state
        clock = self.clock
        now = clock.monotonic()
        graceful = timeout > engine_reserve(timeout)
        self.draining = True
        self._serving = False
        registry = self.registry
        deadline = now + timeout - engine_reserve(timeout)
        if graceful:
            self.drain_deadline = deadline
            if registry is not None:
                registry.drain_deadline = deadline
        elif registry is not None:
            registry.send_bounds_passed = True
        # Sends already waiting re-clip to ``D`` now (ING-LIFE-061).
        self._announce_drain()
        failure: BaseException | None = None
        try:
            await self._set_health(_NOT_SERVING)
            self._close_admission()
            if graceful:
                await self._graceful(now, deadline)
        except BaseException as error:
            failure = error
        late: list[BaseException | None] = [None]

        async def settle() -> None:
            late[0] = await self._settlement("drain")

        try:
            interrupted = await _run_to_end(settle())
        finally:
            self._end_drain_call()
        if failure is None and not interrupted:
            # A ``factory.drain(D)`` that ended after ``D`` failed as a
            # step; its outcome is raised now that the settlement is done.
            failure = late[0]
        elif isinstance(late[0], Exception):
            # The cancellation of ``drain``, or the earlier failure, is the
            # one re-raised; the late failure is logged instead.
            _log_late_factory_drain(late[0])
        if interrupted or isinstance(failure, asyncio.CancelledError):
            raise asyncio.CancelledError from failure
        if failure is not None:
            raise failure

    async def _graceful(self, now: float, deadline: float) -> None:
        """The graceful steps of ``drain``, every wait bounded by ``D``."""
        clock = self.clock
        registry = self.registry
        # Health checkers see NOT_SERVING until the settle point, owners or
        # not (ING-EP-109).
        settle = settle_point(now, deadline)
        await clock.sleep(max(0.0, settle - clock.monotonic()))
        if registry is not None:
            registry.deliver_shutdown(ShutdownTrigger(deadline))
            assert self.config is not None
            files_by = min(
                clock.monotonic() + self.config.session_cleanup_timeout,
                deadline,
            )
            await self._wait_until(
                lambda: registry.settled(FILE_KINDS), files_by
            )
            # Each stream owner acts on the trigger (enters its shutdown
            # finalization, selects a cause or ends) before the listener
            # stop can terminate its RPC (ING-EP-074, ING-LIFE-064).
            await self._wait_for_triggers(registry, deadline)
        # The settle-point steps are complete: the graceful listener stop
        # begins now, while stream owners still finalize under its grace
        # (ING-EP-074); an RPC it then terminates is the plugin's
        # cancellation of its owner (ING-LIFE-064).
        await self._stop_listener(max(0.0, deadline - clock.monotonic()))
        if registry is not None:
            await self._wait_until(
                lambda: registry.settled(STREAM_KINDS), deadline
            )
        # Both waits are bounded by ``D``: on expiry the settlement begins
        # and awaits what is still running (ING-EP-066, ING-EP-093).
        if not await self._end_collector(deadline):
            return
        await self._drain_factory(deadline)

    # @spec ING-EP-074, ING-LIFE-064
    async def _wait_for_triggers(
        self, registry: OwnerRegistry, deadline: float
    ) -> None:
        """Wait, by ``D``, until every stream owner took the trigger or ended.

        An owner has taken it once its ``trigger_taken`` future resolved; a
        record with no such owner attached is not waited for.
        """

        def untaken() -> set[asyncio.Future[Any]]:
            waiting: set[asyncio.Future[Any]] = set()
            for record in list(registry.records.values()):
                if record.kind not in STREAM_KINDS or record.task.done():
                    continue
                observer = _shutdown_observer(record)
                taken = getattr(observer, "trigger_taken", None)
                if isinstance(taken, asyncio.Future) and not taken.done():
                    waiting.update((taken, record.task))
            return waiting

        while (waiting := untaken()) and self.clock.monotonic() < deadline:
            change = registry.changed()
            try:
                await self.clock.wait_for(
                    asyncio.wait(
                        {*waiting, change}, return_when=asyncio.FIRST_COMPLETED
                    ),
                    deadline,
                )
            except TimeoutError:
                return
            finally:
                change.cancel()

    async def _wait_until(
        self, done: Callable[[], bool], deadline: float
    ) -> bool:
        """Wait until ``done()`` holds or ``deadline``; return whether it held.

        Wakes when an owner task or continuation finishes or the registry
        changes.
        """
        registry = self.registry
        assert registry is not None
        while not done():
            if self.clock.monotonic() >= deadline:
                return False
            change = registry.changed()
            watched: set[asyncio.Future[Any]] = {change, *registry.pending()}
            try:
                await self.clock.wait_for(
                    asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED),
                    deadline,
                )
            except TimeoutError:
                pass
            finally:
                change.cancel()
        return True

    # @spec ING-EP-066, ING-EP-093
    async def _drain_factory(self, deadline: float) -> None:
        """Call ``factory.drain(D)``, waiting for it until ``D`` at most.

        The call is never cancelled: when ``D`` passes first it keeps
        running and the settlement awaits it (``_await_factory_drain``)
        before the factory settlement. Its faults are logged when it ends.
        """
        if self.factory is None:
            return
        factory = self.factory

        async def call() -> None:
            async with self._factory_lock:
                faults = await factory.drain(deadline)
            _report_settlement(faults or (), (), "the drain deadline")

        work = asyncio.ensure_future(call())
        self._factory_drain = work
        if await self._wait_by(work, deadline):
            self._factory_drain = None
            work.result()

    async def _wait_by(
        self, task: asyncio.Future[Any], deadline: float
    ) -> bool:
        """Wait for ``task`` until ``deadline``, never cancelling it.

        Returns whether it has ended; its outcome is the caller's to read.
        """
        if not task.done():
            timer = self.clock.timer(deadline)
            try:
                await asyncio.wait(
                    {task, timer}, return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                timer.cancel()
        return task.done()

    # @spec ING-EP-076, ING-EP-093
    async def _await_factory_drain(self) -> BaseException | None:
        """Await a ``factory.drain(D)`` call still running; return its outcome.

        A failure in time (the call ended by ``D``) was raised by
        ``_drain_factory`` as a failing step of the graceful work. A call
        still running when ``D`` passed is awaited here, uncancelled, and a
        late failure (an exception, or the call ending cancelled) is
        returned, not raised, so the settlement still runs the factory
        settlement; ``drain`` then propagates it (ING-EP-076). ``None``
        when there is no such call or it returned.
        """
        work, self._factory_drain = self._factory_drain, None
        if work is None:
            return None
        await asyncio.wait({work})
        if work.cancelled():
            return asyncio.CancelledError()
        return work.exception()

    async def _stop_listener(self, grace: float | None) -> None:
        """Stop the gRPC server once; its exit is then not a fault."""
        if self.server is None or self._listener_stopped:
            return
        self._listener_closing = True
        await self.server.stop(grace)
        self._listener_stopped = True

    async def _end_collector(self, deadline: float = math.inf) -> bool:
        """Signal the fault collector to end and await it, uncancelled.

        The wait ends at ``deadline`` at the latest (the graceful work's
        ``D``); returns whether the collector has ended.
        """
        task = self.fault_collector
        if task is None or task.done():
            return True
        if self._collector is not None:
            self._collector.request_stop()
        if deadline == math.inf:
            await asyncio.wait({task})
            return True
        return await self._wait_by(task, deadline)

    # ---------------------------------------------------------- settlement

    def _end_clipping(self) -> None:
        """Clipping to ``D`` ends and every send bound passes (ING-EP-110)."""
        self.drain_deadline = None
        registry = self.registry
        if registry is not None:
            registry.drain_deadline = None
            registry.send_bounds_passed = True
        self._announce_drain()

    # @spec ING-EP-110, ING-LIFE-061
    def _end_drain_call(self) -> None:
        """The ``drain`` call returns: its passed send bounds end with it.

        Every send bound counts as passed only for the rest of that call
        (ING-EP-110), and from ``stop`` on; afterwards a send gets its
        ordinary bound again. The change is announced to waiting sends.
        """
        registry = self.registry
        if self.torn_down or registry is None or registry.torn_down:
            return
        registry.send_bounds_passed = False
        self._announce_drain()

    # @spec ING-EP-093, ING-EP-110, ING-EP-097, ING-EP-098, ING-EP-103
    async def _settlement(self, where: str) -> BaseException | None:
        """Settle every owner, the listener, the collector and the factory.

        In order: cancel every registered owner's task; stop the listener
        without grace if it still serves; close the registry to sends
        without an owner and await every owner, continuation, child and
        unsettled send; end the collector; await a ``factory.drain(D)``
        still running; run the factory settlement. Every await is
        cooperative, with no time limit. Returns the late outcome of that
        ``factory.drain(D)`` call, for the caller to propagate once the
        settlement is complete (ING-EP-076).
        """
        self._end_clipping()
        registry = self.registry
        if registry is not None:
            registry.cancel_owners()
            # Each owner takes its cancellation before the listener stops.
            await asyncio.sleep(0)
        try:
            await self._stop_listener(None)
        except Exception:
            logger.exception("%s: listener stop failed", PLUGIN_KEY)
        if registry is not None:
            registry.close_sends()
            await self._await_registry(registry)
        await self._end_collector()
        late = await self._await_factory_drain()
        if self.factory is not None:
            await self._settle_and_report(self.factory, where)
        return late

    @staticmethod
    async def _await_registry(registry: OwnerRegistry) -> None:
        """Await everything the registry holds until each has ended."""
        current = asyncio.current_task()
        while True:
            pending = {
                task for task in registry.outstanding() if task is not current
            }
            if not pending:
                break
            await asyncio.wait(pending)
        # Let the done callbacks that unregister records run; a record left
        # with nothing running is no longer an owner.
        await asyncio.sleep(0)
        registry.discard_ended()

    # @spec ING-EP-096, ING-EP-099
    async def _factory_settlement(
        self, factory: Any
    ) -> tuple[list[BaseException], list[asyncio.Future[Any]]]:
        """``drain(math.inf)``, ``teardown_all``, await, ``drain(now)``.

        Returns the faults both drains returned, as returned, and the
        last-resort faults: each task ``teardown_all`` returned that was
        not done when it returned. No exception of a returned task is read.
        """
        faults: list[BaseException] = []
        try:
            async with self._factory_lock:
                faults.extend(await factory.drain(math.inf) or ())
        except Exception as error:
            faults.append(error)
        try:
            returned = {
                task
                for task in (factory.teardown_all() or ())
                if isinstance(task, asyncio.Future)
            }
        except Exception as error:
            faults.append(error)
            returned = set()
        last_resort = [task for task in returned if not task.done()]
        while any(not task.done() for task in returned):
            await asyncio.wait(returned)
        try:
            async with self._factory_lock:
                faults.extend(await factory.drain(self.clock.monotonic()) or ())
        except Exception as error:
            faults.append(error)
        return faults, last_resort

    # ---------------------------------------------------------------- stop

    # @spec ING-EP-079, ING-EP-077, ING-EP-093, ING-EP-102
    async def stop(self, state: Any) -> None:
        """Mark the plugin torn down, settle everything, end the monitor.

        Admission refuses and ``ready`` is false from the first step; the
        settlement has no graceful phase and every send bound counts as
        passed; the listener monitor is cancelled and awaited last. A
        cancellation does not interrupt these steps and is re-raised
        afterwards.
        """
        del state
        self.torn_down = True
        self._serving = False
        registry = self.registry
        if registry is not None:
            registry.torn_down = True
        self._close_admission()
        # Every send bound now counts as passed: wake the waiting sends.
        self._announce_drain()
        interrupted = await _run_to_end(self._stop_steps())
        if interrupted:
            raise asyncio.CancelledError

    async def _stop_steps(self) -> None:
        late = await self._settlement("stop")
        if isinstance(late, Exception):
            # Only a ``drain`` call propagates its factory drain's outcome.
            _log_late_factory_drain(late)
        monitor = self.listener_monitor
        if monitor is not None and not monitor.done():
            monitor.cancel()
            await asyncio.wait({monitor})
        with contextlib.suppress(Exception):
            if monitor is not None and not monitor.cancelled():
                monitor.exception()
