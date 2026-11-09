"""The endpoint plugin vLLM core sees: startup, owners, shutdown and faults.

``RivaFrontendPlugin`` implements the lifecycle core defines for endpoint
plugins (``attach_router``, ``init_state``, ``start``, ``ready``, ``drain``,
``stop``) and adds none. It owns the order of the startup checks, the
registry of transport owner tasks, the ingress cap, the relation between its
bounds and the engine's stream deadlines, shutdown and fault escalation.

The surface modules (``grpc``, ``nim_ws``, ``files``) and the session seam
(``lease``) are imported inside the lifecycle methods: they read this
module's registry types, so a module-level import would be circular.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import importlib.metadata
import inspect
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from grpc_health.v1 import health_pb2
from starlette.routing import Match

from vllm_riva_frontend.admission import FILE_KINDS, IngressCap, OwnerKind
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
from vllm_riva_frontend.errors import ErrorCatalog, FrontendError, catalog
from vllm_riva_frontend.operational import operational_router

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

#: Core's complete-file route, which the frontend's file route replaces.
FILE_ROUTE_PATH = "/v1/audio/transcriptions"
#: Framing allowance of the multipart body above core's upload limit.
UPLOAD_FRAMING_BYTES = 262_144
#: The fault collector's period; a protocol constant, not an option.
COLLECTOR_PERIOD = 1.0
#: The one bounded wait of ``stop``, so cancellation is delivered.
STOP_WAIT = 1.0
#: How long one listener-termination wait lasts before it is renewed.
_MONITOR_POLL = 3600.0

#: The owner kinds the shutdown trigger finalizes; files are cancelled.
STREAM_KINDS = frozenset({OwnerKind.stream, OwnerKind.realtime})

_SERVING = health_pb2.HealthCheckResponse.SERVING
_NOT_SERVING = health_pb2.HealthCheckResponse.NOT_SERVING


# ----------------------------------------------------------- the formulas


# @spec ING-EP-010
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
    cleanup. ``stop`` cancels all of them through ``cancel_all``.
    """

    kind: OwnerKind
    task: asyncio.Task[Any]
    continuation: asyncio.Task[Any] | None = None
    children: set[asyncio.Task[Any]] = field(default_factory=set)
    owner: Any = None
    held_by_handler: bool = False
    unsettled_sends: set[asyncio.Task[Any]] = field(default_factory=set)


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

    Besides the records, it carries what an owner needs from the plugin
    without holding the plugin: ``torn_down`` (owners then call the lease's
    synchronous ``teardown`` only), ``drain_deadline`` and
    ``shutdown_trigger`` while draining, and ``escalate`` for a failure that
    escalates to a requested shutdown.
    """

    # @spec ING-EP-060
    def __init__(self) -> None:
        """Create an empty registry."""
        self._records: dict[int, OwnerRecord] = {}
        self._waiters: set[asyncio.Future[None]] = set()
        self._kept: set[asyncio.Task[Any]] = set()
        self.torn_down = False
        self.drain_deadline: float | None = None
        self.shutdown_trigger: ShutdownTrigger | None = None
        self.on_fault: Callable[[BaseException], None] | None = None

    @property
    def records(self) -> Mapping[int, OwnerRecord]:
        """The live records, read-only."""
        return MappingProxyType(self._records)

    # @spec ING-EP-060, ING-ADM-013
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

    # @spec ING-EP-060
    def keep(self, task: asyncio.Task[Any]) -> None:
        """Keep ``task``, which belongs to no record, until it ends.

        A refusal creates no owner, so its send is kept here instead of
        under a record; ``cancel_all`` cancels it.
        """
        if not task.done():
            self._kept.add(task)
            task.add_done_callback(self._kept.discard)

    def cancel_all(self) -> list[asyncio.Task[Any]]:
        """Cancel every owner, continuation, child and kept task; return all."""
        tasks: list[asyncio.Task[Any]] = list(self._kept)
        for record in list(self._records.values()):
            tasks.append(record.task)
            if record.continuation is not None:
                tasks.append(record.continuation)
            tasks.extend(record.children)
        current = asyncio.current_task()
        for task in tasks:
            if task is not current:
                task.cancel()
        return tasks

    def cancel_owners(self) -> list[asyncio.Task[Any]]:
        """Cancel the owner tasks only; their cleanup continues."""
        current = asyncio.current_task()
        tasks = [record.task for record in self._records.values()]
        for task in tasks:
            if task is not current:
                task.cancel()
        return tasks

    # @spec ING-EP-037, ING-EP-038
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
                record.task.cancel()
            else:
                self._deliver(record, trigger)

    def _deliver(self, record: OwnerRecord, trigger: ShutdownTrigger) -> None:
        if record.task.done():
            return
        observer = _shutdown_observer(record)
        if observer is None:
            record.task.cancel()
            return
        try:
            outcome = observer.observe_shutdown(trigger)
        except Exception:
            logger.exception("%s: an owner refused the trigger", PLUGIN_KEY)
            record.task.cancel()
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
        """Return a future resolved at the next register or unregister."""
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
    """Collects the factory's cleanup faults while the plugin serves.

    Every ``period`` it calls ``factory.drain`` with an immediate deadline,
    under the lock the shutdown ``drain`` also takes, so two callers never
    take the same faults.
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

    # @spec ING-EP-015
    async def run(self) -> None:
        """Collect until cancelled."""
        while True:
            await self._clock.sleep(self._period)
            try:
                async with self._lock:
                    faults = await self._factory.drain(self._clock.monotonic())
            except Exception as error:  # a failing drain is a fault itself
                faults = [error]
            for fault in faults or ():
                if self._on_fault is not None:
                    self._on_fault(fault)


class ListenerMonitor:
    """Watches the gRPC server and reports its exit."""

    def __init__(
        self,
        server: Any,
        on_exit: Callable[[BaseException | None], None],
    ) -> None:
        """Bind the server and the exit handler."""
        self._server = server
        self._on_exit = on_exit

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


# ------------------------------------------------------------ the plugin


def _serves(app: Any, path: str, method: str) -> bool:
    """Whether ``app`` already has a route for ``method path``."""
    return any(
        getattr(route, "path", None) == path
        and method in (getattr(route, "methods", None) or ())
        for route in app.routes
    )


def _routing_target(app: Any, path: str, method: str) -> Any:
    """The route ``app`` dispatches ``method path`` to, or ``None``."""
    scope = {
        "type": "http",
        "path": path,
        "root_path": "",
        "method": method,
        "headers": [],
        "query_string": b"",
    }
    for route in app.router.routes:
        match, _ = route.matches(scope)
        if match is Match.FULL:
            return route
    return None


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


def _core_upload_limit_bytes() -> int:
    """Core's upload limit (``VLLM_MAX_AUDIO_CLIP_FILESIZE_MB``) in bytes."""
    import vllm.envs as envs

    return int(envs.VLLM_MAX_AUDIO_CLIP_FILESIZE_MB * 1024 * 1024)


def _core_decode_limit(service: Any) -> float:
    """Core's decode-duration limit, as the transcription service holds it."""
    limit = getattr(service, "max_audio_decode_duration_s", None)
    if isinstance(limit, (int, float)) and not isinstance(limit, bool):
        return float(limit)
    import vllm.envs as envs

    return float(envs.VLLM_MAX_AUDIO_DECODE_DURATION_S)


# @spec ING-EP-020, ING-VEH-005, ING-VEH-008
class RivaFrontendPlugin:
    """The Riva frontend as one vLLM endpoint plugin.

    Public attributes after ``init_state``: ``config``, ``served_model``,
    ``deadlines``, ``capacity``, ``files_enabled``, ``factory``,
    ``registry``, ``ingress``, ``file_gate``, ``catalog``, ``servicer``,
    ``provenance``, ``transcription`` (core's service or ``None``),
    ``state`` and ``engine_client``. After ``start``: ``server``,
    ``health``, ``fault_collector`` and ``listener_monitor``. Throughout:
    ``clock``, ``draining``, ``drain_deadline``, ``torn_down`` and
    ``fault``.
    """

    name = "riva_frontend"
    required_tasks: tuple[str, ...] = ("realtime",)
    owns_non_http_listener = True

    # @spec ING-EP-006
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
        self._handlers: dict[tuple[str, str], Callable[..., Any]] = {}
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
        self._serving = False
        self._listener_closing = False
        self._shutdown_requested = False
        # Serializes factory.drain between the collector and drain.
        self._factory_lock = asyncio.Lock()
        # Futures of the sends waiting for ``drain`` to set ``D``.
        self._drain_waiters: set[asyncio.Future[None]] = set()

    @property
    def admission_open(self) -> bool:
        """Whether the ingress cap admits owners."""
        return self.ingress is not None and self.ingress.is_open

    # @spec ING-LIFE-051
    def drain_announced(self) -> asyncio.Future[None]:
        """Return a future resolved once ``drain`` has set ``D``.

        It is already resolved while draining. Every send in flight waits on
        one, with or without an owner, so it re-clips its bound to ``D`` the
        moment the drain begins rather than at the settle point.
        """
        future: asyncio.Future[None] = (
            asyncio.get_running_loop().create_future()
        )
        if self.drain_deadline is not None:
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

    # @spec ING-EP-007, ING-EP-004, ING-SHIM-008, ING-FILE-003
    # @spec ING-SHIM-013, ING-SHIM-014
    def attach_router(self, app: Any) -> None:
        """Add the bootstrap, operational and file routes and the dispatcher.

        The operational and bootstrap paths are added as new routes, so an
        existing route on one of them fails startup here. The file route is
        added only when core's ``POST /v1/audio/transcriptions`` exists, and
        replaces it. The dispatcher is middleware: core adds its own
        middleware after the plugin hooks, so core's authentication wraps
        the dispatcher and ``--api-key`` applies to the claimed scopes.
        """
        from vllm_riva_frontend.files import file_route
        from vllm_riva_frontend.nim_ws import BOOTSTRAP_PATH, RealtimeDispatcher

        router = operational_router(self)
        router.add_api_route(
            BOOTSTRAP_PATH,
            self._bootstrap_endpoint(),
            methods=["POST"],
            response_model=None,
        )
        collisions = [
            f"{PLUGIN_KEY}: {method} {route.path} is already served; the "
            "frontend adds this path as new"
            for route in router.routes
            if isinstance(route, APIRoute)
            for method in sorted(getattr(route, "methods", None) or ())
            if _serves(app, route.path, method)
        ]
        if collisions:
            raise StartupError(collisions)
        if _serves(app, FILE_ROUTE_PATH, "POST"):
            router.add_api_route(
                FILE_ROUTE_PATH,
                file_route(self),
                methods=["POST"],
                response_model=None,
            )
        app.include_router(router)
        self._handlers = {
            (route.path, method): route.endpoint
            for route in router.routes
            if isinstance(route, APIRoute)
            for method in getattr(route, "methods", None) or ()
        }
        app.add_middleware(RealtimeDispatcher, plugin=self)
        self._app = app

    # @spec ING-LIFE-045
    def _bootstrap_endpoint(self) -> Callable[[], Any]:
        """The ``POST /v1/realtime/transcription_sessions`` handler."""

        async def realtime_transcription_sessions() -> JSONResponse:
            from vllm_riva_frontend.nim_ws import bootstrap_session

            if self.served_model is None or self.factory is None:
                error = FrontendError(
                    "service_unavailable", "the frontend is not initialized"
                )
                return JSONResponse(error.http_body(503), status_code=503)
            return JSONResponse(
                bootstrap_session(
                    self.served_model, self.factory.default_locale
                )
            )

        return realtime_transcription_sessions

    # ---------------------------------------------------------- init_state

    # @spec ING-EP-003, ING-EP-023, ING-EP-024, ING-EP-025, ING-EP-021
    # @spec ING-EP-031, ING-EP-027, ING-ADM-014, ING-VEH-026
    async def init_state(
        self, engine_client: Any, state: Any, args: Any
    ) -> None:
        """Run the startup checks in order, then build the plugin's parts.

        Order: configuration, served identity, engine stream deadlines,
        capacity, complete-file service, the build (including the route
        table), and the session factory last. Each step raises one
        ``StartupError`` naming everything that failed within it. A failure
        after the factory exists tears the factory down first. No listener
        is bound and no task is started.
        """
        del args  # never an identity or option source
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
        config = self._size_file_service(config, service)
        self.config = config
        self.transcription = service
        self.provenance = build_deployment_provenance(config)

        self._build(config)
        self._check_route_table()

        self.factory = self._discover_factory(engine_client, config)
        try:
            self._check_cadence()
            from vllm_riva_frontend.grpc import RivaServicer

            self.servicer = RivaServicer(self)
        except BaseException:
            self._teardown_factory()
            raise

        if getattr(vllm_config, "shutdown_timeout", None) == 0:
            logger.warning(
                "%s: --shutdown-timeout is 0, so no drain will run; "
                "shutdown is stop alone, which cancels open streams",
                PLUGIN_KEY,
            )

    # @spec ING-FILE-006, ING-FILE-009, ING-FILE-015, ING-FILE-016
    # @spec ING-FILE-018, ING-FILE-046, ING-FILE-047, ING-FILE-054
    def _size_file_service(
        self, config: FrontendConfig, service: Any
    ) -> FrontendConfig:
        """Size the file gate and sub-cap and check core's file limits."""
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
        upload_limit = _core_upload_limit_bytes()
        envelope = config.http_multipart_envelope_max_bytes
        if envelope < upload_limit + UPLOAD_FRAMING_BYTES:
            failures.append(
                f"{PLUGIN_KEY}.http_multipart_envelope_max_bytes ({envelope}) "
                f"must be at least core's upload limit ({upload_limit} bytes, "
                f"VLLM_MAX_AUDIO_CLIP_FILESIZE_MB) plus "
                f"{UPLOAD_FRAMING_BYTES} bytes of framing"
            )
        decode_limit = _core_decode_limit(service)
        duration = config.unary_max_decoded_duration_seconds
        if duration > decode_limit:
            failures.append(
                f"{PLUGIN_KEY}.unary_max_decoded_duration_seconds ({duration}) "
                f"must be at most core's decode-duration limit ({decode_limit} "
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
                config.max_concurrent_files, clock=self.clock
            )
        self.catalog = catalog()

    def _core_serving(self) -> bool:
        """Whether core has not set ``state.draining``."""
        return not getattr(self.state, "draining", False)

    def _check_route_table(self) -> None:
        """Fail unless core's final table routes each path to this plugin."""
        if self._app is None:
            raise StartupError(
                [f"{PLUGIN_KEY}: init_state ran before attach_router"]
            )
        failures = [
            f"{PLUGIN_KEY}: {method} {path} no longer reaches the frontend's "
            "handler; another endpoint plugin overrides it"
            for (path, method), endpoint in self._handlers.items()
            if getattr(
                _routing_target(self._app, path, method), "endpoint", None
            )
            is not endpoint
        ]
        if failures:
            raise StartupError(failures)

    def _discover_factory(
        self, engine_client: Any, config: FrontendConfig
    ) -> Any:
        """Discover the one session factory (session leaf)."""
        from vllm_riva_frontend.lease import discover_session_factory

        try:
            return discover_session_factory(
                engine_client,
                selected=config.session_factory,
                entry_points=self._entry_points,
            )
        except StartupError:
            raise
        except Exception as error:
            raise StartupError(
                [f"{PLUGIN_KEY}: session factory discovery failed: {error}"]
            ) from error

    # @spec ING-CORE-014
    def _check_cadence(self) -> None:
        from vllm_riva_frontend.lease import STREAM_CADENCE

        served = getattr(self.factory, "served_cadences", ())
        if STREAM_CADENCE not in served:
            raise StartupError(
                [
                    f"{PLUGIN_KEY}: the session factory does not serve the "
                    f"streaming cadence {STREAM_CADENCE} (it serves "
                    f"{sorted(served)})"
                ]
            )

    def _teardown_factory(self) -> set[Any]:
        """Call ``factory.teardown_all()``; return the tasks it cancelled."""
        if self.factory is None:
            return set()
        try:
            tasks = self.factory.teardown_all()
        except Exception:
            logger.exception("%s: factory teardown failed", PLUGIN_KEY)
            return set()
        return set(tasks or ())

    # --------------------------------------------------------------- start

    # @spec ING-EP-005, ING-EP-035, ING-EP-008, ING-EP-055, ING-VEH-002
    async def start(self, state: Any) -> None:
        """Bind and start the gRPC listener, then open admission last.

        Health stays ``NOT_SERVING`` until admission opens. A failure tears
        down what was built (supervisors, listener, factory) and re-raises.
        """
        del state
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
            self.fault_collector = asyncio.ensure_future(collector.run())
            monitor = ListenerMonitor(self.server, self._on_listener_exit)
            self.listener_monitor = asyncio.ensure_future(monitor.run())
            self.ingress.open()
            self._serving = True
            await self._set_health(_SERVING)
        except BaseException:
            await self._roll_back_start()
            raise

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
        try:
            if self.server is not None:
                with contextlib.suppress(Exception):
                    await self.server.stop(0)
            if supervisors:
                await asyncio.wait(supervisors)
        finally:
            self._teardown_factory()

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

    # @spec ING-EP-009
    def ready(self, state: Any) -> bool:
        """Whether the listener serves, nothing drains and no fault is recorded.

        Reads attributes only.
        """
        return (
            self._serving
            and not self.draining
            and not self.torn_down
            and self.fault is None
            and not getattr(state, "draining", False)
        )

    # -------------------------------------------------------------- faults

    # @spec ING-EP-017, ING-EP-044, ING-EP-045
    def escalate(self, error: BaseException) -> None:
        """Record a fault and ask core for an orderly restart.

        The triggers are an ``engine_dead`` failure on the stream path, a
        fault the collector returns and an unexpected listener exit. The
        fault makes ``ready`` false; ``state.request_shutdown()`` starts
        core's graceful shutdown once.
        """
        if self.fault is None:
            self.fault = error
        logger.error(
            "%s: fault, requesting an orderly restart: %r", PLUGIN_KEY, error
        )
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        request = getattr(self.state, "request_shutdown", None)
        if callable(request):
            request()
        else:
            logger.error("%s: state.request_shutdown is missing", PLUGIN_KEY)

    # --------------------------------------------------------------- drain

    # @spec ING-EP-010, ING-EP-036, ING-EP-037, ING-EP-038, ING-EP-039
    # @spec ING-EP-040, ING-EP-041, ING-EP-046, ING-EP-015
    async def drain(self, state: Any, timeout: float) -> None:
        """Finish or end every owner by ``D``, then stop the listener.

        ``D`` is ``now + timeout - engine_reserve(timeout)``, leaving core
        time to shut the engine down; every wait here is bounded by it. Owners
        finish on their own until the settle point, where stream owners get
        the shutdown trigger and file requests are cancelled. When
        ``timeout`` is within the reserve, every owner is cancelled at once
        and the remaining time is core's whole budget. A cancelled ``drain``
        cancels the owners without awaiting their cleanup.
        """
        del state
        clock = self.clock
        now = clock.monotonic()
        graceful = timeout > engine_reserve(timeout)
        deadline = now + timeout - (engine_reserve(timeout) if graceful else 0)
        self.draining = True
        self._serving = False
        self.drain_deadline = deadline
        registry = self.registry
        if registry is not None:
            registry.drain_deadline = deadline
        # Sends already waiting re-clip to ``D`` now (ING-LIFE-051).
        self._announce_drain()
        try:
            await self._set_health(_NOT_SERVING)
            if self.ingress is not None:
                self.ingress.close()
            if self.file_gate is not None:
                self.file_gate.close()
            if registry is not None:
                if graceful:
                    # The listener keeps answering NOT_SERVING until the
                    # settle point, owners or not, so health checkers see it.
                    settle = settle_point(now, deadline)
                    await clock.sleep(max(0.0, settle - clock.monotonic()))
                    registry.deliver_shutdown(ShutdownTrigger(deadline))
                else:
                    registry.cancel_owners()
                assert self.config is not None
                files_by = min(
                    clock.monotonic() + self.config.session_cleanup_timeout,
                    deadline,
                )
                await self._wait_until(registry.settled, files_by)
                await self._wait_until(
                    lambda: registry.settled(STREAM_KINDS), deadline
                )
            if self.server is not None:
                self._listener_closing = True
                await self.server.stop(max(0.0, deadline - clock.monotonic()))
            await self._drain_factory(deadline)
        except asyncio.CancelledError:
            if registry is not None:
                registry.cancel_owners()
            raise

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

    async def _drain_factory(self, deadline: float) -> None:
        """Call ``factory.drain(D)`` and log the faults it returns."""
        if self.factory is None:
            return
        async with self._factory_lock:
            faults = await self.factory.drain(deadline)
        for fault in faults or ():
            logger.error(
                "%s: session cleanup fault at shutdown: %r", PLUGIN_KEY, fault
            )

    # ---------------------------------------------------------------- stop

    # @spec ING-EP-059, ING-EP-042, ING-EP-043
    async def stop(self, state: Any) -> None:
        """Tear everything down promptly, awaiting no engine work.

        Marks the plugin torn down (owners then call the lease's synchronous
        ``teardown`` only), tears the factory down, cancels every owner,
        continuation, child (a stream's unsettled sends included), kept task
        and supervisor, stops the listener without grace and waits once, at
        most 1 s, for the cancellations to be delivered.
        """
        del state
        self.torn_down = True
        self._serving = False
        self._listener_closing = True
        registry = self.registry
        if registry is not None:
            registry.torn_down = True
        if self.ingress is not None:
            self.ingress.close()
        if self.file_gate is not None:
            self.file_gate.close()
        cancelled: set[Any] = self._teardown_factory()
        if registry is not None:
            cancelled.update(registry.cancel_all())
        for supervisor in (self.fault_collector, self.listener_monitor):
            if supervisor is not None:
                supervisor.cancel()
                cancelled.add(supervisor)
        if self.server is not None:
            with contextlib.suppress(Exception):
                await self.server.stop(None)
        current = asyncio.current_task()
        pending = {
            task
            for task in cancelled
            if isinstance(task, asyncio.Future)
            and task is not current
            and not task.done()
        }
        if pending:
            with contextlib.suppress(TimeoutError):
                await self.clock.wait_for(
                    asyncio.wait(pending), self.clock.monotonic() + STOP_WAIT
                )
