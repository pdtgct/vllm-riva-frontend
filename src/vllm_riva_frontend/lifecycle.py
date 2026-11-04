"""Endpoint-plugin installation, readiness, and owner drainage."""

from __future__ import annotations

import asyncio
import importlib
import json
import time
from collections.abc import Callable
from contextlib import suppress
from importlib import metadata as importlib_metadata
from types import SimpleNamespace
from typing import Any, Protocol

import grpc
from grpc_health.v1 import health_pb2
from packaging.version import InvalidVersion, Version

from vllm_riva_frontend.admission import (
    AdmissionLease,
    LoadShedGate,
)
from vllm_riva_frontend.config import (
    DeploymentMetadata,
    FrontendConfig,
    build_deployment_provenance,
    load_plugin_config,
    resolve_config_value,
)
from vllm_riva_frontend.grpc import (
    RIVA_SERVICE_NAME,
    build_servicer,
    register_aio_services,
)
from vllm_riva_frontend.lease import DirectLeaseOwner, SessionFactory
from vllm_riva_frontend.nim_http import (
    HttpTranscriptionConfig,
    MultipartLimits,
    NimHttpTranscriptionEndpoint,
)
from vllm_riva_frontend.nim_ws import (
    DispatchDecision,
    bootstrap_session,
    dispatch_decision,
    dispatch_nim_realtime,
)
from vllm_riva_frontend.operational import (
    OPERATIONAL_PATHS,
    operational_response,
)
from vllm_riva_frontend.session_factory import (
    load_session_factory,
    require_served_cadence,
)

_OWNED_HTTP_METHODS = {
    "/v1/realtime/transcription_sessions": frozenset({"POST"}),
    **{path: frozenset({"GET"}) for path in OPERATIONAL_PATHS},
}
#: vLLM core tip that carries endpoint-plugin lifecycle (start / ready /
#: drain / stop and ``owns_non_http_listener``). 0.33.0rc1 is the first
#: release on that line; 0.34 dev checkouts are the unreleased tip
#: (setuptools-scm guess-next-dev). Released 0.34+ is outside this range.
_VLLM_MIN = Version("0.33.0rc1")
_VLLM_RELEASE_MINOR = (0, 33)
_VLLM_DEV_MINORS = frozenset({(0, 33), (0, 34)})
_SUPPORTED_VLLM_RANGE = "0.33.x"


class PluginApplication(Protocol):
    """Host application facts used by the selected plugin."""

    routes: list[object]
    state: object


class PluginAdmission(Protocol):
    """Host-owned readiness boundary used by all inference surfaces."""

    def try_acquire(self) -> AdmissionLease | None:
        """Atomically register new work or reject after admission closes."""


class PluginContext(Protocol):
    """Structural view of the host facts this plugin reads.

    Core passes these through ``init_state`` rather than an
    application-plugin context. ``admission`` is optional: core signals
    shutdown with ``state.draining`` instead of a separate admission
    object. HTTP installation is ``attach_router``, not
    ``install_asgi_wrapper``.
    """

    plugin_name: str
    app: PluginApplication
    engine_client: object
    config: str | None
    admission: PluginAdmission | None


class OwnerToken:
    """Idempotent token for one lifecycle-visible inference owner."""

    def __init__(
        self,
        registry: CompatibilityOwnerRegistry,
        task: asyncio.Task[Any],
    ) -> None:
        """Bind one task to its process-wide compatibility registry."""
        self._registry = registry
        self._task = task
        self._released = False

    async def release(self) -> None:
        """Stop tracking the owner after its lease cleanup has settled."""
        if self._released:
            return
        async with self._registry._lock:
            if not self._released:
                self._released = True
                self._registry._owners.discard(self._task)


class CompatibilityOwnerRegistry:
    """Track the actual ASGI/gRPC tasks which own compatibility leases."""

    def __init__(self) -> None:
        """Create an empty task registry for one plugin lifetime."""
        self._owners: set[asyncio.Task[Any]] = set()
        self._lock = asyncio.Lock()

    @property
    def active(self) -> int:
        """Return the number of owner tasks still awaiting terminal cleanup."""
        return len(self._owners)

    # @spec ING-LIFE-012, ING-VEH-017
    async def register(self, kind: str | None = None) -> OwnerToken:
        """Register the current task before validation or body consumption."""
        del kind
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("compatibility owner requires an asyncio task")
        async with self._lock:
            if task in self._owners:
                raise RuntimeError(
                    "one task cannot own two compatibility sessions"
                )
            self._owners.add(task)
        return OwnerToken(self, task)

    # @spec ING-VEH-017, ING-LIFE-013, ING-LIFE-014
    async def drain(self, *, grace: float, cleanup_timeout: float) -> None:
        """Gracefully wait, then cancel and await bounded owner cleanup."""
        deadline = time.monotonic() + grace
        graceful_window = max(0.0, grace - cleanup_timeout)
        async with self._lock:
            owners = set(self._owners)
        if owners and graceful_window:
            _, owners = await asyncio.wait(owners, timeout=graceful_window)
        for owner in owners:
            owner.cancel()
        remaining = max(0.0, deadline - time.monotonic())
        if owners and remaining:
            _, owners = await asyncio.wait(owners, timeout=remaining)
        await asyncio.sleep(0)
        async with self._lock:
            unresolved = set(self._owners)
        if owners or unresolved:
            raise RuntimeError(
                "compatibility owners exceeded plugin_shutdown_grace"
            )


def _resolve_host_version() -> str:
    """Resolve the installed vLLM version without importing it at load."""
    try:
        return importlib_metadata.version("vllm")
    except importlib_metadata.PackageNotFoundError:
        module = importlib.import_module("vllm")
        version = getattr(module, "__version__", None)
        if not isinstance(version, str):
            raise RuntimeError(
                "cannot resolve installed vLLM version"
            ) from None
        return version


def _version_supported(parsed: Version) -> bool:
    """Return whether ``parsed`` is on the qualified core tip."""
    minor = parsed.release[:2]
    if parsed.is_devrelease or parsed.local is not None:
        return minor in _VLLM_DEV_MINORS
    return minor == _VLLM_RELEASE_MINOR and parsed >= _VLLM_MIN


def _validate_supported_version(version: str) -> None:
    """Fail before bind unless the host is the qualified vLLM core tip.

    Accepted: 0.33 prereleases from 0.33.0rc1 and 0.33 releases, plus
    dev or local builds whose release minor is 0.33 or 0.34 (an
    unreleased checkout of that tip). Released 0.32 and 0.34 are not.
    """
    try:
        parsed = Version(version)
    except InvalidVersion as error:
        raise ValueError(
            f"invalid installed vLLM version: {version}"
        ) from error
    if not _version_supported(parsed):
        raise ValueError(
            "vllm-riva-frontend supports vLLM "
            f"{_SUPPORTED_VLLM_RANGE} (core tip, from 0.33.0rc1, "
            f"including 0.33/0.34 dev checkouts); found {version}"
        )


def _supported_host_lines() -> str:
    """Render the qualified vLLM range the way an operator reads it."""
    return _SUPPORTED_VLLM_RANGE


# @spec ING-VEH-013
def _served_model(context: PluginContext) -> str:
    """Resolve the host registry's primary public model identity."""
    app = getattr(context, "app", None)
    state = getattr(app, "state", None)
    registry = getattr(state, "openai_serving_models", None)
    model_name = getattr(registry, "model_name", None)
    if not callable(model_name):
        raise ValueError(
            "riva_frontend requires the host served model registry"
        )
    try:
        model = model_name()
    except Exception as error:
        raise ValueError(
            "riva_frontend served model registry lookup failed"
        ) from error
    if not isinstance(model, str) or not model:
        raise ValueError(
            "riva_frontend served model registry returned an invalid name"
        )
    return model


def _supported_locales(context: PluginContext) -> frozenset[str]:
    """Derive exact accepted locales from the loaded prompt dictionary."""
    engine_config = getattr(context.engine_client, "model_config", None)
    hf_config = getattr(engine_config, "hf_config", None)
    prompts = getattr(hf_config, "prompt_dictionary", None)
    if not isinstance(prompts, dict) or not prompts:
        raise ValueError("riva_frontend requires a non-empty prompt_dictionary")
    if not all(isinstance(locale, str) and locale for locale in prompts):
        raise ValueError("prompt_dictionary locales must be non-empty strings")
    if "auto" not in prompts:
        raise ValueError(
            "prompt_dictionary must expose the default 'auto' locale"
        )
    return frozenset(prompts)


def _validate_context(context: PluginContext) -> None:
    """Validate the generic host seam and values before any listener bind."""
    if context.plugin_name != "riva_frontend":
        raise ValueError("unexpected endpoint-plugin entry-point name")
    admission = context.admission
    if admission is not None and not callable(
        getattr(admission, "try_acquire", None)
    ):
        raise TypeError("host admission must expose try_acquire")


def _validate_route_collisions(app: object) -> None:
    """Reject method/path ownership collisions before listener creation."""
    routes = getattr(app, "routes", ())
    for route in routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if path not in _OWNED_HTTP_METHODS or methods is None:
            continue
        if _OWNED_HTTP_METHODS[path].intersection(methods):
            raise ValueError(f"endpoint-plugin route collision: {path}")


async def _send_json(
    send: Callable[[dict[str, object]], Any],
    *,
    status: int,
    body: object,
) -> None:
    """Write one complete JSON response over raw ASGI."""
    encoded = json.dumps(body).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(encoded)).encode("ascii")),
            ],
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": encoded,
            "more_body": False,
        }
    )


class RivaFrontendMiddleware:
    """Own only the approved RFC-2 ASGI paths and pass everything else."""

    def __init__(self, app: object, *, lifetime: PluginLifetime) -> None:
        """Bind the wrapped host app and its one selected plugin lifetime."""
        self._app = app
        self._lifetime = lifetime
        self._ws = dispatch_nim_realtime(
            app=app,  # type: ignore[arg-type]
            factory=lifetime.session_factory,
            config=lifetime.adapter_config,
            admission=lifetime.context.admission,
            gate=lifetime.load_shed,
            owner_factory=lifetime._make_owner,
        )

    # @spec ING-NIMWS-001, ING-HTTP-001, ING-SHIM-001, ING-SHIM-006
    async def __call__(
        self,
        scope: dict[str, object],
        receive: Callable[[], Any],
        send: Callable[[dict[str, object]], Any],
    ) -> None:
        """Dispatch exact compatibility scopes without shadowing the host."""
        if scope.get("type") == "http":
            path = scope.get("path")
            method = scope.get("method")
            if path == "/v1/audio/transcriptions" and method == "POST":
                if not self._lifetime.serve_offline:
                    await self._app(scope, receive, send)  # type: ignore[operator]
                    return
                if self._lifetime.draining:
                    await _send_json(
                        send,
                        status=503,
                        body={
                            "error": {
                                "code": "service_unavailable",
                                "message": "endpoint draining",
                            }
                        },
                    )
                    return
                await self._lifetime.http_endpoint(
                    scope,
                    receive,
                    send,  # type: ignore[arg-type]
                )
                return
            if (
                path == "/v1/realtime/transcription_sessions"
                and method == "POST"
            ):
                await _send_json(
                    send,
                    status=200,
                    body=bootstrap_session(self._lifetime.model_name),
                )
                return
            if path in OPERATIONAL_PATHS and method == "GET":
                response = operational_response(
                    str(path),
                    ready=self._lifetime.ready,
                    live=self._lifetime.live,
                    release=self._lifetime.release,
                    model=self._lifetime.model_name,
                    provenance=self._lifetime.provenance,
                )
                assert response is not None
                status, body = response
                await _send_json(send, status=status, body=body)
                return
        if scope.get("type") == "websocket" and self._lifetime.draining:
            decision = dispatch_decision(scope)
            if decision is not DispatchDecision.PASS:
                await send({"type": "websocket.close", "code": 1001})
                return
        await self._ws(scope, receive, send)


class PluginLifetime:
    """One selected plugin nested inside the host engine lifetime."""

    def __init__(
        self,
        context: PluginContext,
        *,
        config_loader: Callable[
            [str | None], tuple[FrontendConfig, DeploymentMetadata]
        ] = load_plugin_config,
        version_resolver: Callable[[], str] = _resolve_host_version,
        server_factory: Callable[..., Any] = grpc.aio.server,
        service_registrar: Callable[..., Any] = register_aio_services,
        session_loader: Callable[..., SessionFactory] | None = None,
    ) -> None:
        """Capture host context and injectable startup seams."""
        self.context = context
        self._config_loader = config_loader
        self._version_resolver = version_resolver
        self._server_factory = server_factory
        self._service_registrar = service_registrar
        self._session_loader = session_loader or load_session_factory
        host_state = getattr(context, "host_state", None)
        if host_state is None:
            host_state = getattr(getattr(context, "app", None), "state", None)
        self._host_state = host_state
        self._config: FrontendConfig | None = None
        self._metadata: DeploymentMetadata | None = None
        self._server: Any | None = None
        self._server_watch: asyncio.Task[None] | None = None
        self._health: Any | None = None
        self._ready_event = asyncio.Event()
        self._failure: asyncio.Future[BaseException] | None = None
        self._entered = False
        self._quiesced = False
        self._stopped = False
        self.ready = False
        self.live = True
        self.release = "0.0.0"
        self.model_name = "unknown"
        self.locales: frozenset[str] = frozenset()
        self._cleanup_faults: list[BaseException] = []
        self.load_shed: LoadShedGate
        self.owners = CompatibilityOwnerRegistry()
        self.http_endpoint: NimHttpTranscriptionEndpoint
        self.adapter_config: object
        self.session_factory: SessionFactory

    @property
    def provenance(self) -> dict[str, str]:
        """Return only this deployment's factual public identifiers."""
        if self._metadata is None or self._config is None:
            return {}
        return build_deployment_provenance(
            self._metadata,
            resampler_identifier=self._config.resampler_identifier,
        )

    @property
    def serve_offline(self) -> bool:
        """Return whether this lifetime owns offline transcription paths."""
        if self._config is None:
            return False
        return self._config.serve_offline

    @property
    def draining(self) -> bool:
        """Return whether new inference work must be refused."""
        if self._quiesced:
            return True
        return bool(getattr(self._host_state, "draining", False))

    @property
    def shutdown_grace(self) -> float:
        """Return the configured finite owner/HTTP shutdown barrier."""
        if self._config is None:
            raise RuntimeError(
                "plugin must enter before reading shutdown grace"
            )
        return self._config.plugin_shutdown_grace

    async def _set_health(self, status: int) -> None:
        """Set aggregate and Riva-service health without leading admission."""
        assert self._health is not None
        await self._health.set("", status)
        await self._health.set(RIVA_SERVICE_NAME, status)

    def _make_owner(
        self, factory: SessionFactory, *, cleanup_timeout: float
    ) -> DirectLeaseOwner:
        """Create one owner wired to process-health escalation."""
        return DirectLeaseOwner(
            factory,
            cleanup_timeout=cleanup_timeout,
            fault_reporter=self._record_cleanup_fault,
        )

    def _record_cleanup_fault(self, error: BaseException) -> None:
        """Latch the first failure for host supervision and withdraw health."""
        self._cleanup_faults.append(error)
        self.ready = False
        self.live = False
        if self._failure is not None and not self._failure.done():
            self._failure.set_result(error)
        request_shutdown = getattr(self._host_state, "request_shutdown", None)
        if callable(request_shutdown):
            request_shutdown()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._withdraw_health_after_failure())

    async def _withdraw_health_after_failure(self) -> None:
        """Withdraw participant health; the host supervisor owns shutdown."""
        if self._health is not None:
            with suppress(BaseException):
                await self._set_health(
                    health_pb2.HealthCheckResponse.NOT_SERVING
                )

    async def _watch_server_termination(self) -> None:
        """Latch an unexpected auxiliary-listener exit for the host."""
        assert self._server is not None
        try:
            await self._server.wait_for_termination()
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            if not self._quiesced:
                self._record_cleanup_fault(error)
            return
        if not self._quiesced:
            self._record_cleanup_fault(
                RuntimeError("Riva gRPC listener terminated unexpectedly")
            )

    def _configure_http(self) -> HttpTranscriptionConfig:
        assert self._config is not None
        return HttpTranscriptionConfig(
            limits=MultipartLimits(
                encoded_audio_bytes=(
                    self._config.unary_max_encoded_audio_bytes
                ),
                envelope_bytes=(self._config.http_multipart_envelope_max_bytes),
                content_type_bytes=(self._config.http_content_type_max_bytes),
                boundary_bytes=(self._config.http_multipart_boundary_max_bytes),
                part_count=self._config.http_multipart_max_parts,
                part_header_bytes=(
                    self._config.http_multipart_max_header_bytes
                ),
                text_field_bytes=self._config.http_text_field_max_bytes,
            ),
            max_riff_header_bytes=self._config.max_riff_header_bytes,
            max_decoded_duration_seconds=(
                self._config.unary_max_decoded_duration_seconds
            ),
            pre_submit_max_samples=self._config.pre_submit_max_samples,
            request_timeout=self._config.http_request_timeout,
            finalization_timeout=(self._config.session_finalization_timeout),
            cleanup_timeout=self._config.session_cleanup_timeout,
        )

    def _install_http(self) -> None:
        """Install ASGI handling when the caller still offers that seam.

        The core path installs middleware from ``attach_router``. Tests and
        any host that still passes ``install_asgi_wrapper`` keep that call.
        """
        install = getattr(self.context, "install_asgi_wrapper", None)
        if callable(install):
            install(lambda app: RivaFrontendMiddleware(app, lifetime=self))

    # @spec ING-VEH-010, ING-VEH-012, ING-VEH-014, ING-GRPC-011
    async def prepare(self) -> None:
        """Validate configuration and the session factory before any bind."""
        self._failure = asyncio.get_running_loop().create_future()
        self._config, self._metadata = self._config_loader(self.context.config)
        try:
            self.release = importlib_metadata.version("vllm-riva-frontend")
        except importlib_metadata.PackageNotFoundError:
            self.release = "unknown"
        _validate_supported_version(self._version_resolver())
        _validate_context(self.context)
        _validate_route_collisions(self.context.app)
        self.model_name = _served_model(self.context)
        self.locales = _supported_locales(self.context)
        assert self._config is not None
        self.session_factory = self._session_loader(
            self.context.engine_client,
            name=self._config.session_factory,
        )
        require_served_cadence(
            self.session_factory, self._config.streaming_cadence
        )
        self.load_shed = LoadShedGate(
            self._config.load_shed_max_sessions,
            owner_register=self.owners.register,
        )
        self.adapter_config = SimpleNamespace(
            **vars(self._config),
            model_name=self.model_name,
            locales=self.locales,
            provenance=self.provenance,
        )
        self.http_endpoint = NimHttpTranscriptionEndpoint(
            factory=self.session_factory,
            load_shed=self.load_shed,
            config=self._configure_http(),
            served_model=self.model_name,
            locales=self.locales,
            admission=self.context.admission,
            owner_factory=self._make_owner,
        )
        self._install_http()

    async def start_listener(self) -> None:
        """Bind the gRPC listener and publish not-serving until mark_serving."""
        assert self._config is not None
        options: list[tuple[str, int]] = [
            (
                "grpc.max_receive_message_length",
                self._config.grpc_receive_max_bytes,
            )
        ]
        if self._config.grpc_keepalive_seconds is not None:
            options.append(
                (
                    "grpc.keepalive_time_ms",
                    int(self._config.grpc_keepalive_seconds * 1000),
                )
            )
        self._server = self._server_factory(options=options)
        servicer = build_servicer(
            factory=self.session_factory,
            config=self._config,
            model_name=self.model_name,
            locales=self.locales,
            load_shed=self.load_shed,
            admission=self.context.admission,
            owner_factory=self._make_owner,
            is_draining=self.draining_now,
        )
        try:
            self._health = self._service_registrar(
                server=self._server, servicer=servicer
            )
            await self._set_health(health_pb2.HealthCheckResponse.NOT_SERVING)
            if self._server.add_insecure_port(self._config.grpc_bind) == 0:
                raise RuntimeError(
                    f"could not bind gRPC listener: {self._config.grpc_bind}"
                )
            await self._server.start()
        except BaseException:
            await self.stop()
            raise
        self._entered = True
        self._ready_event.set()
        self._server_watch = asyncio.create_task(
            self._watch_server_termination()
        )

    def draining_now(self) -> bool:
        """Return the draining predicate the gRPC servicer polls."""
        return self.draining

    # @spec ING-VEH-010, ING-VEH-012, ING-VEH-014, ING-GRPC-011
    async def __aenter__(self) -> PluginLifetime:
        """Validate, install ASGI handling, and bind gRPC not-serving."""
        await self.prepare()
        await self.start_listener()
        return self

    # @spec ING-VEH-018, ING-GRPC-011
    async def wait_ready(self) -> None:
        """Attest that every owned listener is bound and ready for admission."""
        if not self._entered:
            raise RuntimeError("plugin must enter before wait_ready")
        await self._ready_event.wait()

    # @spec ING-VEH-017, ING-VEH-018, ING-GRPC-011
    async def wait_failed(self) -> None:
        """Raise the first latched post-entry participant failure."""
        if not self._entered or self._failure is None:
            raise RuntimeError("plugin must enter before wait_failed")
        error = await asyncio.shield(self._failure)
        raise error

    # @spec ING-VEH-016, ING-GRPC-011, ING-SHIM-001
    async def mark_serving(self) -> None:
        """Publish compatibility readiness after the host transition."""
        if not self._entered or self._health is None:
            raise RuntimeError("plugin must enter before mark_serving")
        await self._set_health(health_pb2.HealthCheckResponse.SERVING)
        self.ready = True

    # @spec ING-VEH-017, ING-GRPC-011, ING-SHIM-001
    async def quiesce_and_drain(self, *, grace: float | None = None) -> None:
        """Become non-ready, drain every owner, then stop gRPC.

        ``grace`` is the host's remaining shutdown budget. Core passes
        that as ``drain(state, timeout)``. When omitted, the configured
        ``plugin_shutdown_grace`` is the budget. Cancellation is not
        swallowed: a cancelled drain stops the listener and reraises.
        """
        if self._quiesced:
            return
        self._quiesced = True
        self.ready = False
        assert self._config is not None
        budget = self._config.plugin_shutdown_grace if grace is None else grace
        try:
            if self._health is not None:
                await self._set_health(
                    health_pb2.HealthCheckResponse.NOT_SERVING
                )
            await self.load_shed.close()
            if budget > 0:
                await self.owners.drain(
                    grace=budget,
                    cleanup_timeout=self._config.session_cleanup_timeout,
                )
            if self._cleanup_faults:
                raise RuntimeError(
                    "terminal cleanup fault requires provider teardown"
                ) from self._cleanup_faults[0]
        except asyncio.CancelledError:
            self.live = False
            raise
        except BaseException:
            self.live = False
            raise
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the gRPC listener without waiting for in-flight owners.

        Idempotent. Drain waits for owners; this only closes the port
        and the termination watch, and returns promptly.
        """
        if self._stopped:
            return
        self._stopped = True
        server = self._server
        watch = self._server_watch
        self._server = None
        self._server_watch = None
        if server is not None:
            with suppress(BaseException):
                await server.stop(0)
        if watch is not None:
            watch.cancel()
            with suppress(BaseException):
                await watch

    async def __aexit__(
        self, exc_type: object, exc: object, traceback: object
    ) -> None:
        """Idempotently stop plugin resources before engine-context exit."""
        del exc_type, exc, traceback
        if self._entered and not self._quiesced:
            await self.quiesce_and_drain()


class _EndpointContext:
    """Host facts gathered from ``init_state`` for one plugin lifetime."""

    def __init__(
        self,
        *,
        engine_client: object,
        app: object,
        config: str | None,
        admission: object | None,
        host_state: object,
    ) -> None:
        """Record the core server objects the lifetime reads."""
        self.plugin_name = "riva_frontend"
        self.engine_client = engine_client
        self.app = app
        self.config = config
        self.admission = admission
        self.host_state = host_state


class _DelegatingMiddleware:
    """Forward ASGI calls to the lifetime once ``prepare`` has finished."""

    def __init__(self, app: object, *, plugin: RivaFrontendPlugin) -> None:
        """Retain the wrapped app and the plugin that owns the lifetime."""
        self._app = app
        self._plugin = plugin
        self._bound: RivaFrontendMiddleware | None = None
        self._bound_lifetime: PluginLifetime | None = None

    async def __call__(
        self,
        scope: dict[str, object],
        receive: Callable[[], Any],
        send: Callable[[dict[str, object]], Any],
    ) -> None:
        """Dispatch owned routes, or pass through before the lifetime exists."""
        lifetime = self._plugin.lifetime
        if lifetime is None or lifetime._config is None:
            await self._app(scope, receive, send)  # type: ignore[operator]
            return
        if self._bound is None or self._bound_lifetime is not lifetime:
            self._bound = RivaFrontendMiddleware(self._app, lifetime=lifetime)
            self._bound_lifetime = lifetime
        await self._bound(scope, receive, send)


class RivaFrontendPlugin:
    """``vllm.endpoint_plugins`` factory for the Riva/NIM frontend.

    The entry point is this class. vLLM calls it with no arguments, then
    ``attach_router``, ``init_state``, ``start``, ``ready``, ``drain``,
    and ``stop``. ``owns_non_http_listener`` is set because the plugin
    binds its own gRPC port (default ``0.0.0.0:50051``).
    """

    name = "riva_frontend"
    required_tasks: tuple[str, ...] | None = None
    owns_non_http_listener = True

    def __init__(
        self,
        *,
        version_resolver: Callable[[], str] | None = None,
        server_factory: Callable[..., Any] | None = None,
        service_registrar: Callable[..., Any] | None = None,
        session_loader: Callable[..., SessionFactory] | None = None,
    ) -> None:
        """Create one plugin. Optional seams exist so tests need no GPU."""
        self._app: object | None = None
        self.lifetime: PluginLifetime | None = None
        self._version_resolver = version_resolver
        self._server_factory = server_factory
        self._service_registrar = service_registrar
        self._session_loader = session_loader

    def attach_router(self, app: object) -> None:
        """Install middleware that claims only this plugin's HTTP paths."""
        add_middleware = getattr(app, "add_middleware", None)
        if not callable(add_middleware):
            raise TypeError("endpoint app must expose add_middleware")
        self._app = app
        add_middleware(_DelegatingMiddleware, plugin=self)

    async def init_state(
        self, engine_client: object, state: object, args: object
    ) -> None:
        """Resolve config and the session factory. Do not bind yet."""
        if self._app is None:
            raise RuntimeError("attach_router must run before init_state")
        admission = getattr(state, "admission", None)
        context = _EndpointContext(
            engine_client=engine_client,
            app=self._app,
            config=resolve_config_value(args),
            admission=admission,
            host_state=state,
        )
        kwargs: dict[str, Any] = {}
        if self._version_resolver is not None:
            kwargs["version_resolver"] = self._version_resolver
        if self._server_factory is not None:
            kwargs["server_factory"] = self._server_factory
        if self._service_registrar is not None:
            kwargs["service_registrar"] = self._service_registrar
        if self._session_loader is not None:
            kwargs["session_loader"] = self._session_loader
        lifetime = PluginLifetime(context, **kwargs)  # type: ignore[arg-type]
        await lifetime.prepare()
        self.lifetime = lifetime

    async def start(self, state: object) -> None:
        """Bind gRPC and publish serving. Clean up if startup fails."""
        del state
        if self.lifetime is None:
            raise RuntimeError("endpoint plugin must init_state before start")
        try:
            await self.lifetime.start_listener()
            await self.lifetime.mark_serving()
        except BaseException:
            await self.lifetime.stop()
            raise

    def ready(self, state: object) -> bool:
        """Report readiness for ``/health``. False while draining."""
        del state
        lifetime = self.lifetime
        if lifetime is None:
            return False
        return lifetime.ready and not lifetime.draining

    async def drain(self, state: object, timeout: float) -> None:
        """Refuse new work and drain owners within the host's budget."""
        del state
        if self.lifetime is None:
            return
        await self.lifetime.quiesce_and_drain(grace=timeout)

    async def stop(self, state: object) -> None:
        """Close the gRPC listener without waiting for owners."""
        del state
        if self.lifetime is None:
            return
        await self.lifetime.stop()
