"""Application-plugin installation, readiness, and owner drainage."""

from __future__ import annotations

import asyncio
import importlib
import json
import time
from collections.abc import Callable
from contextlib import suppress
from importlib import metadata as importlib_metadata
from types import SimpleNamespace
from typing import Any, Protocol, Self, cast

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
    bootstrap_session,
    dispatch_nim_realtime,
)
from vllm_riva_frontend.operational import (
    OPERATIONAL_PATHS,
    operational_response,
)

_OWNED_HTTP_METHODS = {
    "/v1/audio/transcriptions": frozenset({"POST"}),
    "/v1/realtime/transcription_sessions": frozenset({"POST"}),
    **{path: frozenset({"GET"}) for path in OPERATIONAL_PATHS},
}
_SUPPORTED_HOST_MAJOR_MINOR = (0, 24)


class PluginApplication(Protocol):
    """Host application facts used by the selected plugin."""

    routes: list[object]
    state: object


class PluginAdmission(Protocol):
    """Host-owned readiness boundary used by all inference surfaces."""

    def try_acquire(self) -> AdmissionLease | None:
        """Atomically register new work or reject after admission closes."""


class PluginContext(Protocol):
    """Structural view of the generic vLLM-Omni plugin context."""

    plugin_name: str
    app: PluginApplication
    engine_client: object
    config: str | None
    admission: PluginAdmission
    install_asgi_wrapper: Callable[[Callable[[Any], Any]], None]


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
    """Resolve the installed vLLM-Omni version without depending on it."""
    try:
        return importlib_metadata.version("vllm-omni")
    except importlib_metadata.PackageNotFoundError:
        module = importlib.import_module("vllm_omni")
        version = getattr(module, "__version__", None)
        if not isinstance(version, str):
            raise RuntimeError(
                "cannot resolve installed vLLM-Omni version"
            ) from None
        return version


def _validate_supported_version(version: str) -> None:
    """Fail before bind unless the host is the declared 0.24 line.

    A build carrying a PEP 440 dev, pre-release, or local segment — an
    editable setuptools-scm '0.1.dev2155+g...' checkout, or a
    '0.24.0.dev0+cu130' nightly — is an intentional unreleased host and is
    allowed through; its base version is not a reliable line indicator.
    Only a clean release version is held to the exact supported major.minor.
    """
    try:
        parsed = Version(version)
    except InvalidVersion as error:
        raise ValueError(
            f"invalid installed vLLM-Omni version: {version}"
        ) from error
    if parsed.is_devrelease or parsed.is_prerelease or parsed.local is not None:
        return
    if parsed.release[:2] != _SUPPORTED_HOST_MAJOR_MINOR:
        raise ValueError(
            f"vllm-riva-frontend supports vLLM-Omni 0.24.x; found {version}"
        )


# @spec ING-VEH-012, ING-VEH-014, PORT-RTC-003
def _create_nemotron_session_factory(
    engine_client: object,
    *,
    app_state: object = None,
) -> SessionFactory:
    """Acquire the model-specific session factory from the selected host.

    Observation inheritance (host PORT-OBS-003): when the host has a
    streaming observer installed in its serving app state, this plugin
    is the "model-aware downstream participant" the host's factory
    docstring names — it wires observation by constructing the internal
    ``NemotronSessionFactory(engine=..., observer=...)`` with the HOST'S
    installed observer, resolved via the host's own
    ``resolve_installed_observer`` seam. The plugin never constructs,
    registers, or duplicates any metric family itself (ENV-MOD-004);
    it only reuses the one observer the host installed. On a host
    without the observation seam, or with no observer installed, the
    pinned public constructor is used unchanged and sessions are simply
    unobserved — exactly the pre-observability behavior.
    """
    module = importlib.import_module("vllm_omni.entrypoints.nemotron_session")

    observer: object = None
    if app_state is not None:
        try:
            install_module = importlib.import_module(
                "vllm_omni.metrics.streaming_install"
            )
        except ModuleNotFoundError:
            install_module = None
        if install_module is not None:
            resolve = getattr(
                install_module, "resolve_installed_observer", None
            )
            if callable(resolve):
                observer = resolve(app_state)

    if observer is not None:
        internal = getattr(module, "NemotronSessionFactory", None)
        if callable(internal):
            factory = internal(engine=engine_client, observer=observer)
            if not callable(getattr(factory, "open", None)):
                raise TypeError(
                    "Nemotron session factory must expose async open"
                )
            return cast(SessionFactory, factory)

    constructor = getattr(module, "create_nemotron_session_factory", None)
    if not callable(constructor):
        raise TypeError(
            "vLLM-Omni does not expose create_nemotron_session_factory"
        )
    factory = constructor(engine_client)
    if not callable(getattr(factory, "open", None)):
        raise TypeError("Nemotron session factory must expose async open")
    return cast(SessionFactory, factory)


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
        raise ValueError("unexpected application-plugin entry-point name")
    if not callable(getattr(context, "install_asgi_wrapper", None)):
        raise TypeError("plugin context must expose install_asgi_wrapper")
    if not callable(getattr(context.admission, "try_acquire", None)):
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
            raise ValueError(f"application-plugin route collision: {path}")


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
    ) -> None:
        """Capture host context and injectable startup seams."""
        self.context = context
        self._config_loader = config_loader
        self._version_resolver = version_resolver
        self._server_factory = server_factory
        self._service_registrar = service_registrar
        self._config: FrontendConfig | None = None
        self._metadata: DeploymentMetadata | None = None
        self._server: Any | None = None
        self._server_watch: asyncio.Task[None] | None = None
        self._health: Any | None = None
        self._ready_event = asyncio.Event()
        self._failure: asyncio.Future[BaseException] | None = None
        self._entered = False
        self._quiesced = False
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

    # @spec ING-VEH-010, ING-VEH-012, ING-VEH-014, ING-GRPC-011
    async def __aenter__(self) -> Self:
        """Validate, install ASGI handling, and bind gRPC not-serving."""
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
        self.session_factory = _create_nemotron_session_factory(
            self.context.engine_client,
            app_state=getattr(
                getattr(self.context, "app", None), "state", None
            ),
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
        self.context.install_asgi_wrapper(
            lambda app: RivaFrontendMiddleware(app, lifetime=self)
        )
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
            if self._server is not None:
                with suppress(BaseException):
                    await self._server.stop(0)
            raise
        self._entered = True
        self._ready_event.set()
        self._server_watch = asyncio.create_task(
            self._watch_server_termination()
        )
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
    async def quiesce_and_drain(self) -> None:
        """Become non-ready, drain every owner, then stop gRPC."""
        if self._quiesced:
            return
        self._quiesced = True
        self.ready = False
        assert self._config is not None
        try:
            if self._health is not None:
                await self._set_health(
                    health_pb2.HealthCheckResponse.NOT_SERVING
                )
            await self.load_shed.close()
            await self.owners.drain(
                grace=self._config.plugin_shutdown_grace,
                cleanup_timeout=self._config.session_cleanup_timeout,
            )
            if self._cleanup_faults:
                raise RuntimeError(
                    "terminal cleanup fault requires provider teardown"
                ) from self._cleanup_faults[0]
        except BaseException:
            self.live = False
            raise
        finally:
            if self._server is not None:
                await self._server.stop(0)
            if self._server_watch is not None:
                with suppress(asyncio.CancelledError):
                    await self._server_watch

    async def __aexit__(
        self, exc_type: object, exc: object, traceback: object
    ) -> None:
        """Idempotently stop plugin resources before engine-context exit."""
        del exc_type, exc, traceback
        if self._entered and not self._quiesced:
            await self.quiesce_and_drain()
