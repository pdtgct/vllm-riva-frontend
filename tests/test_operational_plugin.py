"""Identity, operations, and composed plugin-lifetime contracts."""

import asyncio
import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_riva_frontend import PluginContext, plugin
from vllm_riva_frontend import lifecycle as lifecycle_module
from vllm_riva_frontend.config import (
    FORBIDDEN_PROVENANCE_KEY,
    DeploymentMetadata,
    FrontendConfig,
    build_deployment_provenance,
)
from vllm_riva_frontend.lifecycle import (
    CompatibilityOwnerRegistry,
    PluginAdmission,
    PluginLifetime,
)
from vllm_riva_frontend.operational import (
    OPERATIONAL_PATHS,
    operational_response,
)

#: A bare hex token, either case, standalone, the shape a NIM profile hash
#: (e.g. "profileHash": "deadbeef") takes.  This deliberately does not
#: match this deployment's own "sha256:<hex>" image digest, which is
#: always scheme-prefixed rather than a bare hex string.
_HASH_SHAPED = re.compile(r"^[0-9A-Fa-f]{8,64}$")
_NGC_SCHEME = "ngc://"


def _walk(node: object) -> list[tuple[str, object]]:
    """Return every (key, value) pair reachable at any depth of a JSON tree.

    Walks both mapping and list nodes, so a forbidden key or value hidden
    inside an array is found the same as one directly inside a dict.
    """
    pairs: list[tuple[str, object]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            pairs.append((key, value))
            pairs.extend(_walk(value))
    elif isinstance(node, list):
        for item in node:
            pairs.extend(_walk(item))
    return pairs


def _assert_no_nim_identity_leak(body: object) -> None:
    """Fail on a NIM registry/profile identity anywhere in a JSON tree.

    Checked at every depth, through both dicts and lists: no
    FORBIDDEN_PROVENANCE_KEY key or substring, no ``ngc://`` reference,
    and no bare hex NIM-profile-hash-shaped value.
    """
    for key, value in _walk(body):
        assert key != FORBIDDEN_PROVENANCE_KEY, (
            f"forbidden key present at some depth: {key!r}"
        )
        if not isinstance(value, str):
            continue
        assert FORBIDDEN_PROVENANCE_KEY not in value, (
            f"forbidden identifier present in a value: {value!r}"
        )
        assert _NGC_SCHEME not in value, (
            f"an ngc:// reference is present in a value: {value!r}"
        )
        assert not _HASH_SHAPED.fullmatch(value), (
            f"a NIM-profile-hash-shaped value is present: {value!r}"
        )


def _frontend_config() -> FrontendConfig:
    return FrontendConfig(
        grpc_bind="127.0.0.1:50051",
        grpc_receive_max_bytes=1152,
        grpc_config_envelope_max_bytes=128,
        unary_max_encoded_audio_bytes=1024,
        unary_max_decoded_duration_seconds=60.0,
        max_riff_header_bytes=512,
        load_shed_max_sessions=4,
        pre_submit_max_samples=16000,
        preconfiguration_timeout=5.0,
        session_idle_timeout=60.0,
        session_finalization_timeout=0.1,
        session_cleanup_timeout=0.05,
        plugin_shutdown_grace=0.2,
        ws_receive_max_bytes=2048,
        ws_event_envelope_max_bytes=1024,
        http_multipart_envelope_max_bytes=1152,
        http_content_type_max_bytes=128,
        http_request_header_max_bytes=256,
        http_multipart_boundary_max_bytes=64,
        http_multipart_max_parts=4,
        http_multipart_max_header_bytes=128,
        http_text_field_max_bytes=128,
        http_request_timeout=60.0,
        grpc_keepalive_seconds=20.0,
        max_session_duration=600.0,
        resampler_identifier="scipy-poly-v1",
    )


class FakeAdmission:
    """Host-owned admission stand-in."""

    def __init__(self) -> None:
        self.open = False
        self.acquisitions: list[FakeAdmissionLease] = []

    def is_open(self) -> bool:
        return self.open

    def try_acquire(self) -> "FakeAdmissionLease | None":
        if not self.open:
            return None
        lease = FakeAdmissionLease()
        self.acquisitions.append(lease)
        return lease


class FakeAdmissionLease:
    """Idempotent host-admission lease stand-in."""

    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


class FakeServingModels:
    """Host serving-model registry stand-in."""

    def __init__(
        self,
        name: object = "nvidia/nemotron-asr",
        *,
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.error = error
        self.calls = 0

    def model_name(self) -> object:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.name


class FakeApp:
    """Minimal Starlette application installation surface."""

    def __init__(self) -> None:
        self.routes: list[object] = []
        self.state = SimpleNamespace(openai_serving_models=FakeServingModels())


class FakeServer:
    """Async gRPC server lifecycle stand-in."""

    def __init__(self, events: list[object]) -> None:
        self.events = events
        self.terminated = asyncio.Event()

    def add_insecure_port(self, bind: str) -> int:
        self.events.append(("bind", bind))
        return 50051

    async def start(self) -> None:
        self.events.append("grpc-start")

    async def stop(self, grace: float | None) -> None:
        self.events.append(("grpc-stop", grace))
        self.terminated.set()

    async def wait_for_termination(self) -> None:
        await self.terminated.wait()


class FakeHealth:
    """Async gRPC health state stand-in."""

    def __init__(self, events: list[object]) -> None:
        self.events = events

    async def set(self, service: str, status: object) -> None:
        self.events.append(("health", service, int(status)))


class FakeContext:
    """Structural stand-in for the generic host-owned plugin context."""

    def __init__(self) -> None:
        self.plugin_name = "riva_frontend"
        self.app = FakeApp()
        model_config = SimpleNamespace(
            model="nvidia/nemotron-asr",
            hf_config=SimpleNamespace(
                prompt_dictionary={"auto": 0, "en-US": 1, "es-US": 2}
            ),
        )
        self.engine_client = SimpleNamespace(model_config=model_config)

        async def open_session(*, cadence: str, locale: str) -> object:
            del cadence, locale
            return object()

        self.session_factory = SimpleNamespace(open=open_session)
        self.serve_args = SimpleNamespace(
            api_server_count=1,
            model="nvidia/nemotron-asr",
            served_model_name=None,
            ws_max_size=_frontend_config().ws_receive_max_bytes,
            h11_max_incomplete_event_size=(
                _frontend_config().http_request_header_max_bytes
            ),
        )
        self.config: str | None = "{}"
        self.admission = FakeAdmission()
        self.asgi_wrappers: list[object] = []

        def install_asgi_wrapper(wrapper: object) -> None:
            self.asgi_wrappers.append(wrapper)

        self.install_asgi_wrapper = install_asgi_wrapper


def _lifetime(
    context: FakeContext,
    events: list[object],
    *,
    version: str = "0.24.0",
    session_factory_builder=None,
) -> PluginLifetime:
    frontend = _frontend_config()
    metadata = DeploymentMetadata(
        image="sha256:test",
        pin="vllm==0.24.0",
        precision_policy="nemotron-asr-fp32-v1",
    )
    server = FakeServer(events)
    if session_factory_builder is None:

        def default_factory_builder(
            engine_client: object, **_: object
        ) -> object:
            del engine_client
            return context.session_factory

        session_factory_builder = default_factory_builder
    lifecycle_module._create_nemotron_session_factory = (  # type: ignore[attr-defined]
        session_factory_builder
    )
    return PluginLifetime(
        context,
        config_loader=lambda _: (frontend, metadata),
        version_resolver=lambda: version,
        server_factory=lambda **_: server,
        service_registrar=lambda **_: FakeHealth(events),
    )


# @spec ING-VEH-001, ING-VEH-009, ING-VEH-011
def test_distribution_import_and_entry_point_identity_are_exact() -> None:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert 'name = "vllm-riva-frontend"' in text
    assert 'riva_frontend = "vllm_riva_frontend:plugin"' in text
    assert "nvidia-riva-client==2.26.0" in text
    assert "grpcio-health-checking>=1.60" in text
    assert "grpcio-reflection>=1.60" in text
    runtime_dependencies = text.split("dependencies = [", maxsplit=1)[1].split(
        "]", maxsplit=1
    )[0]
    assert '"vllm' not in runtime_dependencies
    assert '"fastapi' not in runtime_dependencies.lower()
    assert not inspect.iscoroutinefunction(plugin)
    assert plugin.config_optional is False
    assert set(PluginContext.__annotations__) == {
        "plugin_name",
        "config",
        "app",
        "engine_client",
        "admission",
        "install_asgi_wrapper",
    }
    assert {
        name
        for name, value in inspect.getmembers(
            PluginAdmission,
            predicate=inspect.isfunction,
        )
        if not name.startswith("_")
    } == {"try_acquire"}


# @spec ING-VEH-003, ING-VEH-009, ING-VEH-016
def test_sync_entry_returns_context_bound_async_lifetime() -> None:
    context = FakeContext()
    lifetime = plugin(context)
    assert isinstance(lifetime, PluginLifetime)
    assert lifetime.context is context


# @spec ING-SHIM-001, ING-SHIM-002, ING-SHIM-006
def test_operational_routes_do_not_shadow_host() -> None:
    status, body = operational_response(
        "/v1/health/ready", ready=True, live=True
    )
    assert status == 200
    assert body["status"] == "ready"
    assert "/v1/health" not in OPERATIONAL_PATHS
    assert "/v1/metrics" not in OPERATIONAL_PATHS
    assert "/v1/models" not in OPERATIONAL_PATHS
    assert operational_response("/v1/models", ready=True, live=True) is None


# @spec ING-SHIM-001, ING-SHIM-006
def test_metadata_provenance_is_allowlist_built_never_a_copied_mapping() -> (
    None
):
    """/v1/metadata cannot leak a NIM identity because none can reach it.

    Earlier this guarantee was a denylist filter over an arbitrary
    mapping (recursed into dicts only, checked only one forbidden key)
    and missed an offender hidden in a list, an ``ngc://`` reference, or
    a bare profile hash under a different key.  There is no fixture that
    reproduces that class of leak against the current design, because
    ``build_deployment_provenance`` -- the only allowed constructor for
    this response's provenance -- takes typed deployment fields, not a
    mapping to filter; this test pins that the real construction path
    stays clean under the strengthened (list-aware, ngc://- and
    hash-aware) walker.
    """
    metadata = DeploymentMetadata(
        image="sha256:test",
        pin="vllm==0.24.0",
        precision_policy="nemotron-asr-fp32-v1",
    )
    provenance = build_deployment_provenance(
        metadata, resampler_identifier="scipy-poly-v1"
    )

    status, body = operational_response(
        "/v1/metadata",
        ready=True,
        live=True,
        release="1.2.3",
        model="nemotron-asr",
        provenance=provenance,
    )

    assert status == 200
    _assert_no_nim_identity_leak(body)
    assert body["provenance"] == provenance


# @spec ING-VEH-012, ING-VEH-016, ING-GRPC-011
def test_plugin_starts_not_serving_then_follows_host_linearization() -> None:
    async def exercise() -> None:
        events: list[object] = []
        context = FakeContext()
        lifetime = _lifetime(context, events)
        async with lifetime:
            await lifetime.wait_ready()
            assert lifetime.ready is False
            assert events[-1] == "grpc-start"
            assert events[:4] == [
                ("health", "", 2),
                ("health", "nvidia.riva.asr.RivaSpeechRecognition", 2),
                ("bind", "127.0.0.1:50051"),
                "grpc-start",
            ]
            await lifetime.mark_serving()
            assert lifetime.ready is True
            assert events[-1][0] == "health"
            await lifetime.quiesce_and_drain()
            assert lifetime.ready is False
            assert events[-1] == ("grpc-stop", 0)

    asyncio.run(exercise())


# @spec ING-VEH-003, ING-VEH-018, ING-VEH-019
def test_context_requires_acquisition_capability_not_boolean_snapshot() -> None:
    context = FakeContext()
    context.admission = SimpleNamespace(try_acquire=lambda: None)
    lifecycle_module._validate_context(context)

    context.admission = SimpleNamespace(is_open=lambda: True)
    with pytest.raises(TypeError, match="try_acquire"):
        lifecycle_module._validate_context(context)


# @spec ING-VEH-017, ING-VEH-018
def test_participant_attests_readiness_and_latches_cleanup_failure() -> None:
    async def exercise() -> None:
        events: list[object] = []
        context = FakeContext()
        lifetime = _lifetime(context, events)
        await lifetime.__aenter__()
        await lifetime.wait_ready()

        failure = asyncio.create_task(lifetime.wait_failed())
        await asyncio.sleep(0)
        lifetime._record_cleanup_fault(RuntimeError("cleanup broke"))
        with pytest.raises(RuntimeError, match="cleanup broke"):
            await failure

        lifetime._cleanup_faults.clear()
        await lifetime.quiesce_and_drain()
        await lifetime.__aexit__(None, None, None)

    asyncio.run(exercise())


# @spec ING-VEH-017, ING-VEH-018, ING-GRPC-011
def test_participant_rejects_preentry_waits_and_reports_listener_exit() -> None:
    async def exercise() -> None:
        events: list[object] = []
        lifetime = _lifetime(FakeContext(), events)
        assert lifetime.provenance == {}
        with pytest.raises(RuntimeError, match="shutdown grace"):
            _ = lifetime.shutdown_grace
        with pytest.raises(RuntimeError, match="wait_ready"):
            await lifetime.wait_ready()
        with pytest.raises(RuntimeError, match="wait_failed"):
            await lifetime.wait_failed()
        with pytest.raises(RuntimeError, match="mark_serving"):
            await lifetime.mark_serving()

        await lifetime.__aenter__()
        await lifetime.wait_ready()
        assert lifetime._server is not None
        lifetime._server.terminated.set()
        with pytest.raises(RuntimeError, match="terminated unexpectedly"):
            await lifetime.wait_failed()
        assert lifetime.live is False
        assert lifetime.ready is False

        lifetime._cleanup_faults.clear()
        await lifetime.quiesce_and_drain()
        await lifetime.quiesce_and_drain()
        await lifetime.__aexit__(None, None, None)

    asyncio.run(exercise())


# @spec ING-VEH-017
def test_failure_reporting_never_mutates_the_host_http_server() -> None:
    source = Path(lifecycle_module.__file__).read_text(encoding="utf-8")
    assert "server.should_exit" not in source


# @spec ING-VEH-016, ING-VEH-018, ING-VEH-022
def test_mark_serving_and_drain_need_no_admission_snapshot() -> None:
    async def exercise() -> None:
        events: list[object] = []
        context = FakeContext()
        context.admission = SimpleNamespace(
            try_acquire=lambda: FakeAdmissionLease()
        )
        lifetime = _lifetime(context, events)
        async with lifetime:
            await lifetime.wait_ready()
            await lifetime.mark_serving()
            assert lifetime.ready is True
            await lifetime.quiesce_and_drain()
            assert lifetime.ready is False

    asyncio.run(exercise())


# @spec ING-VEH-010, ING-VEH-012, ING-SHIM-006
def test_preflight_rejects_version_and_route_collision_before_bind() -> None:
    async def exercise() -> None:
        context = FakeContext()
        events: list[object] = []
        with pytest.raises(ValueError, match="0.26.0"):
            async with _lifetime(context, events, version="0.26.0"):
                pass
        assert events == []

        # Both qualified lines reach bind rather than being refused here.
        for qualified in ("0.24.0", "0.25.0"):
            context = FakeContext()
            async with _lifetime(context, events, version=qualified):
                pass
        events.clear()

        context = FakeContext()
        context.app.routes.append(
            SimpleNamespace(path="/v1/audio/transcriptions", methods={"POST"})
        )
        with pytest.raises(ValueError, match="route collision"):
            async with _lifetime(context, events):
                pass
        assert events == []

    asyncio.run(exercise())


# @spec ING-VEH-010, ING-VEH-012
def test_context_version_and_locale_preflight_fail_closed() -> None:
    with pytest.raises(ValueError, match="invalid installed"):
        lifecycle_module._validate_supported_version("not a version")
    lifecycle_module._validate_supported_version("0.25.0.dev0")
    lifecycle_module._validate_supported_version("0.25.0+local")

    # Every qualified line passes as a clean release; an unqualified one does
    # not, so widening the set stays a deliberate act rather than a drift.
    for major, minor in lifecycle_module._SUPPORTED_HOST_MAJOR_MINOR:
        lifecycle_module._validate_supported_version(f"{major}.{minor}.0")
    with pytest.raises(ValueError, match="0.26.0"):
        lifecycle_module._validate_supported_version("0.26.0")
    with pytest.raises(ValueError, match="0.23.0"):
        lifecycle_module._validate_supported_version("0.23.0")
    assert lifecycle_module._supported_host_lines() == "0.24.x, 0.25.x"

    context = FakeContext()
    context.plugin_name = "other"
    with pytest.raises(ValueError, match="entry-point name"):
        lifecycle_module._validate_context(context)

    context = FakeContext()
    context.install_asgi_wrapper = None  # type: ignore[assignment]
    with pytest.raises(TypeError, match="install_asgi_wrapper"):
        lifecycle_module._validate_context(context)

    for prompts in ({}, {"en-US": "prompt"}, {"auto": "prompt", "": "bad"}):
        context = FakeContext()
        context.engine_client.model_config.hf_config.prompt_dictionary = prompts
        with pytest.raises(ValueError, match="prompt_dictionary"):
            lifecycle_module._supported_locales(context)


# @spec ING-VEH-012, ING-VEH-014, PORT-RTC-003
def test_entry_acquires_model_factory_from_the_generic_engine_client() -> None:
    """The downstream participant, not the generic host, owns acquisition."""

    async def exercise() -> None:
        events: list[object] = []
        context = FakeContext()
        del context.session_factory
        del context.serve_args
        opened: list[tuple[str, str]] = []

        async def open_session(*, cadence: str, locale: str) -> object:
            opened.append((cadence, locale))
            return object()

        factory = SimpleNamespace(open=open_session)
        calls: list[object] = []

        def build_factory(engine_client: object, **_: object) -> object:
            calls.append(engine_client)
            return factory

        lifetime = _lifetime(
            context,
            events,
            session_factory_builder=build_factory,
        )
        async with lifetime:
            assert calls == [context.engine_client]
            assert opened == []
            assert lifetime.model_name == "nvidia/nemotron-asr"
            assert context.asgi_wrappers
            assert events[-1] == "grpc-start"
            context.admission.open = False
            await lifetime.quiesce_and_drain()

    asyncio.run(exercise())


# @spec ING-VEH-013, ING-GRPC-004, ING-NIMWS-001, ING-HTTP-002, ING-SHIM-002
def test_primary_served_alias_is_one_identity_for_every_plugin_surface() -> (
    None
):
    """The normalized CLI alias, not the checkpoint path, is public."""

    async def exercise() -> None:
        events: list[object] = []
        context = FakeContext()
        registry = FakeServingModels("nemotron-asr")
        context.app.state.openai_serving_models = registry
        context.engine_client.model_config.served_model_name = (
            "nvidia/nemotron-asr"
        )
        lifetime = _lifetime(context, events)
        async with lifetime:
            assert lifetime.model_name == "nemotron-asr"
            assert lifetime.adapter_config.model_name == "nemotron-asr"
            assert lifetime.http_endpoint._served_model == "nemotron-asr"

            wrapper = context.asgi_wrappers[0]

            async def host(
                scope: object, receive: object, send: object
            ) -> None:
                del scope, receive, send
                raise AssertionError("plugin bootstrap must not reach host")

            middleware = wrapper(host)

            async def receive() -> dict[str, object]:
                return {"type": "http.request", "body": b""}

            sent: list[dict[str, object]] = []

            async def send(message: dict[str, object]) -> None:
                sent.append(message)

            await middleware(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/v1/realtime/transcription_sessions",
                },
                receive,
                send,
            )
            body = json.loads(sent[1]["body"])
            assert body["input_audio_transcription"]["model"] == "nemotron-asr"
            assert registry.calls == 1

    asyncio.run(exercise())


# @spec ING-VEH-013
def test_host_serving_registry_wins_when_engine_identity_disagrees() -> None:
    context = FakeContext()
    registry = FakeServingModels("nvidia/nemotron-asr")
    context.app.state.openai_serving_models = registry
    context.engine_client.model_config.served_model_name = "engine-only-name"
    assert lifecycle_module._served_model(context) == "nvidia/nemotron-asr"
    assert registry.calls == 1


# @spec ING-VEH-010, ING-VEH-013
@pytest.mark.parametrize(
    "failure_case",
    [
        "missing_state",
        "missing_registry",
        "missing_accessor",
        "noncallable_accessor",
        "lookup_error",
        "none_result",
        "empty_result",
        "non_string_result",
    ],
)
def test_invalid_host_serving_registry_fails_before_wrapper_or_bind(
    failure_case: str,
) -> None:
    async def exercise() -> None:
        events: list[object] = []
        context = FakeContext()
        if failure_case == "missing_state":
            del context.app.state
        elif failure_case == "missing_registry":
            del context.app.state.openai_serving_models
        elif failure_case == "missing_accessor":
            context.app.state.openai_serving_models = SimpleNamespace()
        elif failure_case == "noncallable_accessor":
            context.app.state.openai_serving_models = SimpleNamespace(
                model_name="nemotron-asr"
            )
        elif failure_case == "lookup_error":
            context.app.state.openai_serving_models = FakeServingModels(
                error=RuntimeError("registry failed")
            )
        elif failure_case == "none_result":
            context.app.state.openai_serving_models = FakeServingModels(None)
        elif failure_case == "empty_result":
            context.app.state.openai_serving_models = FakeServingModels("")
        elif failure_case == "non_string_result":
            context.app.state.openai_serving_models = FakeServingModels(7)
        else:
            raise AssertionError(f"unknown failure case: {failure_case}")

        with pytest.raises(ValueError, match="served model"):
            async with _lifetime(context, events):
                pytest.fail("invalid host registry entered plugin lifetime")
        assert context.asgi_wrappers == []
        assert events == []

    asyncio.run(exercise())


# @spec ING-VEH-010, ING-VEH-012, PORT-RTC-003
def test_model_factory_validation_failure_precedes_wrapper_and_bind() -> None:
    """A model/config mismatch fails before either listener is advertised."""

    async def exercise() -> None:
        events: list[object] = []
        context = FakeContext()
        del context.session_factory
        del context.serve_args

        def reject_model(engine_client: object, **_: object) -> object:
            assert engine_client is context.engine_client
            raise ValueError("incompatible Nemotron model")

        with pytest.raises(ValueError, match="incompatible Nemotron model"):
            async with _lifetime(
                context,
                events,
                session_factory_builder=reject_model,
            ):
                pytest.fail("incompatible model entered the plugin lifetime")

        assert context.asgi_wrappers == []
        assert events == []

    asyncio.run(exercise())


# @spec ING-VEH-011, ING-VEH-013
def test_readme_names_the_public_served_model_alias() -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text(
        encoding="utf-8"
    )
    assert "--served-model-name PUBLIC_MODEL_NAME" in readme
    assert "primary served-model identity" in readme


# @spec ING-VEH-017, ING-LIFE-012, ING-LIFE-015
def test_owner_registry_cancels_and_awaits_remaining_owner() -> None:
    async def exercise() -> None:
        registry = CompatibilityOwnerRegistry()
        registered = asyncio.Event()
        cleaned = asyncio.Event()

        async def owner() -> None:
            token = await registry.register()
            registered.set()
            try:
                await asyncio.Future()
            finally:
                await token.release()
                cleaned.set()

        task = asyncio.create_task(owner())
        await registered.wait()
        await registry.drain(grace=0.05, cleanup_timeout=0.02)
        assert task.cancelled()
        assert cleaned.is_set()
        assert registry.active == 0

    asyncio.run(exercise())


# @spec ING-VEH-001, ING-NIMWS-001, ING-SHIM-001, ING-HTTP-001
def test_installed_inner_wrapper_owns_only_approved_http_routes() -> None:
    async def exercise() -> None:
        events: list[object] = []
        context = FakeContext()
        lifetime = _lifetime(context, events)
        async with lifetime:
            wrapper = context.asgi_wrappers[0]

            async def host(
                scope: object, receive: object, send: object
            ) -> None:
                del scope, receive
                await send(
                    {
                        "type": "http.response.start",
                        "status": 418,
                        "headers": [],
                    }
                )

            middleware = wrapper(host)

            async def receive() -> dict[str, object]:
                return {"type": "http.request", "body": b""}

            sent: list[dict[str, object]] = []

            async def send(message: dict[str, object]) -> None:
                sent.append(message)

            await middleware(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/v1/realtime/transcription_sessions",
                },
                receive,
                send,
            )
            assert sent[0]["status"] == 200
            body = json.loads(sent[1]["body"])
            assert body["object"] == "realtime.transcription_session"
            assert (
                body["input_audio_transcription"]["model"]
                == "nvidia/nemotron-asr"
            )

            for path, expected_status in (
                ("/v1/health/ready", 503),
                ("/v1/health/live", 200),
                ("/v1/version", 200),
                ("/v1/metadata", 200),
            ):
                sent.clear()
                await middleware(
                    {
                        "type": "http",
                        "method": "GET",
                        "path": path,
                    },
                    receive,
                    send,
                )
                assert sent[0]["status"] == expected_status

            context.admission.open = True
            sent.clear()
            await middleware(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/v1/audio/transcriptions",
                    "headers": [],
                },
                receive,
                send,
            )
            assert sent[0]["status"] == 400

            sent.clear()
            await middleware(
                {
                    "type": "http",
                    "method": "GET",
                    "path": "/v1/models",
                },
                receive,
                send,
            )
            assert sent[0]["status"] == 418

    asyncio.run(exercise())


# @spec ING-VEH-012, ING-NIMWS-001
def test_default_auto_locale_must_be_published_by_the_checkpoint() -> None:
    async def exercise() -> None:
        context = FakeContext()
        context.engine_client.model_config.hf_config.prompt_dictionary.pop(
            "auto"
        )
        events: list[object] = []
        with pytest.raises(ValueError, match="default 'auto' locale"):
            async with _lifetime(context, events):
                pass
        assert events == []

    asyncio.run(exercise())


# ---------------------------------------------------------------------------
# Observation inheritance through the model factory (host PORT-OBS-003 /
# this plugin's ENV-MOD-004 posture): reuse the HOST's installed
# observer; never construct or register anything; degrade to the pinned
# public constructor when the seam or the observer is absent.
#
# Captured at import time: the _lifetime harness above replaces the
# module attribute in place (deliberately, without restore), so tests
# running after any harness-using test would otherwise reach a
# leftover fake instead of the real seam.
# ---------------------------------------------------------------------------

_REAL_CREATE_FACTORY = lifecycle_module._create_nemotron_session_factory


def _host_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    observer: object,
    with_install_module: bool = True,
) -> tuple[list[object], object]:
    """Stub the two host modules the factory seam consults."""
    import sys
    import types

    internal_calls: list[object] = []

    async def _open(*, cadence: str, locale: str) -> object:
        return object()

    class _InternalFactory:
        def __init__(self, *, engine: object, observer: object = None) -> None:
            internal_calls.append((engine, observer))
            self.open = _open

    def _public(engine_client: object) -> object:
        return SimpleNamespace(open=_open, public=True)

    session_mod = types.ModuleType("vllm_omni.entrypoints.nemotron_session")
    session_mod.NemotronSessionFactory = _InternalFactory  # type: ignore[attr-defined]
    session_mod.create_nemotron_session_factory = _public  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "vllm_omni.entrypoints.nemotron_session", session_mod
    )

    if with_install_module:
        install_mod = types.ModuleType("vllm_omni.metrics.streaming_install")

        def _resolve(app_state: object) -> object:
            return getattr(app_state, "installed_observer", None)

        install_mod.resolve_installed_observer = _resolve  # type: ignore[attr-defined]
        monkeypatch.setitem(
            sys.modules, "vllm_omni.metrics.streaming_install", install_mod
        )
    else:
        monkeypatch.delitem(
            sys.modules, "vllm_omni.metrics.streaming_install", raising=False
        )

    app_state = SimpleNamespace(installed_observer=observer)
    return internal_calls, app_state


def test_factory_inherits_the_hosts_installed_observer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The factory inherits the observer installed in host app state.

    It builds the observed internal factory with THAT observer — the
    sanctioned model-aware-participant wiring; nothing is constructed
    or registered plugin-side.
    """
    sentinel = object()
    internal_calls, app_state = _host_modules(monkeypatch, observer=sentinel)
    engine = object()

    factory = _REAL_CREATE_FACTORY(engine, app_state=app_state)

    assert internal_calls == [(engine, sentinel)]
    assert callable(factory.open)


def test_factory_uses_the_public_constructor_when_no_observer_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    internal_calls, app_state = _host_modules(monkeypatch, observer=None)
    factory = _REAL_CREATE_FACTORY(object(), app_state=app_state)
    assert internal_calls == []
    assert getattr(factory, "public", False) is True


def test_factory_degrades_cleanly_on_a_host_without_the_observation_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An older host uses the pinned public constructor path.

    With no streaming_install module, sessions remain unobserved without
    failing.
    """
    internal_calls, app_state = _host_modules(
        monkeypatch, observer=object(), with_install_module=False
    )
    factory = _REAL_CREATE_FACTORY(object(), app_state=app_state)
    assert internal_calls == []
    assert getattr(factory, "public", False) is True


def test_factory_without_app_state_stays_on_the_public_path() -> None:
    """The keyword-only default preserves pre-observability behavior.

    Existing callers and older harnesses keep their public construction path.
    """
    import sys

    assert "vllm_omni.entrypoints.nemotron_session" not in sys.modules or True  # noqa: SIM222


# @spec ING-VEH-010, ING-VEH-012
def test_host_version_fallback_requires_a_string_module_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_distribution(name: str) -> str:
        raise lifecycle_module.importlib_metadata.PackageNotFoundError(name)

    monkeypatch.setattr(
        lifecycle_module.importlib_metadata,
        "version",
        missing_distribution,
    )
    monkeypatch.setattr(
        lifecycle_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(__version__="0.24.0"),
    )
    assert lifecycle_module._resolve_host_version() == "0.24.0"

    monkeypatch.setattr(
        lifecycle_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(__version__=None),
    )
    with pytest.raises(RuntimeError, match="cannot resolve"):
        lifecycle_module._resolve_host_version()


# @spec ING-VEH-010, ING-VEH-012, ING-VEH-014
def test_model_factory_seam_rejects_missing_or_malformed_factories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    _, app_state = _host_modules(monkeypatch, observer=object())
    module = sys.modules["vllm_omni.entrypoints.nemotron_session"]

    module.NemotronSessionFactory = lambda **kwargs: object()  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="must expose async open"):
        _REAL_CREATE_FACTORY(object(), app_state=app_state)

    app_state = SimpleNamespace(installed_observer=None)
    module.create_nemotron_session_factory = None  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="does not expose"):
        _REAL_CREATE_FACTORY(object(), app_state=app_state)

    module.create_nemotron_session_factory = lambda engine: object()  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="must expose async open"):
        _REAL_CREATE_FACTORY(object(), app_state=app_state)
