"""Capability, route addition, files-enabled flag and startup checks."""

from __future__ import annotations

import logging

import grpc
import pytest
from support.app import (
    CORE_TRANSLATION_ROUTE,
    PLUGIN_PREFIX,
    all_routes,
    find_route,
)
from support.fake_transcription import Block, FakeTranscriptionServing, Return
from support.grpc_harness import FakeServicerContext
from support.http import CORE_ROUTE, ROUTE, MiB, assert_core_error, until
from support.riva import recognize, recognize_request

CORE_UPLOAD_LIMIT = 25 * MiB  # VLLM_MAX_AUDIO_CLIP_FILESIZE_MB default
FRAMING = 262_144


def _pcm(seconds: float = 0.5) -> bytes:
    from support.audio import pcm16_bytes, pcm16_sine

    return pcm16_bytes(pcm16_sine(seconds))


async def _config_types(plugin) -> list[str]:
    from riva.client.proto import riva_asr_pb2

    response = await plugin.servicer.GetRivaSpeechRecognitionConfig(
        riva_asr_pb2.RivaSpeechRecognitionConfigRequest(),
        FakeServicerContext(),
    )
    return [entry.parameters["type"] for entry in response.model_config]


# ---------------------------------------------------------- route addition


def _core_transcription_endpoint():
    from vllm.entrypoints.speech_to_text.transcription import api_router

    return api_router.create_transcriptions


def _core_translation_module() -> str:
    from vllm.entrypoints.speech_to_text.translation import api_router

    return api_router.__name__


def _posts(app, path: str) -> list:
    return [
        route
        for route in app.routes
        if getattr(route, "path", None) == path
        and "POST" in (getattr(route, "methods", None) or ())
    ]


# @spec ING-FILE-075
async def test_file_route_added_beside_core_route_only_when_core_has_it(
    files_env, clock
):
    from fastapi import FastAPI
    from support.app import attach_plugins
    from support.fake_session import FakeSessionFactory, fake_entry_points
    from support.state import make_args

    from vllm_riva_frontend.lifecycle import RivaFrontendPlugin

    env = await files_env(start=False)
    # The frontend's route is added under its namespace, once, and is the
    # frontend's own handler.
    routes = _posts(env.app, ROUTE)
    assert len(routes) == 1, f"no single frontend route at {ROUTE}"
    frontend = routes[0]
    assert frontend.endpoint is not _core_transcription_endpoint()
    # Core's route stays registered, once, with core's own handler.
    (core,) = _posts(env.app, CORE_ROUTE)
    assert core.endpoint is _core_transcription_endpoint()

    # Without core's route (here: a server whose task set has
    # ``transcription`` but whose route table lacks core's route), the
    # frontend adds no file route: core's route is the signal, not the task.
    bare = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    bare.state.args = make_args()
    factory = FakeSessionFactory(clock=clock)
    plugin = RivaFrontendPlugin(
        clock=clock,
        entry_points=fake_entry_points(nemotron=factory.constructor),
    )
    attach_plugins(bare, [plugin], ("transcription", "realtime"))
    assert find_route(bare, "POST", ROUTE) is None
    assert find_route(bare, "POST", CORE_ROUTE) is None


# @spec ING-FILE-075, ING-FILE-014
async def test_without_transcription_task_no_file_route_is_added(files_env):
    env = await files_env(tasks=("realtime",), start=False)
    assert find_route(env.app, "POST", ROUTE) is None
    assert find_route(env.app, "POST", CORE_ROUTE) is None
    assert find_route(env.app, *CORE_TRANSLATION_ROUTE) is None


# @spec ING-FILE-075
async def test_no_translation_route(files_env):
    env = await files_env(start=False)
    # The frontend serves no translation route of its own ...
    for kind, path, _, _ in all_routes(env.app):
        if path.startswith(PLUGIN_PREFIX):
            assert "translation" not in path, (kind, path)
    # ... and does not mount, replace or shadow core's.
    (translations,) = _posts(env.app, CORE_TRANSLATION_ROUTE[1])
    assert translations.endpoint.__module__ == _core_translation_module()
    for kind, path, _, _ in all_routes(env.app):
        if kind == "mount":
            assert not CORE_TRANSLATION_ROUTE[1].startswith(path), path
    # Every frontend route is under the namespace; none is core's path.
    frontend_paths = {
        path
        for _, path, _, endpoint in all_routes(env.app)
        if getattr(endpoint, "__module__", "").startswith("vllm_riva_frontend")
    }
    assert frontend_paths, "no frontend route found"
    assert all(p.startswith(PLUGIN_PREFIX) for p in frontend_paths), (
        frontend_paths
    )


# @spec ING-FILE-076
@pytest.mark.parametrize("files_enabled", [True, False], ids=["on", "off"])
async def test_core_route_unchanged_and_outside_cap_gate_guard(
    files_env, files_enabled
):
    from support.http import file_request

    engine = {} if files_enabled else {"num_gpu_blocks": 2}
    env = await files_env(engine=engine)
    assert env.plugin.files_enabled is files_enabled
    if files_enabled:
        # A full file gate does not hold back core's route.
        gate = env.plugin.file_gate
        deadline = env.clock.monotonic() + 3600
        held = [
            await gate.acquire(deadline)
            for _ in range(env.plugin.config.max_concurrent_files)
        ]
        in_flight = gate.in_flight
    # A closed ingress cap does not refuse core's route.
    env.plugin.ingress.close()

    block = Block(then=Return())
    env.service.script(block)
    # A guarded sampling value: core's validation decides it, not the
    # frontend's guard.
    exchange = file_request(
        env.app, [("temperature", "0.5")], path=CORE_ROUTE
    ).start()
    await until(
        lambda: block.entered.is_set() or exchange.done, what="core's call"
    )
    assert block.entered.is_set(), (exchange.status, exchange.body)
    # Core's call holds no frontend owner, admission or permit.
    assert not env.plugin.registry.records
    assert env.plugin.ingress.count == 0
    if files_enabled:
        assert env.plugin.file_gate.in_flight == in_flight
        assert env.plugin.file_gate.waiting == 0
    block.release()
    await exchange.finish()
    assert exchange.status == 200, exchange.body
    (call,) = env.calls
    assert call.request.temperature == 0.5
    if files_enabled:
        for permit in held:
            permit.release()


# ------------------------------------------------------------- capability


# @spec ING-FILE-014, ING-FILE-015
async def test_recognize_is_unimplemented_without_core_service(files_env):
    env = await files_env(tasks=("realtime",))
    assert env.plugin.files_enabled is False
    assert env.plugin.file_gate is None
    outcome = await recognize(
        env.plugin.servicer, recognize_request(_pcm()), FakeServicerContext()
    )
    assert outcome.status is grpc.StatusCode.UNIMPLEMENTED
    assert outcome.details.startswith("unsupported_capability:")
    assert "offline" not in await _config_types(env.plugin)


# @spec ING-FILE-015
async def test_files_enabled_flag_decides_recognize_and_offline_listing(
    files_env,
):
    env = await files_env()
    assert env.plugin.files_enabled is True
    assert env.plugin.file_gate is not None
    types = await _config_types(env.plugin)
    assert "online" in types
    assert "offline" in types
    outcome = await recognize(
        env.plugin.servicer, recognize_request(_pcm()), FakeServicerContext()
    )
    assert outcome.abort is None, outcome.details
    assert outcome.response is not None
    assert len(env.calls) == 1


# @spec ING-FILE-016, ING-FILE-015
@pytest.mark.parametrize(
    "capacity",
    [
        {"num_gpu_blocks": 2},  # one slot
        {"max_num_seqs": 1},  # active limit 1
        {"max_num_seqs": 16, "max_num_active_seqs": 1},
    ],
    ids=["one-slot", "one-active", "one-active-explicit"],
)
async def test_tiny_capacity_disables_files_with_a_warning(
    files_env, caplog, capacity
):
    caplog.set_level(logging.WARNING)
    env = await files_env(engine=capacity)
    assert env.plugin.files_enabled is False
    assert env.plugin.config.max_concurrent_files == 0
    assert env.plugin.config.max_file_owners == 0
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and r.name.startswith("vllm_riva_frontend")
    ]
    assert warnings, "files disabled without a startup warning"
    outcome = await recognize(
        env.plugin.servicer, recognize_request(_pcm()), FakeServicerContext()
    )
    assert outcome.status is grpc.StatusCode.UNIMPLEMENTED
    assert outcome.details.startswith("unsupported_capability:")
    assert "offline" not in await _config_types(env.plugin)
    assert env.calls == []


# @spec ING-FILE-083, ING-FILE-055
async def test_disabled_files_route_answers_501_first(
    files_env,
):
    env = await files_env(engine={"num_gpu_blocks": 2})
    assert find_route(env.app, "POST", ROUTE) is not None
    exchange = await env.request([("model", "x")]).finish()
    assert_core_error(exchange, 501, "unsupported_capability")
    assert exchange.chunks_delivered == 0
    assert not env.plugin.registry.records
    assert env.plugin.ingress.count == 0
    assert env.calls == []

    # The files-enabled check precedes the ingress cap: a closed cap does
    # not turn the answer into 503.
    env.plugin.ingress.close()
    exchange = await env.request().finish()
    assert_core_error(exchange, 501, "unsupported_capability")
    assert exchange.chunks_delivered == 0


# --------------------------------------------------------------- gate size


# @spec ING-FILE-018, ING-FILE-009
@pytest.mark.parametrize(
    ("configured", "engine"),
    [
        (16, {}),  # equal to the slot count (16)
        (17, {}),
        (8, {"max_num_seqs": 8}),  # equal to the active limit
        (1, {"num_gpu_blocks": 2}),  # explicit size on a one-slot server
    ],
    ids=["equal-slots", "above-slots", "equal-active", "tiny-explicit"],
)
async def test_configured_gate_not_below_capacity_fails_startup(
    files_env, configured, engine
):
    from vllm_riva_frontend.lifecycle import StartupError

    with pytest.raises(StartupError) as raised:
        await files_env(
            riva_frontend={"max_concurrent_files": configured},
            engine=engine,
            start=False,
        )
    assert any("max_concurrent_files" in f for f in raised.value.failures)


# @spec ING-FILE-009, ING-FILE-018
async def test_configured_gate_below_both_limits_is_accepted(files_env):
    env = await files_env(riva_frontend={"max_concurrent_files": 15})
    assert env.plugin.files_enabled is True
    assert env.plugin.config.max_concurrent_files == 15
    capacity = env.plugin.capacity
    assert env.plugin.config.max_concurrent_files < capacity.slots
    assert env.plugin.config.max_concurrent_files < capacity.active_limit


# @spec ING-FILE-018
async def test_configured_gate_is_not_checked_without_core_service(
    files_env,
):
    env = await files_env(
        tasks=("realtime",), riva_frontend={"max_concurrent_files": 16}
    )
    assert env.plugin.files_enabled is False


# @spec ING-FILE-009
async def test_derived_gate_is_below_both_limits(files_env):
    env = await files_env(engine={"num_gpu_blocks": 3, "max_num_seqs": 2})
    # slots 2, active 2: the derived default is 1, below both.
    assert env.plugin.files_enabled is True
    assert env.plugin.config.max_concurrent_files == 1


# ---------------------------------------------------------- body and decode


# @spec ING-FILE-006
@pytest.mark.parametrize(
    "envelope",
    [CORE_UPLOAD_LIMIT + FRAMING - 1, 0, -1, float("inf")],
    ids=["below-upload-plus-framing", "zero", "negative", "infinite"],
)
async def test_envelope_below_core_upload_limit_fails_startup(
    files_env, envelope
):
    from vllm_riva_frontend.lifecycle import StartupError

    with pytest.raises(StartupError) as raised:
        await files_env(
            riva_frontend={"http_multipart_envelope_max_bytes": envelope},
            start=False,
        )
    assert any(
        "http_multipart_envelope_max_bytes" in f for f in raised.value.failures
    )


# @spec ING-FILE-006
async def test_envelope_at_core_upload_limit_plus_framing_is_accepted(
    files_env,
):
    env = await files_env(
        riva_frontend={
            "http_multipart_envelope_max_bytes": CORE_UPLOAD_LIMIT + FRAMING
        }
    )
    assert env.plugin.files_enabled is True


# @spec ING-FILE-006
async def test_envelope_check_follows_core_upload_setting(
    files_env, monkeypatch
):
    from vllm_riva_frontend.lifecycle import StartupError

    monkeypatch.setenv("VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", "40")
    with pytest.raises(StartupError):
        await files_env(start=False)  # the default envelope is 32.25 MiB


# @spec ING-FILE-046
async def test_unary_duration_above_core_decode_limit_fails_startup(
    files_env, monkeypatch
):
    from vllm_riva_frontend.lifecycle import StartupError

    monkeypatch.setenv("VLLM_MAX_AUDIO_DECODE_DURATION_S", "300")
    service = FakeTranscriptionServing()
    service.max_audio_decode_duration_s = 300
    with pytest.raises(StartupError) as raised:
        await files_env(service=service, start=False)
    assert any(
        "unary_max_decoded_duration_seconds" in f for f in raised.value.failures
    )


# @spec ING-FILE-046
async def test_unary_duration_at_core_decode_limit_is_accepted(files_env):
    env = await files_env(
        riva_frontend={"unary_max_decoded_duration_seconds": 600}
    )
    assert env.plugin.files_enabled is True


# @spec ING-FILE-081
async def test_no_startup_relation_between_duration_and_model_len(
    files_env, caplog
):
    # A max_model_len far too small for a 600 s file still starts, with no
    # warning about it: the deployment guide sizes ``--max-model-len``.
    caplog.set_level(logging.WARNING)
    env = await files_env(engine={"max_model_len": 128})
    assert env.plugin.files_enabled is True
    messages = [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING
        and r.name.startswith("vllm_riva_frontend")
    ]
    assert not any("max_model_len" in m for m in messages), messages


# ------------------------------------------------------ inherited sampling


# @spec ING-FILE-054
async def test_inherited_guarded_sampling_settings_log_a_warning(
    files_env, caplog
):
    caplog.set_level(logging.WARNING)
    # ``init_state`` reads the settings where core's serving keeps them
    # (``default_sampling_params``, built from the model's
    # ``get_diff_sampling_param()``; implementation layout, Shapes).
    service = FakeTranscriptionServing(
        default_sampling_params={"temperature": 0.7, "top_p": 0.9}
    )
    await files_env(service=service, start=False)
    messages = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
        and r.name.startswith("vllm_riva_frontend")
    ]
    assert any("temperature" in m for m in messages), messages
