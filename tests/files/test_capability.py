"""Capability, route replacement, files-enabled flag and startup checks."""

from __future__ import annotations

import logging

import grpc
import pytest
from support.app import find_route
from support.fake_transcription import FakeTranscriptionServing
from support.grpc_harness import FakeServicerContext
from support.http import ROUTE, MiB, assert_core_error
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


# ------------------------------------------------------- route replacement


# @spec ING-FILE-003
async def test_route_replaces_core_transcriptions_and_keeps_translations(
    files_env,
):
    from vllm.entrypoints.speech_to_text.transcription import (
        api_router as core_transcription,
    )
    from vllm.entrypoints.speech_to_text.translation import (
        api_router as core_translation,
    )

    env = await files_env(start=False)
    transcriptions = [
        route
        for route in env.app.routes
        if getattr(route, "path", None) == ROUTE
        and "POST" in (getattr(route, "methods", None) or ())
    ]
    assert len(transcriptions) == 1
    assert transcriptions[0].endpoint is not (
        core_transcription.create_transcriptions
    )
    translations = find_route(env.app, "POST", "/v1/audio/translations")
    assert translations is not None
    assert translations.endpoint.__module__ == core_translation.__name__


# @spec ING-FILE-003, ING-FILE-014
async def test_without_transcription_task_no_file_route_is_added(files_env):
    env = await files_env(tasks=("realtime",), start=False)
    assert find_route(env.app, "POST", ROUTE) is None
    assert find_route(env.app, "POST", "/v1/audio/translations") is None


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


# @spec ING-FILE-017, ING-FILE-055
async def test_disabled_files_route_answers_501_before_cap_and_body(
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


# @spec ING-FILE-047
async def test_no_startup_relation_between_duration_and_model_len(
    files_env,
):
    # A max_model_len far too small for a 600 s file still starts: the
    # budget is the model's check per request, not a frontend relation.
    env = await files_env(engine={"max_model_len": 128})
    assert env.plugin.files_enabled is True


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
