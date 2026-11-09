"""``Recognize``: the complete-file path on core's transcription service."""

from __future__ import annotations

import asyncio
import io
import struct

import grpc
import numpy as np
import pytest
from support.audio import (
    float_from_pcm16,
    g711_bytes,
    pcm16_bytes,
    pcm16_ramp,
    pcm16_sine,
)
from support.fake_session import FakeSessionFactory
from support.fake_transcription import (
    Block,
    Raise,
    Return,
    error_response,
    transcription_response,
)
from support.grpc_harness import FakeServicerContext
from support.http import until
from support.riva import recognize, recognize_request

from .helpers import GuardedCall, is_uuid


def _wav_samples(wav: bytes) -> tuple[dict, np.ndarray]:
    """Parse a canonical 44-byte WAV into its format and int16 samples."""
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    assert wav[12:16] == b"fmt "
    code, channels, rate, _, _, bits = struct.unpack("<HHIIHH", wav[20:36])
    assert wav[36:40] == b"data"
    (length,) = struct.unpack("<I", wav[40:44])
    assert len(wav) == 44 + length
    fmt = {"code": code, "channels": channels, "rate": rate, "bits": bits}
    return fmt, np.frombuffer(wav[44:], dtype="<i2")


class _HeldGate:
    """Every permit of the file gate, held until ``release()``."""

    def __init__(self, permits: list) -> None:
        self.permits = permits

    def release(self) -> None:
        for permit in self.permits:
            permit.release()


async def _held_gate(env) -> _HeldGate:
    """Take every permit, so the next ``Recognize`` must wait for one.

    The gate's size is the derived ``max_concurrent_files`` (4 on the
    default 16-slot engine, ING-ADM-014); holding fewer leaves a permit free
    and a test of the gate wait would pass or fail without waiting.
    """
    gate = env.plugin.file_gate
    size = env.plugin.config.max_concurrent_files
    deadline = env.clock.monotonic() + 3600
    held = _HeldGate([await gate.acquire(deadline) for _ in range(size)])
    assert size >= 1 and gate.in_flight == size, "the gate is not full"
    return held


# ------------------------------------------------------------ the path


# @spec ING-FILE-001, ING-FILE-010
@pytest.mark.parametrize(
    ("encoding", "rate", "make"),
    [
        ("LINEAR_PCM", 16_000, lambda s: pcm16_bytes(s)),
        ("LINEAR_PCM", 8_000, lambda s: pcm16_bytes(s)),
        ("MULAW", 8_000, lambda s: g711_bytes(s, "mulaw")),
        ("ALAW", 8_000, lambda s: g711_bytes(s, "alaw")),
    ],
    ids=["pcm16-16k", "pcm16-8k", "mulaw", "alaw"],
)
async def test_recognize_sends_core_one_16k_wav_and_plain_request(
    files_env, encoding, rate, make
):
    env = await files_env()
    env.service.default_text = "hello"
    samples = pcm16_sine(0.5, rate=rate)
    outcome = await recognize(
        env.plugin.servicer,
        recognize_request(
            make(samples), encoding=encoding, sample_rate_hertz=rate
        ),
        FakeServicerContext(),
    )
    assert outcome.abort is None, outcome.details
    (call,) = env.calls
    fmt, wav_samples = _wav_samples(call.audio_data)
    assert fmt == {"code": 1, "channels": 1, "rate": 16_000, "bits": 16}
    assert len(wav_samples) == 8_000
    request = call.request
    assert request.response_format == "json"
    assert not request.stream
    assert request.use_beam_search is False
    assert request.max_completion_tokens is None
    assert "use_beam_search" not in request.model_fields_set
    assert "max_completion_tokens" not in request.model_fields_set
    # No session factory on this path.
    assert env.factory.leases == []
    assert "open" not in env.factory.names()


# @spec ING-FILE-001
async def test_recognize_answers_only_after_core_completes(files_env):
    env = await files_env()
    block = Block(then=Return(transcription_response("done")))
    env.service.script(block)
    rpc = asyncio.ensure_future(
        recognize(
            env.plugin.servicer,
            recognize_request(pcm16_bytes(pcm16_sine(0.2))),
            FakeServicerContext(),
        )
    )
    await asyncio.wait_for(block.entered.wait(), 2)
    assert env.plugin.file_gate.in_flight == 1
    await asyncio.sleep(0)
    assert not rpc.done()
    block.release()
    outcome = await asyncio.wait_for(rpc, 2)
    assert outcome.response.results[0].alternatives[0].transcript == "done"
    await until(lambda: env.plugin.file_gate.in_flight == 0, what="release")


# @spec ING-FILE-045
@pytest.mark.parametrize("text", ["the transcript", ""])
async def test_recognize_returns_one_result_one_alternative(files_env, text):
    env = await files_env()
    env.service.script(Return(transcription_response(text)))
    outcome = await recognize(
        env.plugin.servicer,
        recognize_request(pcm16_bytes(pcm16_sine(0.2))),
        FakeServicerContext(),
    )
    response = outcome.response
    assert response is not None, outcome.details
    assert len(response.results) == 1
    assert len(response.results[0].alternatives) == 1
    assert response.results[0].alternatives[0].transcript == text


# ------------------------------------------------------------- locale


# @spec ING-FILE-013
@pytest.mark.parametrize(
    ("locale", "language"),
    [("en-US", "en"), ("de-DE", "de"), ("auto", None)],
)
async def test_recognize_sends_the_exact_locale(files_env, locale, language):
    env = await files_env()
    await recognize(
        env.plugin.servicer,
        recognize_request(pcm16_bytes(pcm16_sine(0.2)), language_code=locale),
        FakeServicerContext(),
    )
    (call,) = env.calls
    assert call.request.vllm_xargs == {"locale": locale}
    assert call.request.language == language


# @spec ING-FILE-013
async def test_unsupported_language_of_a_known_locale_sends_no_language(
    files_env,
):
    factory = FakeSessionFactory(locales=("en-US", "fr-FR", "auto"))
    env = await files_env(factory=factory)
    await recognize(
        env.plugin.servicer,
        recognize_request(pcm16_bytes(pcm16_sine(0.2)), language_code="fr-FR"),
        FakeServicerContext(),
    )
    (call,) = env.calls
    assert call.request.vllm_xargs == {"locale": "fr-FR"}
    assert call.request.language is None


# @spec ING-FILE-013
async def test_empty_language_code_takes_the_default_locale(files_env):
    factory = FakeSessionFactory(default_locale="de-DE")
    env = await files_env(factory=factory)
    await recognize(
        env.plugin.servicer,
        recognize_request(pcm16_bytes(pcm16_sine(0.2)), language_code=""),
        FakeServicerContext(),
    )
    (call,) = env.calls
    assert call.request.vllm_xargs == {"locale": "de-DE"}
    assert call.request.language == "de"


# @spec ING-FILE-013
async def test_no_locale_and_no_default_names_language_code(files_env):
    factory = FakeSessionFactory(default_locale=None)
    env = await files_env(factory=factory)
    outcome = await recognize(
        env.plugin.servicer,
        recognize_request(pcm16_bytes(pcm16_sine(0.2)), language_code=""),
        FakeServicerContext(),
    )
    assert outcome.status is grpc.StatusCode.INVALID_ARGUMENT
    assert outcome.details.startswith("invalid_config_field:")
    assert "language_code" in outcome.details
    assert env.calls == []


# @spec ING-FILE-013, ING-FILE-002
async def test_unknown_locale_is_refused_before_core(files_env):
    env = await files_env()
    outcome = await recognize(
        env.plugin.servicer,
        recognize_request(pcm16_bytes(pcm16_sine(0.2)), language_code="xx-XX"),
        FakeServicerContext(),
    )
    assert outcome.status is grpc.StatusCode.INVALID_ARGUMENT
    assert outcome.details.startswith("unknown_locale:")
    assert env.calls == []


# @spec ING-FILE-013
@pytest.mark.parametrize(
    ("locale", "language"),
    [
        ("en-US", "en"),
        ("enUS", "en"),
        ("haw-US", "haw"),
        ("zh-Hans-CN", "zh"),
        ("es", "es"),
        ("auto", None),
        ("e-US", None),
        ("abcd-EF", None),
        ("EN-US", None),
    ],
)
def test_locale_language_rule(locale, language):
    from vllm_riva_frontend.files import locale_language

    assert locale_language(locale) == language


# @spec ING-FILE-013
def test_recognize_language_needs_a_supported_language():
    from vllm_riva_frontend.files import recognize_language

    supported = {"en": "english", "de": "german"}
    assert recognize_language("en-US", supported) == "en"
    assert recognize_language("fr-FR", supported) is None
    assert recognize_language("auto", supported) is None


# @spec ING-FILE-044
@pytest.mark.parametrize("client_locale", ["en-US", "de-DE", "auto", ""])
async def test_recognize_locale_equals_the_stream_locale(
    files_env, client_locale
):
    from vllm_riva_frontend.lease import resolve_locale

    env = await files_env()
    await recognize(
        env.plugin.servicer,
        recognize_request(
            pcm16_bytes(pcm16_sine(0.2)), language_code=client_locale
        ),
        FakeServicerContext(),
    )
    (call,) = env.calls
    stream_locale = resolve_locale(
        client_locale, env.factory, field="language_code"
    )
    assert call.request.vllm_xargs == {"locale": stream_locale}


# ----------------------------------------------------------- request ID


# @spec ING-FILE-027
async def test_recognize_fixes_its_engine_id_with_a_minimal_request(
    files_env,
):
    env = await files_env()
    await recognize(
        env.plugin.servicer,
        recognize_request(pcm16_bytes(pcm16_sine(0.2))),
        FakeServicerContext(metadata=[("x-request-id", "client")]),
    )
    (call,) = env.calls
    headers = call.raw_request.scope["headers"]
    assert len(headers) == 1
    name, value = headers[0]
    assert name == b"x-request-id"
    assert is_uuid(value.decode())
    assert call.request_id == f"transcribe-{value.decode()}"
    assert call.raw_request.scope["type"] == "http"


# @spec ING-FILE-027
def test_build_raw_request_without_scope_is_minimal():
    from vllm_riva_frontend.files import build_raw_request

    request = build_raw_request()
    assert request.scope["type"] == "http"
    assert [k for k, _ in request.scope["headers"]] == [b"x-request-id"]


# ------------------------------------------------ validation before permit


# @spec ING-FILE-002
@pytest.mark.parametrize(
    ("kwargs", "audio", "status", "code"),
    [
        (
            {"encoding": "FLAC"},
            b"fLaC\x00\x00",
            grpc.StatusCode.INVALID_ARGUMENT,
            "unsupported_format",
        ),
        (
            {"encoding": "LINEAR_PCM", "sample_rate_hertz": 44_100},
            b"\x00\x00" * 100,
            grpc.StatusCode.INVALID_ARGUMENT,
            "unsupported_format",
        ),
        (
            {"encoding": "ENCODING_UNSPECIFIED", "sample_rate_hertz": 0},
            b"not a riff header at all" * 4,
            grpc.StatusCode.INVALID_ARGUMENT,
            "unsupported_format",
        ),
        (
            {"encoding": "MULAW", "sample_rate_hertz": 8_000},
            b"\xff" * (8_000 * 601),  # 601 s, above the 600 s bound
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            "request_too_large",
        ),
        (
            {"language_code": "xx-XX"},
            b"\x00\x00" * 100,
            grpc.StatusCode.INVALID_ARGUMENT,
            "unknown_locale",
        ),
    ],
    ids=["encoding", "rate", "riff", "duration", "locale"],
)
async def test_validation_fails_before_waiting_for_a_permit(
    files_env, kwargs, audio, status, code
):
    env = await files_env()
    held = await _held_gate(env)
    outcome = await asyncio.wait_for(
        recognize(
            env.plugin.servicer,
            recognize_request(audio, **kwargs),
            FakeServicerContext(),
        ),
        5,
    )
    assert outcome.status is status, outcome.details
    assert outcome.details.startswith(f"{code}:")
    assert env.plugin.file_gate.waiting == 0
    assert env.calls == []
    held.release()


# @spec ING-FILE-002
async def test_encoded_bound_fails_before_waiting_for_a_permit(files_env):
    env = await files_env(
        riva_frontend={
            "unary_max_encoded_audio_bytes": 4096,
            "max_riff_header_bytes": 1024,
            "grpc_receive_max_bytes": 4096 + 65_536,
        }
    )
    held = await _held_gate(env)
    outcome = await asyncio.wait_for(
        recognize(
            env.plugin.servicer,
            recognize_request(b"\x00\x00" * 4096),
            FakeServicerContext(),
        ),
        5,
    )
    assert outcome.status is grpc.StatusCode.RESOURCE_EXHAUSTED
    assert outcome.details.startswith("request_too_large:")
    assert env.plugin.file_gate.waiting == 0
    held.release()


# @spec ING-FILE-002
async def test_observed_cancellation_wins_over_a_validation_error(files_env):
    from support.grpc_harness import RpcAbort

    env = await files_env()
    context = FakeServicerContext(clock=env.clock)
    context.cancel()
    rpc = asyncio.ensure_future(
        env.plugin.servicer.Recognize(
            recognize_request(b"\x00\x00", language_code="xx-XX"), context
        )
    )
    done, _ = await asyncio.wait({rpc}, timeout=2)
    assert rpc in done, "a cancelled Recognize did not end"
    # The cancellation propagates: no response, and not the validation
    # error. The client is gone, so no particular status is required; a
    # status the handler does set is the cancellation's own.
    if not rpc.cancelled():
        error = rpc.exception()
        assert isinstance(error, RpcAbort), f"ended with {error!r}"
        assert error.code is grpc.StatusCode.CANCELLED, error
    assert env.calls == []
    # Cleanup: no owner, admission slot or permit is left behind.
    await until(lambda: not env.plugin.registry.records, what="unregistered")
    assert env.plugin.ingress.count == 0
    assert env.plugin.ingress.file_count == 0
    assert env.plugin.file_gate.in_flight == 0
    assert env.plugin.file_gate.waiting == 0


# @spec ING-FILE-002, ING-FILE-065
async def test_expired_deadline_wins_over_a_validation_error(files_env):
    env = await files_env()
    outcome = await recognize(
        env.plugin.servicer,
        recognize_request(b"\x00\x00", language_code="xx-XX"),
        FakeServicerContext(time_remaining=0.0, clock=env.clock),
    )
    assert outcome.status is grpc.StatusCode.DEADLINE_EXCEEDED
    assert outcome.response is None
    assert env.calls == []


# ------------------------------------------------------------ deadlines


# @spec ING-FILE-065
async def test_rpc_deadline_applies_during_the_gate_wait(files_env):
    env = await files_env()
    held = await _held_gate(env)
    rpc = asyncio.ensure_future(
        recognize(
            env.plugin.servicer,
            recognize_request(pcm16_bytes(pcm16_sine(0.2))),
            FakeServicerContext(time_remaining=20.0, clock=env.clock),
        )
    )
    await until(lambda: env.plugin.file_gate.waiting == 1, what="waiting")
    await env.clock.advance(20.1)  # before the 30 s gate bound
    outcome = await asyncio.wait_for(rpc, 2)
    # The wait is clipped to the RPC deadline and expires as it, not as the
    # gate's own bound (``busy``).
    assert outcome.response is None
    assert outcome.status is grpc.StatusCode.DEADLINE_EXCEEDED
    assert outcome.details.startswith("request_timeout:"), outcome.details
    assert env.calls == []
    held.release()


# @spec ING-FILE-065, ING-ERR-056
async def test_one_rpc_deadline_spans_transit_gate_wait_and_execution(
    files_env,
):
    env = await files_env()
    # The deadline is set 20 s out before the servicer is entered; 5 s pass
    # in transit, 7 s in the gate wait and the rest in execution.
    context = FakeServicerContext(time_remaining=20.0, clock=env.clock)
    deadline = context.deadline
    await env.clock.advance(5.0)
    held = await _held_gate(env)
    block = Block(then=Return(transcription_response("late")))
    env.service.script(block)
    rpc = asyncio.ensure_future(
        recognize(
            env.plugin.servicer,
            recognize_request(pcm16_bytes(pcm16_sine(0.2))),
            context,
        )
    )
    await until(lambda: env.plugin.file_gate.waiting == 1, what="waiting")
    await env.clock.advance(7.0)
    held.release()
    await asyncio.wait_for(block.entered.wait(), 2)
    # 12 s of the 20 are spent; execution gets only what is left, and the
    # RPC expires at its original absolute deadline, as itself.
    await env.clock.advance(deadline - env.clock.monotonic() - 0.1)
    assert not rpc.done()
    assert env.engine.aborted == []
    await env.clock.advance(0.2)
    await until(lambda: env.engine.aborted, what="abort at the RPC deadline")
    block.release()
    outcome = await asyncio.wait_for(rpc, 5)
    assert outcome.response is None
    assert outcome.status is grpc.StatusCode.DEADLINE_EXCEEDED
    assert outcome.details.startswith("request_timeout:"), outcome.details
    await until(lambda: env.plugin.file_gate.in_flight == 0, what="release")


# @spec ING-FILE-065, ING-FILE-068
async def test_rpc_deadline_during_execution_aborts_without_response(
    files_env,
):
    env = await files_env()
    block = Block(then=Return(transcription_response("late")))
    env.service.script(block)
    rpc = asyncio.ensure_future(
        recognize(
            env.plugin.servicer,
            recognize_request(pcm16_bytes(pcm16_sine(0.2))),
            FakeServicerContext(time_remaining=10.0, clock=env.clock),
        )
    )
    await asyncio.wait_for(block.entered.wait(), 2)
    await env.clock.advance(10.1)
    await until(lambda: env.engine.aborted, what="abort")
    assert set(env.engine.aborted) == {env.calls[0].request_id}
    block.release()
    outcome = await asyncio.wait_for(rpc, 5)
    assert outcome.response is None
    assert outcome.status is grpc.StatusCode.DEADLINE_EXCEEDED
    assert outcome.details.startswith("request_timeout:"), outcome.details
    await until(lambda: env.plugin.file_gate.in_flight == 0, what="release")


# @spec ING-FILE-065, ING-ERR-056, ING-FILE-067
async def test_rpc_deadline_due_with_a_core_failure_wins(files_env):
    env = await files_env()
    block = Block(then=Raise(RuntimeError("core failed at the deadline")))
    env.service.script(block)
    context = FakeServicerContext(time_remaining=10.0, clock=env.clock)
    rpc = asyncio.ensure_future(
        recognize(
            env.plugin.servicer,
            recognize_request(pcm16_bytes(pcm16_sine(0.2))),
            context,
        )
    )
    await asyncio.wait_for(block.entered.wait(), 2)
    await env.clock.settle()
    # Core's failure and the RPC deadline's expiry reach the owner together:
    # the expired deadline is arbitrated before the failure is selected.
    block.release()
    env.clock.set(context.deadline)
    outcome = await asyncio.wait_for(rpc, 5)
    assert outcome.response is None
    assert outcome.status is grpc.StatusCode.DEADLINE_EXCEEDED, outcome.details
    assert outcome.details.startswith("request_timeout:"), outcome.details
    await until(lambda: env.plugin.file_gate.in_flight == 0, what="release")


# @spec ING-FILE-064
async def test_execution_bound_expiry_is_request_timeout(files_env):
    env = await files_env()
    block = Block()
    env.service.script(block)
    rpc = asyncio.ensure_future(
        recognize(
            env.plugin.servicer,
            recognize_request(pcm16_bytes(pcm16_sine(0.2))),
            FakeServicerContext(),
        )
    )
    await asyncio.wait_for(block.entered.wait(), 2)
    await env.clock.advance(env.plugin.config.file_execution_timeout - 1)
    assert not rpc.done()
    await env.clock.advance(2)
    await until(lambda: env.engine.aborted, what="abort")
    block.release()
    outcome = await asyncio.wait_for(rpc, 5)
    assert outcome.status is grpc.StatusCode.DEADLINE_EXCEEDED
    assert outcome.details.startswith("request_timeout:")


# @spec ING-FILE-068, ING-FILE-029, ING-FILE-074
async def test_cancelled_recognize_aborts_and_settles_before_release(
    files_env,
):
    env = await files_env()
    guarded = GuardedCall(env.service)
    block = Block(then=Return(transcription_response("late")))
    env.service.script(block)
    context = FakeServicerContext()
    rpc = asyncio.ensure_future(
        recognize(
            env.plugin.servicer,
            recognize_request(pcm16_bytes(pcm16_sine(0.2))),
            context,
        )
    )
    await asyncio.wait_for(block.entered.wait(), 2)
    context.cancel()
    await until(lambda: env.engine.aborted, what="abort")
    count = len(env.engine.aborted)
    await env.clock.advance(0.25)
    await until(lambda: len(env.engine.aborted) > count, what="retry")
    assert env.plugin.file_gate.in_flight == 1
    before = len(env.engine.aborted)
    block.release()
    await until(lambda: env.plugin.file_gate.in_flight == 0, what="release")
    assert len(env.engine.aborted) > before, "no final abort"
    assert guarded.cancelled == 0
    outcome = await asyncio.wait_for(rpc, 2)
    assert outcome.response is None


# ------------------------------------------------------------- errors


# @spec ING-FILE-047
async def test_token_budget_refusal_is_request_too_large(files_env):
    from vllm_riva_frontend.errors import MODEL_LEN_PREFIX

    assert MODEL_LEN_PREFIX == "file exceeds max_model_len: "
    env = await files_env()
    env.service.script(Raise(ValueError(MODEL_LEN_PREFIX + "80000 > 76800")))
    outcome = await recognize(
        env.plugin.servicer,
        recognize_request(pcm16_bytes(pcm16_sine(0.2))),
        FakeServicerContext(),
    )
    assert outcome.status is grpc.StatusCode.RESOURCE_EXHAUSTED
    assert outcome.details.startswith("request_too_large:")


# @spec ING-FILE-050
@pytest.mark.parametrize(
    ("behaviour", "status", "code", "field"),
    [
        (
            Return(error_response("bad", 400, param="language")),
            grpc.StatusCode.INVALID_ARGUMENT,
            "invalid_config_field",
            "language",
        ),
        (
            Return(error_response("bad", 422)),
            grpc.StatusCode.INVALID_ARGUMENT,
            "invalid_config_field",
            "audio",
        ),
        (
            Return(error_response("too big", 413)),
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            "request_too_large",
            None,
        ),
        (
            Return(error_response("down", 503)),
            grpc.StatusCode.UNAVAILABLE,
            "service_unavailable",
            None,
        ),
        (
            Return(error_response("boom", 500)),
            grpc.StatusCode.INTERNAL,
            "internal",
            None,
        ),
        (
            Raise(ValueError("Invalid or unsupported audio file.")),
            grpc.StatusCode.INVALID_ARGUMENT,
            "invalid_audio",
            None,
        ),
        (
            Raise(ValueError("something else")),
            grpc.StatusCode.INVALID_ARGUMENT,
            "invalid_config_field",
            None,
        ),
        (
            Raise(RuntimeError("unexpected")),
            grpc.StatusCode.INTERNAL,
            "internal",
            None,
        ),
    ],
    ids=[
        "returned-400",
        "returned-422",
        "returned-413",
        "returned-503",
        "returned-500",
        "raised-decoder",
        "raised-value",
        "raised-other",
    ],
)
async def test_returned_and_raised_core_failures_project_separately(
    files_env, behaviour, status, code, field
):
    env = await files_env()
    env.service.script(behaviour)
    outcome = await recognize(
        env.plugin.servicer,
        recognize_request(pcm16_bytes(pcm16_sine(0.2))),
        FakeServicerContext(),
    )
    assert outcome.status is status, outcome.details
    assert outcome.details.startswith(f"{code}:")
    if field is not None:
        assert field in outcome.details
    await until(lambda: env.plugin.file_gate.in_flight == 0, what="release")


# ------------------------------------------------------- load counter


# @spec ING-FILE-049
async def test_recognize_is_outside_the_server_load_counter(files_env):
    env = await files_env(load_tracking=True)
    block = Block()
    env.service.script(block)
    rpc = asyncio.ensure_future(
        recognize(
            env.plugin.servicer,
            recognize_request(pcm16_bytes(pcm16_sine(0.2))),
            FakeServicerContext(),
        )
    )
    await asyncio.wait_for(block.entered.wait(), 2)
    assert env.state.server_load_metrics == 0
    block.release()
    await asyncio.wait_for(rpc, 2)
    assert env.state.server_load_metrics == 0


# --------------------------------------------------------- equivalence


def _stream_side(encoding: str, rate: int, data: bytes) -> np.ndarray:
    from vllm_riva_frontend.frontend import StreamingAudioFrontend, requantize

    frontend = StreamingAudioFrontend(encoding=encoding, sample_rate_hz=rate)
    pieces = list(frontend.decode(data, max_samples=4096))
    pieces.append(frontend.drain())
    samples = np.concatenate([p for p in pieces if len(p)]).astype(np.float32)
    return requantize(samples)


# @spec ING-FILE-044
@pytest.mark.parametrize(
    ("encoding", "rate", "law"),
    [
        ("LINEAR_PCM", 16_000, None),
        ("LINEAR_PCM", 8_000, None),
        ("MULAW", 8_000, "mulaw"),
        ("ALAW", 8_000, "alaw"),
    ],
    ids=["pcm16-16k", "pcm16-8k", "mulaw", "alaw"],
)
async def test_recognize_samples_equal_the_stream_side_samples(
    files_env, encoding, rate, law
):
    env = await files_env()
    source = pcm16_sine(1.3, rate=rate, frequency=313.0, amplitude=0.7)
    data = pcm16_bytes(source) if law is None else g711_bytes(source, law)
    await recognize(
        env.plugin.servicer,
        recognize_request(data, encoding=encoding, sample_rate_hertz=rate),
        FakeServicerContext(),
    )
    (call,) = env.calls
    _, wav_samples = _wav_samples(call.audio_data)
    handed_to_core = float_from_pcm16(wav_samples)
    expected = _stream_side(encoding, rate, data)
    assert handed_to_core.shape == expected.shape
    assert np.array_equal(handed_to_core, expected)


# @spec ING-FILE-058
@pytest.mark.engine
def test_core_decoder_reads_the_wav_as_int16_over_32768():
    from vllm.multimodal.media.audio import load_audio

    from vllm_riva_frontend.frontend import encode_wav16

    values = np.concatenate(
        [pcm16_ramp(65_536), pcm16_sine(0.25, amplitude=0.99)]
    )
    wav = encode_wav16(float_from_pcm16(values))
    assert len(wav) == 44 + 2 * len(values)
    decoded, rate = load_audio(io.BytesIO(wav), sr=None, mono=True)
    assert int(rate) == 16_000
    expected = values.astype(np.float32) / 32768.0
    assert decoded.dtype == np.float32
    assert np.array_equal(decoded, expected)
