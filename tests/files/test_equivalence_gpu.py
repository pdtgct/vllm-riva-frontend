"""A complete file's transcript equals a factory session's (ING-FILE-056).

It needs a CUDA device, the core tree, the model plugin installed (its
``vllm.general_plugins`` and session-factory entry points), the published
card named by ``NEMOTRON_CARD_DIR`` and a 16 kHz mono 16-bit speech WAV named
by ``NEMOTRON_TEST_AUDIO``: silence would make both transcripts empty and the
comparison vacuous.

Matched conditions (files design, Equivalence): one in-process engine serves
both paths, so precision is the same, and endpointing is off on both. The
file runs at the engine's default lookahead, ``num_lookahead_tokens`` (13 in
the reference deployment, the 1120 ms cadence), and the comparison session
opens at that cadence, not at the 560 ms streaming arm. Both use one fixed
locale (no ``update_locale``) and the same samples: the input is PCM16, so
the ``Recognize`` conversion's requantization is exact and the stream side
is fed the same values. The file goes through the frontend's ``Recognize``
into core's own transcription service; the session through the factory the
frontend discovered.
"""

from __future__ import annotations

import os
import wave

import numpy as np
import pytest
from support.app import serve_plugin
from support.audio import float_from_pcm16, pcm16_bytes
from support.grpc_harness import free_port
from support.riva import recognize, recognize_request

pytestmark = [pytest.mark.gpu, pytest.mark.real_factory, pytest.mark.engine]

FILE_RIGHT_CONTEXT = 13
FILE_CADENCE = "1120ms"
STREAM_RIGHT_CONTEXT = 6
PIECE = 8_000


def _speech() -> np.ndarray:
    path = os.environ.get("NEMOTRON_TEST_AUDIO")
    if not path:
        pytest.skip("NEMOTRON_TEST_AUDIO (a speech WAV) is not set")
    with wave.open(path, "rb") as wav:
        assert wav.getframerate() == 16_000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        data = wav.readframes(wav.getnframes())
    return np.frombuffer(data, dtype="<i2").copy()


@pytest.fixture
async def engine():
    """An in-process engine on the published card at the file cadence."""
    card = os.environ.get("NEMOTRON_CARD_DIR")
    if not card:
        pytest.skip("NEMOTRON_CARD_DIR is not set")
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm_nemotron_asr_streaming import register

    register()
    args = AsyncEngineArgs(
        model=card,
        config_format="nemotron_asr_streaming",
        tokenizer_mode="nemotron_asr_streaming",
        hf_overrides={
            "num_lookahead_tokens": FILE_RIGHT_CONTEXT,
            "supported_num_lookahead_tokens": [
                STREAM_RIGHT_CONTEXT,
                FILE_RIGHT_CONTEXT,
            ],
        },
        async_scheduling=False,
        enable_prefix_caching=False,
        enforce_eager=True,
        dtype="float32",
        max_model_len=16_384,
        max_num_seqs=2,
        gpu_memory_utilization=0.3,
        additional_config={
            "riva_frontend": {"grpc_bind": f"127.0.0.1:{free_port()}"}
        },
    )
    llm = AsyncLLM.from_engine_args(args)
    try:
        yield llm
    finally:
        llm.shutdown()


async def _session_transcript(factory, samples, locale: str) -> str:
    lease = await factory.open(cadence=FILE_CADENCE, locale=locale)
    try:
        audio = float_from_pcm16(samples)
        for start in range(0, audio.size, PIECE):
            await lease.feed(
                audio[start : start + PIECE], on_accepted=lambda n: None
            )
        text = await lease.flush()
        await lease.finish()
    finally:
        await lease.release()
    return text


# @spec ING-FILE-056
async def test_a_file_transcript_equals_a_session_at_the_file_cadence(engine):
    from vllm.entrypoints.openai.models.protocol import BaseModelPath
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels
    from vllm.entrypoints.speech_to_text.transcription.serving import (
        OpenAIServingTranscription,
    )

    from vllm_riva_frontend.lifecycle import RivaFrontendPlugin

    samples = _speech()
    name = engine.model_config.served_model_name
    models = OpenAIServingModels(
        engine, [BaseModelPath(name=name, model_path=engine.model_config.model)]
    )
    service = OpenAIServingTranscription(engine, models, request_logger=None)
    served = await serve_plugin(
        RivaFrontendPlugin(), engine, models=models, transcription=service
    )
    try:
        factory = served.factory
        assert FILE_CADENCE in factory.served_cadences
        locale = factory.default_locale or min(factory.locales)
        outcome = await recognize(
            served.servicer,
            recognize_request(pcm16_bytes(samples), language_code=locale),
        )
        assert outcome.abort is None, outcome.details
        assert outcome.error is None, outcome.error
        session_text = await _session_transcript(factory, samples, locale)
    finally:
        await served.stop()
    assert outcome.transcript, "speech must give a non-empty transcript"
    assert outcome.transcript == session_text
