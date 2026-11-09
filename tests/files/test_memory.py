"""Per-owner frontend audio memory on both file paths (``tracemalloc``)."""

from __future__ import annotations

import gc
import tracemalloc

import pytest
from support.audio import pcm16_bytes, pcm16_sine
from support.grpc_harness import FakeServicerContext
from support.http import MiB, multipart_body
from support.riva import recognize, recognize_request

ACCEPTANCE = 128 * MiB
CORE_UPLOAD_LIMIT = 25 * MiB  # VLLM_MAX_AUDIO_CLIP_FILESIZE_MB default


def _measure_start() -> int:
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    return tracemalloc.get_traced_memory()[0]


def _measure_end(baseline: int) -> int:
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak - baseline


# @spec ING-FILE-043
async def test_route_peak_memory_is_within_the_bound(files_env):
    env = await files_env()
    envelope = env.plugin.config.http_multipart_envelope_max_bytes
    payload = b"\x5a" * CORE_UPLOAD_LIMIT
    body = multipart_body([("model", env.models.model_name())], file=payload)
    exchange = env.request(body=body)
    del payload, body

    baseline = _measure_start()
    try:
        await exchange.finish(timeout=120)
    finally:
        grown = _measure_end(baseline)
    assert exchange.status == 200, exchange.body[:200]
    assert len(env.calls[0].audio_data) == CORE_UPLOAD_LIMIT
    assert grown <= 3 * envelope, grown
    assert grown < ACCEPTANCE, grown


# @spec ING-FILE-043
@pytest.mark.parametrize("rate", [16_000, 8_000], ids=["16k", "8k-resampled"])
async def test_recognize_peak_memory_is_within_the_bound(files_env, rate):
    env = await files_env()
    config = env.plugin.config
    seconds = config.unary_max_decoded_duration_seconds
    audio = pcm16_bytes(pcm16_sine(seconds, rate=rate, amplitude=0.3))
    request = recognize_request(audio, sample_rate_hertz=rate)
    del audio

    baseline = _measure_start()
    try:
        outcome = await recognize(
            env.plugin.servicer, request, FakeServicerContext()
        )
    finally:
        grown = _measure_end(baseline)
    assert outcome.abort is None, outcome.details
    bound = (
        config.unary_max_encoded_audio_bytes
        + 6 * 16_000 * seconds
        + 44
        + 4 * MiB
    )
    assert grown <= bound, grown
    assert grown < ACCEPTANCE, grown
