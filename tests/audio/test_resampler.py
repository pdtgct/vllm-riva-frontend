"""The pinned ``scipy-poly-v1`` resampler, segments and bounded slices.

The reference is ``scipy.signal.resample_poly(x, up=2, down=1)`` of the
whole segment; a segment ends at finalization (``drain``) or at a Realtime
``clear`` (``reset``).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.signal import resample_poly
from support.audio import pcm16_bytes, pcm16_sine

BIG = 1 << 30


def _frontend(encoding: str = "LINEAR_PCM", rate: int = 8_000):
    from vllm_riva_frontend.frontend import StreamingAudioFrontend

    return StreamingAudioFrontend(encoding=encoding, sample_rate_hz=rate)


def _concat(pieces) -> np.ndarray:
    pieces = [np.asarray(p) for p in pieces]
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(pieces)


def _stream(frontend, data: bytes, cuts, max_samples: int = BIG):
    """Decode ``data`` split at ``cuts``; returns the pieces."""
    pieces = []
    start = 0
    for cut in [*cuts, len(data)]:
        pieces.extend(frontend.decode(data[start:cut], max_samples=max_samples))
        start = cut
    return pieces


def _reference(samples: np.ndarray) -> np.ndarray:
    return resample_poly(samples.astype(np.float64) / 32768.0, up=2, down=1)


def _signal(seconds: float = 0.25) -> np.ndarray:
    rng = np.random.default_rng(1234)
    tone = pcm16_sine(seconds, rate=8_000, frequency=700.0, amplitude=0.4)
    noise = rng.integers(-3000, 3000, size=tone.size)
    return np.clip(tone.astype(np.int64) + noise, -32768, 32767).astype(
        np.int16
    )


# @spec ING-FE-017
def test_the_resampler_identifier_is_pinned() -> None:
    from vllm_riva_frontend.frontend import RESAMPLER_ID

    assert RESAMPLER_ID == "scipy-poly-v1"


def _values(provenance) -> list:
    if dataclasses.is_dataclass(provenance):
        return list(dataclasses.asdict(provenance).values())
    if hasattr(provenance, "_asdict"):
        return list(provenance._asdict().values())
    if hasattr(provenance, "model_dump"):
        return list(provenance.model_dump().values())
    return list(vars(provenance).values())


# @spec ING-FE-017
def test_the_resampler_identifier_is_published_in_provenance() -> None:
    from vllm_riva_frontend.config import (
        build_deployment_provenance,
        parse_config,
    )

    provenance = build_deployment_provenance(parse_config({}))
    assert "scipy-poly-v1" in _values(provenance)


# @spec ING-FE-021, ING-FE-017
@pytest.mark.parametrize("seed", range(6))
def test_any_split_ending_at_finalization_equals_resample_poly(seed) -> None:
    samples = _signal()
    data = pcm16_bytes(samples)
    rng = np.random.default_rng(seed)
    cuts = sorted(rng.choice(np.arange(1, len(data)), size=7, replace=False))
    frontend = _frontend()
    out = _concat([*_stream(frontend, data, cuts), frontend.drain()])
    expected = _reference(samples)
    assert out.shape == expected.shape == (2 * samples.size,)
    np.testing.assert_allclose(out, expected, atol=1e-6)


# @spec ING-FE-021, ING-FE-010
def test_the_lookahead_is_drained_once_at_finalization() -> None:
    samples = _signal()
    frontend = _frontend()
    before = _concat(frontend.decode(pcm16_bytes(samples), max_samples=BIG))
    tail = np.asarray(frontend.drain())
    # the filter's 20-sample lookahead at the 8 kHz input is 20 output
    # samples held back until finalization
    assert before.size + tail.size == 2 * samples.size
    assert tail.size > 0
    assert np.asarray(frontend.drain()).size == 0  # once


# @spec ING-FE-021
def test_sample_by_sample_feeding_still_equals_the_reference() -> None:
    samples = _signal(0.02)
    data = pcm16_bytes(samples)
    frontend = _frontend()
    out = _concat(
        [*_stream(frontend, data, range(1, len(data))), frontend.drain()]
    )
    np.testing.assert_allclose(out, _reference(samples), atol=1e-6)


# @spec ING-FE-022
def test_output_does_not_depend_on_the_split() -> None:
    samples = _signal()
    data = pcm16_bytes(samples)
    a = _frontend()
    b = _frontend()
    out_a = _concat([*_stream(a, data, [2, 400, 401, 1000]), a.drain()])
    out_b = _concat([*_stream(b, data, [999, 3001]), b.drain()])
    np.testing.assert_allclose(out_a, out_b, atol=1e-6)


# @spec ING-FE-022, ING-FE-021
def test_clear_ends_a_segment_with_a_prefix_and_restarts_the_filter() -> None:
    first = _signal(0.1)
    second = pcm16_sine(0.15, rate=8_000, frequency=300.0)
    frontend = _frontend()
    head = _concat(frontend.decode(pcm16_bytes(first), max_samples=BIG))
    frontend.reset()  # Realtime ``clear``
    reference_first = _reference(first)
    assert head.size <= reference_first.size
    np.testing.assert_allclose(head, reference_first[: head.size], atol=1e-6)
    out = _concat(
        [
            *frontend.decode(pcm16_bytes(second), max_samples=BIG),
            frontend.drain(),
        ]
    )
    # a fresh segment: as if the stream had started at ``second``
    np.testing.assert_allclose(out, _reference(second), atol=1e-6)


# @spec ING-FE-022, ING-FE-020
def test_clear_discards_the_byte_residual() -> None:
    frontend = _frontend(rate=16_000)
    list(frontend.decode(b"\x01", max_samples=BIG))
    assert frontend.residual_bytes == 1
    frontend.reset()
    assert frontend.residual_bytes == 0
    out = _concat(frontend.decode(b"\x02\x00", max_samples=BIG))
    np.testing.assert_array_equal(out, np.array([2 / 32768], np.float32))


# @spec ING-FE-011
@pytest.mark.parametrize(("rate", "limit"), [(16_000, 100), (8_000, 100)])
def test_a_message_is_decoded_in_slices_of_at_most_the_bound(
    rate, limit
) -> None:
    samples = pcm16_sine(0.1, rate=rate)
    data = pcm16_bytes(samples)
    sliced_fe = _frontend(rate=rate)
    whole_fe = _frontend(rate=rate)
    slices = list(sliced_fe.decode(data, max_samples=limit))
    whole = _concat(whole_fe.decode(data, max_samples=BIG))
    assert len(slices) > 1
    assert all(0 < np.asarray(s).size <= limit for s in slices)
    np.testing.assert_allclose(_concat(slices), whole, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(sliced_fe.drain()), np.asarray(whole_fe.drain()), atol=1e-6
    )


# @spec ING-FE-011, ING-FE-020
def test_residual_and_filter_state_carry_across_slices_and_messages() -> None:
    samples = _signal()
    data = pcm16_bytes(samples)
    frontend = _frontend()
    pieces = _stream(frontend, data, [17, 501, 502], max_samples=37)
    assert all(np.asarray(p).size <= 37 for p in pieces)
    out = _concat([*pieces, frontend.drain()])
    np.testing.assert_allclose(out, _reference(samples), atol=1e-6)


# @spec ING-FE-010
def test_finalization_discards_a_partial_sample() -> None:
    frontend = _frontend(rate=16_000)
    out = _concat(frontend.decode(b"\x01\x00\x02", max_samples=BIG))
    np.testing.assert_array_equal(out, np.array([1 / 32768], np.float32))
    tail = np.asarray(frontend.drain())
    assert tail.size == 0
    assert frontend.residual_bytes == 0
