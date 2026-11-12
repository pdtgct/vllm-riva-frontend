"""Decoders: ITU G.711 tables, little-endian PCM16, whole-sample carry.

``support.audio`` implements the public ITU-T G.711 reference expansion
independently of the frontend and is the oracle here.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import resample_poly
from support.audio import (
    alaw_table,
    g711_bytes,
    mulaw_table,
    pcm16_bytes,
    pcm16_ramp,
    pcm16_sine,
)

BIG = 1 << 30


def _frontend(encoding: str, rate: int):
    from vllm_riva_frontend.frontend import StreamingAudioFrontend

    return StreamingAudioFrontend(encoding=encoding, sample_rate_hz=rate)


def _decode(frontend, data: bytes, max_samples: int = BIG) -> list[np.ndarray]:
    return list(frontend.decode(data, max_samples=max_samples))


def _concat(pieces) -> np.ndarray:
    pieces = [np.asarray(p) for p in pieces]
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(pieces)


def _as_float(table) -> np.ndarray:
    table = np.asarray(table)
    if np.issubdtype(table.dtype, np.integer):
        return table.astype(np.float64) / 32768.0
    return table.astype(np.float64)


# @spec ING-FE-002, ING-FE-018
@pytest.mark.parametrize(
    ("name", "oracle"), [("mulaw", mulaw_table), ("alaw", alaw_table)]
)
def test_g711_tables_are_the_itu_expansion(name, oracle) -> None:
    from vllm_riva_frontend import frontend

    table = (frontend.mulaw_table if name == "mulaw" else frontend.alaw_table)()
    assert len(table) == 256
    np.testing.assert_array_equal(
        _as_float(table), oracle().astype(np.float64) / 32768.0
    )


# @spec ING-FE-002, ING-FE-018
def test_g711_anchor_values() -> None:
    from vllm_riva_frontend.frontend import alaw_table as fe_alaw
    from vllm_riva_frontend.frontend import mulaw_table as fe_mulaw

    assert _as_float(fe_mulaw())[0x00] == pytest.approx(-32124 / 32768)
    assert _as_float(fe_mulaw())[0xFF] == 0.0
    assert _as_float(fe_alaw())[0x55] == pytest.approx(-8 / 32768)
    assert _as_float(fe_alaw())[0xD5] == pytest.approx(8 / 32768)


# @spec ING-FE-002, ING-FE-018
def test_pcm16_is_little_endian_int16_over_32768() -> None:
    samples = np.array([0, 1, 256, -1, -256, 32767, -32768], dtype=np.int16)
    out = _concat(
        _decode(_frontend("LINEAR_PCM", 16_000), pcm16_bytes(samples))
    )
    expected = samples.astype(np.float32) / np.float32(32768.0)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, expected)


# @spec ING-FE-018
def test_decoded_samples_are_float32_in_minus_one_to_one() -> None:
    samples = pcm16_ramp(65_536)  # every int16 value
    out = _concat(
        _decode(_frontend("LINEAR_PCM", 16_000), pcm16_bytes(samples))
    )
    assert out.dtype == np.float32
    assert out.min() == -1.0
    assert out.max() < 1.0
    np.testing.assert_array_equal(out * 32768.0, samples.astype(np.float32))


# @spec ING-FE-002, ING-FE-018, ING-FE-017
@pytest.mark.parametrize(
    ("encoding", "law", "oracle"),
    [("MULAW", "mulaw", mulaw_table), ("ALAW", "alaw", alaw_table)],
)
def test_g711_stream_decodes_through_the_table_then_resamples(
    encoding, law, oracle
) -> None:
    codes = g711_bytes(pcm16_sine(0.05, rate=8_000), law)
    frontend = _frontend(encoding, 8_000)
    out = _concat([*_decode(frontend, codes), frontend.drain()])
    expanded = oracle()[np.frombuffer(codes, dtype=np.uint8)]
    expected = resample_poly(expanded.astype(np.float64) / 32768.0, 2, 1)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, expected, atol=1e-6)


# @spec ING-FE-020
def test_pcm_sample_split_across_messages_is_decoded_once() -> None:
    frontend = _frontend("LINEAR_PCM", 16_000)
    assert _concat(_decode(frontend, b"\x01")).size == 0
    assert frontend.residual_bytes == 1
    out = _concat(_decode(frontend, b"\x00"))
    np.testing.assert_array_equal(out, np.array([1 / 32768], dtype=np.float32))
    assert frontend.residual_bytes == 0


# @spec ING-FE-020
@pytest.mark.parametrize("cut", [1, 3, 5, 7, 999])
def test_any_byte_split_decodes_the_same_samples(cut) -> None:
    samples = pcm16_sine(0.1)
    data = pcm16_bytes(samples)
    frontend = _frontend("LINEAR_PCM", 16_000)
    out = _concat(
        [*_decode(frontend, data[:cut]), *_decode(frontend, data[cut:])]
    )
    np.testing.assert_array_equal(
        out, samples.astype(np.float32) / np.float32(32768.0)
    )


# @spec ING-FE-020
def test_g711_carries_no_residual() -> None:
    frontend = _frontend("MULAW", 8_000)
    _decode(frontend, b"\xff\xff\xff")
    assert frontend.residual_bytes == 0


# @spec ING-FE-009, ING-FE-018
@pytest.mark.parametrize(
    ("encoding", "rate"), [("LINEAR_PCM", 16_000), ("LINEAR_PCM", 8_000)]
)
def test_every_piece_is_a_whole_sample_float32_vector(encoding, rate) -> None:
    frontend = _frontend(encoding, rate)
    data = pcm16_bytes(pcm16_sine(0.2, rate=rate))
    pieces = [
        *_decode(frontend, data[:333], max_samples=500),
        *_decode(frontend, data[333:], max_samples=500),
        frontend.drain(),
    ]
    for piece in pieces:
        assert isinstance(piece, np.ndarray)
        assert piece.dtype == np.float32
        assert piece.ndim == 1
        assert piece.flags.c_contiguous
