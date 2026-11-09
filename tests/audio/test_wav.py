"""The in-memory WAV ``Recognize`` hands core, and the requantization."""

from __future__ import annotations

import builtins
import io
import os
import struct
import tempfile

import grpc
import numpy as np
import pytest
import soundfile
from support.audio import pcm16_bytes, pcm16_ramp, pcm16_sine
from support.riva import recognize, recognize_request


def _pcm16(x):
    from vllm_riva_frontend.frontend import pcm16

    return pcm16(np.asarray(x, dtype=np.float32))


# @spec ING-FE-012
def test_requantization_rounds_half_to_even_and_clips() -> None:
    scaled = np.array(
        [0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 100.49, 32767.5, 4e4, -4e4, -32768.4]
    )
    out = _pcm16(scaled / 32768.0)
    assert out.dtype == np.int16
    np.testing.assert_array_equal(
        out, [0, 2, 2, 0, -2, -2, 100, 32767, 32767, -32768, -32768]
    )


# @spec ING-FE-012
def test_requantization_is_exact_for_16_bit_values() -> None:
    from vllm_riva_frontend.frontend import requantize

    values = pcm16_ramp(65_536)
    x = values.astype(np.float32) / np.float32(32768.0)
    np.testing.assert_array_equal(_pcm16(x), values)
    out = requantize(x)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, x)


# @spec ING-FE-012
def test_the_stream_side_helper_is_pcm16_over_32768() -> None:
    from vllm_riva_frontend.frontend import requantize

    rng = np.random.default_rng(7)
    x = rng.uniform(-1.2, 1.2, size=4096).astype(np.float32)
    np.testing.assert_array_equal(
        requantize(x), (_pcm16(x).astype(np.float32) / np.float32(32768.0))
    )


# @spec ING-FE-012
def test_stream_decoding_does_not_requantize() -> None:
    from vllm_riva_frontend.frontend import StreamingAudioFrontend, requantize

    frontend = StreamingAudioFrontend(
        encoding="LINEAR_PCM", sample_rate_hz=8_000
    )
    data = pcm16_bytes(pcm16_sine(0.1, rate=8_000))
    out = np.concatenate(
        [*frontend.decode(data, max_samples=1 << 30), frontend.drain()]
    )
    # resampled 8 kHz audio is not on the 1/32768 grid; a live stream is
    # handed the resampler's output as is
    assert np.any(requantize(out) != out)


def _parse_wav(blob: bytes) -> dict:
    riff, size, wave = struct.unpack("<4sI4s", blob[:12])
    fmt, fmt_size = struct.unpack("<4sI", blob[12:20])
    code, channels, rate, byte_rate, align, bits = struct.unpack(
        "<HHIIHH", blob[20:36]
    )
    data, data_size = struct.unpack("<4sI", blob[36:44])
    return {
        "riff": riff,
        "size": size,
        "wave": wave,
        "fmt": fmt,
        "fmt_size": fmt_size,
        "code": code,
        "channels": channels,
        "rate": rate,
        "byte_rate": byte_rate,
        "align": align,
        "bits": bits,
        "data": data,
        "data_size": data_size,
    }


# @spec ING-FE-008
def test_wav_has_the_canonical_44_byte_header() -> None:
    from vllm_riva_frontend.frontend import encode_wav16

    samples = pcm16_sine(0.1).astype(np.float32) / 32768.0
    blob = encode_wav16(samples)
    header = _parse_wav(blob)
    assert header == {
        "riff": b"RIFF",
        "size": 36 + 2 * samples.size,
        "wave": b"WAVE",
        "fmt": b"fmt ",
        "fmt_size": 16,
        "code": 1,
        "channels": 1,
        "rate": 16_000,
        "byte_rate": 32_000,
        "align": 2,
        "bits": 16,
        "data": b"data",
        "data_size": 2 * samples.size,
    }
    assert len(blob) == 44 + 2 * samples.size
    assert blob[44:] == _pcm16(samples).astype("<i2").tobytes()


# @spec ING-FE-008, ING-FE-012
def test_wav_reads_back_as_the_requantized_samples() -> None:
    from vllm_riva_frontend.frontend import encode_wav16, requantize

    rng = np.random.default_rng(3)
    samples = rng.uniform(-0.9, 0.9, size=3200).astype(np.float32)
    data, rate = soundfile.read(
        io.BytesIO(encode_wav16(samples)), dtype="int16"
    )
    assert rate == 16_000
    np.testing.assert_array_equal(data, _pcm16(samples))
    np.testing.assert_array_equal(
        data.astype(np.float32) / np.float32(32768.0), requantize(samples)
    )


# @spec ING-FE-008
def test_wav_encoding_touches_no_filesystem(monkeypatch) -> None:
    from vllm_riva_frontend.frontend import encode_wav16

    def refuse(*args, **kwargs):
        raise AssertionError("filesystem use")

    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(io, "open", refuse)
    monkeypatch.setattr(os, "open", refuse)
    for name in ("NamedTemporaryFile", "TemporaryFile", "mkstemp", "mkdtemp"):
        monkeypatch.setattr(tempfile, name, refuse)
    blob = encode_wav16(np.zeros(160, dtype=np.float32))
    assert blob[:4] == b"RIFF"


# @spec ING-FE-008
def test_empty_audio_is_a_valid_empty_wav() -> None:
    from vllm_riva_frontend.frontend import encode_wav16

    blob = encode_wav16(np.zeros(0, dtype=np.float32))
    assert len(blob) == 44
    assert _parse_wav(blob)["data_size"] == 0


# ----------------------------------------------------- through Recognize


def _decoded(encoding: str, rate: int, data: bytes) -> np.ndarray:
    from vllm_riva_frontend.frontend import StreamingAudioFrontend

    frontend = StreamingAudioFrontend(encoding=encoding, sample_rate_hz=rate)
    return np.concatenate(
        [*frontend.decode(data, max_samples=1 << 30), frontend.drain()]
    )


# @spec ING-FE-008, ING-FE-012
@pytest.mark.parametrize(
    ("encoding", "rate", "law"),
    [
        ("LINEAR_PCM", 16_000, None),
        ("LINEAR_PCM", 8_000, None),
        ("MULAW", 8_000, "mulaw"),
        ("ALAW", 8_000, "alaw"),
    ],
)
async def test_recognize_hands_core_the_wav_of_the_stream_decoder(
    encoding, rate, law, started, fake_transcription
) -> None:
    from support.audio import g711_bytes

    from vllm_riva_frontend.frontend import encode_wav16

    samples = pcm16_sine(0.2, rate=rate)
    data = pcm16_bytes(samples) if law is None else g711_bytes(samples, law)
    outcome = await recognize(
        started.servicer,
        recognize_request(data, encoding=encoding, sample_rate_hertz=rate),
    )
    assert outcome.status is grpc.StatusCode.OK, outcome.details
    assert outcome.response is not None
    assert len(fake_transcription.calls) == 1
    wav = fake_transcription.calls[0].audio_data
    assert wav == encode_wav16(_decoded(encoding, rate, data))
    if encoding == "LINEAR_PCM" and rate == 16_000:
        assert wav[44:] == data  # 16 kHz PCM round-trips exactly
