"""Audio front-end contracts for the three RFC-2 compatibility surfaces."""

import numpy as np
import pytest
from scipy.signal import resample_poly

from vllm_riva_frontend.frontend import (
    ACCEPT_MATRIX,
    RESAMPLER_ID,
    FormatError,
    RiffFormat,
    StreamingAudioFrontend,
    alaw_table,
    mulaw_table,
    sniff_riff,
    validate_format,
)


def _riff_pcm16(*, payload: bytes, prefix: bytes = b"") -> bytes:
    """Build a PCM16 RIFF/WAVE fixture with an optional ancillary chunk."""
    fmt = (
        b"fmt "
        + (16).to_bytes(4, "little")
        + b"\x01\x00\x01\x00\x80>\x00\x00"
        + b"\x00\xfa\x00\x00\x02\x00"
        + b"\x10\x00"
    )
    body = b"WAVE" + fmt + prefix + b"data" + len(payload).to_bytes(4, "little")
    return b"RIFF" + len(body + payload).to_bytes(4, "little") + body + payload


# @spec ING-FE-001
def test_accept_matrix_is_exactly_the_four_supported_mono_cells() -> None:
    assert {
        ("LINEAR_PCM", 16000),
        ("LINEAR_PCM", 8000),
        ("MULAW", 8000),
        ("ALAW", 8000),
    } == ACCEPT_MATRIX


# @spec ING-FE-001
def test_off_matrix_format_names_the_invalid_fields() -> None:
    error = validate_format("MULAW", 16000)
    assert error is not None
    assert error.code == "unsupported_format"
    assert set(error.fields) == {"encoding", "sample_rate_hz"}


# @spec ING-FE-001
def test_non_mono_format_names_channels_even_when_codec_cell_is_valid() -> None:
    error = validate_format("LINEAR_PCM", 16000, channels=2)
    assert error is not None
    assert error.code == "unsupported_format"
    assert error.fields == ("channels",)


# @spec ING-FE-002
def test_g711_tables_use_public_itu_anchor_values() -> None:
    assert mulaw_table()[0x00] == pytest.approx(-32124 / 32768)
    assert mulaw_table()[0xFF] == 0.0
    assert alaw_table()[0x55] == pytest.approx(-8 / 32768)
    assert alaw_table()[0xD5] == pytest.approx(8 / 32768)


# @spec ING-FE-002
def test_g711_streaming_frontend_decodes_mulaw() -> None:
    frontend = StreamingAudioFrontend(encoding="MULAW", sample_rate_hz=8000)
    samples = np.concatenate((frontend.push(b"\xff"), frontend.flush()))
    assert samples.tolist() == [0.0, 0.0]


# @spec ING-FE-003
def test_pcm_sample_split_across_messages_waits_for_the_second_byte() -> None:
    frontend = StreamingAudioFrontend(
        encoding="LINEAR_PCM", sample_rate_hz=16000
    )
    assert frontend.push(b"\x01").tolist() == []
    assert frontend.push(b"\x00").tolist() == [1.0 / 32768.0]


# @spec ING-FE-002, ING-VEH-004
def test_every_frontend_piece_matches_the_rfc1_float32_array_boundary() -> None:
    frontend = StreamingAudioFrontend(
        encoding="LINEAR_PCM", sample_rate_hz=16000
    )
    for piece in (frontend.push(b"\x01\x00\x02\x00"), frontend.flush()):
        assert isinstance(piece, np.ndarray)
        assert piece.dtype == np.float32
        assert piece.ndim == 1
        assert piece.flags.c_contiguous


# @spec ING-FE-004
def test_8k_streaming_resample_equals_whole_signal_scipy_poly() -> None:
    samples = np.arange(-240, 240, dtype=np.int16)
    raw = samples.astype("<i2").tobytes()
    frontend = StreamingAudioFrontend(
        encoding="LINEAR_PCM", sample_rate_hz=8000
    )
    pieces = [
        frontend.push(raw[:7]),
        frontend.push(raw[7:318]),
        frontend.push(raw[318:]),
        frontend.flush(),
    ]
    streamed = np.asarray([item for piece in pieces for item in piece])
    expected = resample_poly(
        samples.astype(np.float32) / 32768.0, up=2, down=1
    ).astype(np.float32)
    assert RESAMPLER_ID == "scipy-poly-v1"
    assert frontend.resampler_identifier == RESAMPLER_ID
    np.testing.assert_allclose(streamed, expected, atol=1e-6)


# @spec ING-GRPC-007, ING-NIMWS-009
def test_riff_data_offset_skips_ancillary_chunks_and_metadata() -> None:
    junk = b"JUNK" + (3).to_bytes(4, "little") + b"abc\x00"
    payload = b"\x01\x00\x02\x00"
    riff = _riff_pcm16(payload=payload, prefix=junk)
    resolved = sniff_riff(riff)
    assert resolved == RiffFormat(
        encoding="LINEAR_PCM",
        sample_rate_hz=16000,
        channels=1,
        data_offset=len(riff) - len(payload),
        data_bytes=len(payload),
    )
    assert riff[resolved.data_offset :] == payload


# @spec ING-GRPC-007, ING-NIMWS-009
@pytest.mark.parametrize("cut", [0, 3, 11, 20, 43])
def test_riff_truncated_prefix_requests_more_bytes(cut: int) -> None:
    assert sniff_riff(_riff_pcm16(payload=b"\x00\x00")[:cut]) is None


# @spec ING-GRPC-007, ING-NIMWS-009
def test_riff_malformed_or_unsupported_header_is_stable() -> None:
    malformed = sniff_riff(b"OggS not a wave")
    assert isinstance(malformed, FormatError)
    assert malformed.code == "unsupported_format"
    assert malformed.fields == ("encoding", "sample_rate_hz")

    pcm8 = bytearray(_riff_pcm16(payload=b"\x00"))
    pcm8[34:36] = b"\x08\x00"
    unsupported = sniff_riff(bytes(pcm8))
    assert isinstance(unsupported, FormatError)
    assert unsupported.code == "unsupported_format"


# @spec ING-GRPC-007, ING-NIMWS-009
def test_riff_header_bound_rejects_before_retaining_more_bytes() -> None:
    rejection = sniff_riff(b"RIFF" + b"x" * 9, max_header_bytes=8)
    assert isinstance(rejection, FormatError)
    assert rejection.code == "request_too_large"
    assert rejection.fields == ("riff_header",)


# @spec ING-GRPC-007, ING-NIMWS-009
def test_complete_header_resolves_even_when_payload_exceeds_scan_bound() -> (
    None
):
    payload = b"\x00\x00" * 256
    riff = _riff_pcm16(payload=payload)
    resolved = sniff_riff(riff, max_header_bytes=64)
    assert isinstance(resolved, RiffFormat)
    assert resolved.data_offset == len(riff) - len(payload)
    assert resolved.data_bytes == len(payload)
