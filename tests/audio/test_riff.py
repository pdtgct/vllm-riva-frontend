"""RIFF/WAVE header inference (``frontend.sniff_riff``).

The streaming surfaces resolve an undeclared format from the header at the
head of the audio; ``sniff_riff`` is the shared parser: a ``RiffFormat`` for
every parseable header (an off-matrix cell included), ``None`` while more
bytes are needed, a ``FormatError`` otherwise. These tests own the parser's
results; what each surface then answers on its wire is the streaming leaf's.
"""

from __future__ import annotations

import pytest
from support.audio import (
    DATA_LENGTH_UNKNOWN,
    pcm16_bytes,
    pcm16_sine,
    wav_bytes,
    wav_header_size,
)

BOUND = 1 << 20


def _sniff(buffer: bytes, bound: int = BOUND):
    from vllm_riva_frontend.frontend import sniff_riff

    return sniff_riff(buffer, max_header_bytes=bound)


def _riff_format():
    from vllm_riva_frontend.frontend import RiffFormat

    return RiffFormat


def _format_error():
    from vllm_riva_frontend.frontend import FormatError

    return FormatError


# @spec ING-GRPC-029, ING-NIMWS-009
def test_data_offset_skips_ancillary_chunks() -> None:
    payload = pcm16_bytes(pcm16_sine(0.01))
    extra = [(b"LIST", b"INFOabc"), (b"JUNK", b"\x00" * 6)]
    riff = wav_bytes(payload, extra_chunks=extra)
    resolved = _sniff(riff)
    assert isinstance(resolved, _riff_format())
    assert resolved.encoding == "LINEAR_PCM"
    assert resolved.sample_rate_hz == 16_000
    assert resolved.channels == 1
    assert resolved.data_offset == wav_header_size(extra)
    assert riff[resolved.data_offset :] == payload


# @spec ING-GRPC-029
@pytest.mark.parametrize(
    ("encoding", "name", "rate"),
    [
        ("mulaw", "MULAW", 8_000),
        ("alaw", "ALAW", 8_000),
        ("pcm16", "LINEAR_PCM", 8_000),
    ],
)
def test_header_resolves_each_format_code(encoding, name, rate) -> None:
    resolved = _sniff(wav_bytes(b"\x00" * 16, encoding=encoding, rate=rate))
    assert isinstance(resolved, _riff_format())
    assert (resolved.encoding, resolved.sample_rate_hz) == (name, rate)


# @spec ING-GRPC-029, ING-NIMWS-009
@pytest.mark.parametrize("cut", [0, 3, 11, 20, 43])
def test_truncated_header_requests_more_bytes(cut) -> None:
    assert _sniff(wav_bytes(b"\x00\x00")[:cut]) is None


# @spec ING-GRPC-029, ING-NIMWS-009, ING-ERR-003
def test_non_riff_bytes_are_unsupported_format_naming_fields() -> None:
    error = _sniff(b"OggS not a wave file at all, sorry" * 2)
    assert isinstance(error, _format_error())
    assert error.fields


# @spec ING-GRPC-029, ING-ERR-003
@pytest.mark.parametrize(
    ("header", "cell", "field"),
    [
        ({"channels": 2}, ("LINEAR_PCM", 16_000, 2), "channels"),
        ({"rate": 44_100}, ("LINEAR_PCM", 44_100, 1), "sample_rate_hz"),
    ],
    ids=["stereo", "44k1"],
)
def test_an_off_matrix_header_parses_and_the_cell_check_refuses_it(
    header, cell, field
) -> None:
    from vllm_riva_frontend.frontend import validate_format

    # The parser reports what the header declares; refusing the cell is the
    # accept-matrix check's job, so every surface refuses it the same way.
    resolved = _sniff(wav_bytes(b"\x00" * 8, **header))
    assert isinstance(resolved, _riff_format()), resolved
    assert (
        resolved.encoding,
        resolved.sample_rate_hz,
        resolved.channels,
    ) == cell
    error = validate_format(*cell)
    assert error is not None
    assert error.fields == (field,)


# @spec ING-GRPC-029, ING-NIMWS-009
def test_header_bound_rejects_before_retaining_more_bytes() -> None:
    riff = wav_bytes(b"\x00\x00", extra_chunks=[(b"JUNK", b"\x00" * 64)])
    error = _sniff(riff[:40], bound=32)
    assert isinstance(error, _format_error())


# @spec ING-GRPC-029
def test_complete_header_resolves_even_when_payload_exceeds_the_bound() -> None:
    payload = b"\x00\x00" * 256
    riff = wav_bytes(payload)
    resolved = _sniff(riff, bound=64)
    assert isinstance(resolved, _riff_format())
    assert resolved.data_offset == 44


# @spec ING-GRPC-021
def test_explicit_data_length_is_reported() -> None:
    payload = pcm16_bytes(pcm16_sine(0.01))
    resolved = _sniff(wav_bytes(payload, data_length=100))
    assert isinstance(resolved, _riff_format())
    assert resolved.data_length == 100


# @spec ING-GRPC-021
@pytest.mark.parametrize("sentinel", [0, DATA_LENGTH_UNKNOWN])
def test_sentinel_data_length_means_to_the_end_of_the_stream(sentinel) -> None:
    payload = pcm16_bytes(pcm16_sine(0.01))
    resolved = _sniff(wav_bytes(payload, data_length=sentinel))
    assert isinstance(resolved, _riff_format())
    # an unknown length, not an explicit truncation at 0 or 4 GiB
    assert resolved.data_length is None
