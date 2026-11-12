"""Deterministic audio fixtures generated in code (no audio files).

PCM16 is little-endian ``int16``. The G.711 codecs follow ITU-T G.711 (the
public reference algorithm: 14-bit segment encoding for mu-law with bias
0x84, 13-bit for A-law with the 0x55 toggle); ``mulaw_decode`` and
``alaw_decode`` are the inverse expansions, so a test can check a frontend
decoder against them. ``wav_bytes`` writes a RIFF/WAVE container with a
canonical 16-byte ``fmt `` chunk, optional extra chunks before ``data`` and
a ``data`` length that may be the explicit size or one of the sentinels
(``0`` and ``0xFFFFFFFF``) a streaming writer uses.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable
from functools import cache

import numpy as np

RATE_16K = 16_000
RATE_8K = 8_000

WAV_FORMAT_CODES = {"pcm16": 1, "alaw": 6, "mulaw": 7}
WAV_BITS = {"pcm16": 16, "alaw": 8, "mulaw": 8}

# ------------------------------------------------------------------ signals


def pcm16_sine(
    seconds: float,
    *,
    rate: int = RATE_16K,
    frequency: float = 440.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    """A sine tone as ``int16`` samples."""
    count = int(round(seconds * rate))
    t = np.arange(count, dtype=np.float64) / rate
    wave = np.sin(2.0 * np.pi * frequency * t) * amplitude * 32767.0
    return np.rint(wave).astype(np.int16)


def pcm16_ramp(count: int, *, start: int = -32768, step: int = 1) -> np.ndarray:
    """``count`` consecutive ``int16`` values from ``start``."""
    values = start + step * np.arange(count, dtype=np.int64)
    return np.clip(values, -32768, 32767).astype(np.int16)


def silence_pcm16(seconds: float, *, rate: int = RATE_16K) -> np.ndarray:
    return np.zeros(int(round(seconds * rate)), dtype=np.int16)


def pcm16_bytes(samples: np.ndarray) -> bytes:
    """Little-endian bytes of ``int16`` samples."""
    return np.asarray(samples, dtype="<i2").tobytes()


def pcm16_from_bytes(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2")


def float_from_pcm16(samples: np.ndarray) -> np.ndarray:
    """The ``float32`` form the frontend produces: ``int16 / 32768``."""
    return (np.asarray(samples, dtype=np.float32) / 32768.0).astype(np.float32)


# -------------------------------------------------------------------- G.711

_BIAS = 0x84
_CLIP = 8159
_SEG_UEND = (0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF)
_SEG_AEND = (0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF)


def _segment(value: int, table: tuple[int, ...]) -> int:
    for index, end in enumerate(table):
        if value <= end:
            return index
    return len(table)


def _linear_to_ulaw(pcm: int) -> int:
    pcm >>= 2
    if pcm < 0:
        pcm = -pcm
        mask = 0x7F
    else:
        mask = 0xFF
    pcm = min(pcm, _CLIP)
    pcm += _BIAS >> 2
    seg = _segment(pcm, _SEG_UEND)
    if seg >= 8:
        return 0x7F ^ mask
    return ((seg << 4) | ((pcm >> (seg + 1)) & 0x0F)) ^ mask


def _ulaw_to_linear(code: int) -> int:
    code = ~code & 0xFF
    t = ((code & 0x0F) << 3) + _BIAS
    t <<= (code & 0x70) >> 4
    return (_BIAS - t) if (code & 0x80) else (t - _BIAS)


def _linear_to_alaw(pcm: int) -> int:
    pcm >>= 3
    if pcm >= 0:
        mask = 0xD5
    else:
        mask = 0x55
        pcm = -pcm - 1
    seg = _segment(pcm, _SEG_AEND)
    if seg >= 8:
        return 0x7F ^ mask
    value = seg << 4
    if seg < 2:
        value |= (pcm >> 1) & 0x0F
    else:
        value |= (pcm >> seg) & 0x0F
    return value ^ mask


def _alaw_to_linear(code: int) -> int:
    code ^= 0x55
    t = (code & 0x0F) << 4
    seg = (code & 0x70) >> 4
    if seg == 0:
        t += 8
    elif seg == 1:
        t += 0x108
    else:
        t += 0x108
        t <<= seg - 1
    return t if (code & 0x80) else -t


@cache
def _encode_table(law: str) -> np.ndarray:
    encode = _linear_to_ulaw if law == "mulaw" else _linear_to_alaw
    table = np.empty(65536, dtype=np.uint8)
    for value in range(-32768, 32768):
        table[value + 32768] = encode(value)
    return table


@cache
def _decode_table(law: str) -> np.ndarray:
    decode = _ulaw_to_linear if law == "mulaw" else _alaw_to_linear
    return np.array([decode(code) for code in range(256)], dtype=np.int16)


def mulaw_encode(samples: np.ndarray) -> np.ndarray:
    """``int16`` samples to mu-law codes (``uint8``)."""
    index = np.asarray(samples, dtype=np.int32) + 32768
    return _encode_table("mulaw")[index]


def mulaw_decode(codes: np.ndarray | bytes) -> np.ndarray:
    """Mu-law codes to ``int16`` samples (the ITU expansion)."""
    return _decode_table("mulaw")[np.frombuffer(bytes(codes), dtype=np.uint8)]


def alaw_encode(samples: np.ndarray) -> np.ndarray:
    index = np.asarray(samples, dtype=np.int32) + 32768
    return _encode_table("alaw")[index]


def alaw_decode(codes: np.ndarray | bytes) -> np.ndarray:
    return _decode_table("alaw")[np.frombuffer(bytes(codes), dtype=np.uint8)]


def mulaw_table() -> np.ndarray:
    """The 256-entry mu-law expansion table as ``int16``."""
    return _decode_table("mulaw").copy()


def alaw_table() -> np.ndarray:
    return _decode_table("alaw").copy()


def g711_bytes(samples: np.ndarray, law: str) -> bytes:
    """Encode ``int16`` samples as ``"mulaw"`` or ``"alaw"`` bytes."""
    encoder = mulaw_encode if law == "mulaw" else alaw_encode
    return encoder(samples).tobytes()


# ---------------------------------------------------------------------- WAV

DATA_LENGTH_UNKNOWN = 0xFFFFFFFF


def wav_bytes(
    payload: bytes,
    *,
    encoding: str = "pcm16",
    rate: int = RATE_16K,
    channels: int = 1,
    data_length: int | None = None,
    extra_chunks: Iterable[tuple[bytes, bytes]] = (),
    riff_size: int | None = None,
) -> bytes:
    """A RIFF/WAVE container around already-encoded ``payload`` bytes.

    ``encoding`` is ``pcm16``, ``mulaw`` or ``alaw``. ``data_length`` is the
    ``data`` chunk's declared length (``None`` means ``len(payload)``; ``0``
    and ``DATA_LENGTH_UNKNOWN`` are the sentinels). ``extra_chunks`` are
    written between ``fmt `` and ``data`` (for example ``(b"LIST", b"...")``).
    """
    code = WAV_FORMAT_CODES[encoding]
    bits = WAV_BITS[encoding]
    block_align = channels * bits // 8
    fmt = struct.pack(
        "<HHIIHH", code, channels, rate, rate * block_align, block_align, bits
    )
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    for tag, chunk in extra_chunks:
        body += tag + struct.pack("<I", len(chunk)) + chunk
        if len(chunk) % 2:
            body += b"\x00"
    length = len(payload) if data_length is None else data_length
    body += b"data" + struct.pack("<I", length) + payload
    size = (len(body)) if riff_size is None else riff_size
    return b"RIFF" + struct.pack("<I", size) + body


def wav_pcm16(
    samples: np.ndarray, *, rate: int = RATE_16K, **kwargs: object
) -> bytes:
    """A 16-bit PCM WAV of ``int16`` samples."""
    return wav_bytes(
        pcm16_bytes(samples), encoding="pcm16", rate=rate, **kwargs
    )


def wav_header_size(extra_chunks: Iterable[tuple[bytes, bytes]] = ()) -> int:
    """The byte offset of the first sample in ``wav_bytes`` output."""
    size = 44
    for _, chunk in extra_chunks:
        size += 8 + len(chunk) + (len(chunk) % 2)
    return size
