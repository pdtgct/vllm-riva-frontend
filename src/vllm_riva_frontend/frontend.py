"""Bounded audio conversion shared by every RFC-2 compatibility surface.

The transport adapters validate their own wire declarations, then hand the
resolved cell to :class:`StreamingAudioFrontend`.  This module deliberately
does not know about a transport, lease, or request lifecycle: it turns one
accepted byte stream into 16 kHz mono samples and reports only stable format
facts/errors.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import firwin, lfilter

FloatSamples = np.ndarray

ACCEPT_MATRIX = frozenset(
    {
        ("LINEAR_PCM", 16000),
        ("LINEAR_PCM", 8000),
        ("MULAW", 8000),
        ("ALAW", 8000),
    }
)
RESAMPLER_ID = "scipy-poly-v1"


@dataclass(frozen=True)
class RiffFormat:
    """Format facts resolved from a RIFF/WAVE header."""

    encoding: str
    sample_rate_hz: int
    channels: int
    data_offset: int
    data_bytes: int


@dataclass(frozen=True)
class FormatError:
    """Stable format rejection with the fields a caller should name."""

    code: str
    fields: tuple[str, ...]
    detail: str = ""


def _unsupported_format(*fields: str, detail: str = "") -> FormatError:
    """Create the common explicit off-matrix rejection."""
    return FormatError("unsupported_format", fields, detail)


# @spec ING-FE-001
def validate_format(
    encoding: str, sample_rate_hz: int, channels: int = 1
) -> FormatError | None:
    """Validate exactly one supported mono format cell.

    The matrix is intentionally small: it makes the audio contract
    deterministic across Riva gRPC, NIM WebSocket, and NIM HTTP.  G.711 is
    8 kHz only, so a false 16 kHz declaration is rejected rather than
    resampled into silently corrupt audio.
    """
    fields: list[str] = []
    if (encoding, sample_rate_hz) not in ACCEPT_MATRIX:
        fields.extend(("encoding", "sample_rate_hz"))
    if channels != 1:
        fields.append("channels")
    if not fields:
        return None
    return _unsupported_format(
        *fields,
        detail=(
            f"({encoding}, {sample_rate_hz} Hz, {channels} ch) is outside "
            "the supported mono accept-matrix"
        ),
    )


_WAV_FORMAT_CODES = {1: "LINEAR_PCM", 6: "ALAW", 7: "MULAW"}
_WAV_FORMAT_BITS = {"LINEAR_PCM": 16, "MULAW": 8, "ALAW": 8}


def _riff_error(detail: str) -> FormatError:
    """Return the deferred-RIFF dialect's stable rejection shape."""
    return _unsupported_format("encoding", "sample_rate_hz", detail=detail)


# @spec ING-GRPC-007, ING-NIMWS-009
def sniff_riff(
    header: bytes, *, max_header_bytes: int | None = None
) -> RiffFormat | FormatError | None:
    """Resolve a bounded RIFF/WAVE header without treating it as audio.

    ``None`` means that the supplied prefix could still form a valid header;
    callers retain at most their configured header bound and retry after more
    input.  A returned :class:`RiffFormat` carries the byte offset of the
    data chunk so the caller never decodes RIFF metadata as PCM.  This parser
    does not allocate based on untrusted chunk sizes.
    """
    limit_reached = False
    if max_header_bytes is not None:
        if type(max_header_bytes) is not int or max_header_bytes <= 0:
            raise ValueError("max_header_bytes must be a positive integer")
        limit_reached = len(header) >= max_header_bytes
        header = header[:max_header_bytes]

    def need_more() -> FormatError | None:
        if limit_reached:
            return FormatError(
                "request_too_large",
                ("riff_header",),
                "RIFF/WAVE header exceeds max_riff_header_bytes",
            )
        return None

    if header[: min(4, len(header))] != b"RIFF"[: min(4, len(header))]:
        return _riff_error("no parseable RIFF/WAVE header at audio head")
    if len(header) < 12:
        return need_more()
    if header[8:12] != b"WAVE":
        return _riff_error("RIFF header does not declare WAVE")

    position = 12
    resolved_fmt: tuple[int, int, int, int] | None = None
    while True:
        if len(header) < position + 8:
            return need_more()
        chunk_id = header[position : position + 4]
        chunk_size = int.from_bytes(
            header[position + 4 : position + 8], "little"
        )
        body_start = position + 8
        body_end = body_start + chunk_size
        next_position = body_end + (chunk_size & 1)
        if chunk_id == b"fmt ":
            if chunk_size < 16:
                return _riff_error("WAVE fmt chunk is shorter than 16 bytes")
            if len(header) < body_start + 16:
                return need_more()
            body = header[body_start : body_start + 16]
            resolved_fmt = (
                int.from_bytes(body[0:2], "little"),
                int.from_bytes(body[2:4], "little"),
                int.from_bytes(body[4:8], "little"),
                int.from_bytes(body[14:16], "little"),
            )
        elif chunk_id == b"data":
            if resolved_fmt is None:
                return _riff_error("WAVE data chunk precedes fmt chunk")
            format_code, channels, sample_rate_hz, bits = resolved_fmt
            encoding = _WAV_FORMAT_CODES.get(format_code)
            if encoding is None or bits != _WAV_FORMAT_BITS[encoding]:
                return _riff_error(
                    f"WAVE format code {format_code} at {bits}-bit is "
                    "unsupported"
                )
            return RiffFormat(
                encoding=encoding,
                sample_rate_hz=sample_rate_hz,
                channels=channels,
                data_offset=body_start,
                data_bytes=chunk_size,
            )

        if len(header) < next_position:
            return need_more()
        position = next_position


# @spec ING-FE-002
def mulaw_table() -> FloatSamples:
    """Return the public ITU G.711 mu-law expansion table as float32."""
    encoded = np.arange(256, dtype=np.int32) ^ 0xFF
    magnitude = ((encoded & 0x0F) << 3) + 0x84
    magnitude = magnitude << ((encoded & 0x70) >> 4)
    linear = np.where(encoded & 0x80, 0x84 - magnitude, magnitude - 0x84)
    return np.asarray(linear / 32768.0, dtype=np.float32)


# @spec ING-FE-002
def alaw_table() -> FloatSamples:
    """Return the public ITU G.711 A-law expansion table as float32."""
    encoded = np.arange(256, dtype=np.int32) ^ 0x55
    segment = (encoded & 0x70) >> 4
    magnitude = (encoded & 0x0F) << 4
    magnitude = np.where(
        segment == 0,
        magnitude + 8,
        (magnitude + 0x108) << np.maximum(segment - 1, 0),
    )
    linear = np.where(encoded & 0x80, magnitude, -magnitude)
    return np.asarray(linear / 32768.0, dtype=np.float32)


class _StreamingResampler:
    """Stateful 8 kHz to 16 kHz scipy-poly-v1 implementation."""

    _HALF_LEN = 20

    def __init__(self) -> None:
        """Initialize scipy's public polyphase-equivalent FIR state."""
        self._filter = (
            firwin(2 * self._HALF_LEN + 1, 0.5, window=("kaiser", 5.0)) * 2.0
        )
        self._state = np.zeros(len(self._filter) - 1)
        self._initial_skip = self._HALF_LEN

    def push(self, samples: FloatSamples) -> FloatSamples:
        """Resample one decoded piece while retaining FIR history."""
        if not len(samples):
            return np.zeros(0, dtype=np.float32)
        upsampled = np.zeros(2 * len(samples))
        upsampled[0::2] = samples
        result, self._state = lfilter(
            self._filter, 1.0, upsampled, zi=self._state
        )
        return self._emit(result)

    def flush(self) -> FloatSamples:
        """Drain the resampler's finite lookahead tail exactly once."""
        result, self._state = lfilter(
            self._filter,
            1.0,
            np.zeros(self._HALF_LEN),
            zi=self._state,
        )
        return self._emit(result)

    def _emit(self, samples: FloatSamples) -> FloatSamples:
        """Discard initial filter delay once and normalize output dtype."""
        if self._initial_skip:
            skipped = min(self._initial_skip, len(samples))
            self._initial_skip -= skipped
            samples = samples[skipped:]
        return np.asarray(samples, dtype=np.float32)


# @spec ING-FE-002, ING-FE-003, ING-FE-004
class StreamingAudioFrontend:
    """Decode one accepted stream to normalized 16 kHz mono float32 samples."""

    def __init__(self, *, encoding: str, sample_rate_hz: int) -> None:
        """Bind one already-validated matrix cell.

        Direct construction rejects invalid cells as a programmer error;
        adapters use :func:`validate_format` to project client-facing errors.
        """
        if (encoding, sample_rate_hz) not in ACCEPT_MATRIX:
            raise ValueError(
                f"({encoding}, {sample_rate_hz}) is outside the accept-matrix"
            )
        self._bytes_per_sample = 2 if encoding == "LINEAR_PCM" else 1
        self._table: FloatSamples | None = None
        if encoding == "MULAW":
            self._table = mulaw_table()
        elif encoding == "ALAW":
            self._table = alaw_table()
        self._resampler = (
            _StreamingResampler() if sample_rate_hz == 8000 else None
        )
        self._byte_tail = b""

    @property
    def resampler_identifier(self) -> str | None:
        """Return provenance for the 8 kHz path and none for direct PCM."""
        return RESAMPLER_ID if self._resampler is not None else None

    # @spec ING-FE-002, ING-FE-003, ING-FE-004
    def push(self, data: bytes) -> FloatSamples:
        """Decode complete samples and retain any incomplete trailing sample."""
        combined = self._byte_tail + data
        usable = len(combined) - (len(combined) % self._bytes_per_sample)
        self._byte_tail = combined[usable:]
        if self._table is None:
            samples = np.asarray(
                np.frombuffer(combined[:usable], dtype="<i2").astype(np.float32)
                / 32768.0,
                dtype=np.float32,
            )
        else:
            samples = self._table[
                np.frombuffer(combined[:usable], dtype=np.uint8)
            ]
        if self._resampler is not None:
            samples = self._resampler.push(samples)
        return np.ascontiguousarray(samples, dtype=np.float32)

    # @spec ING-FE-003, ING-FE-004
    def flush(self) -> FloatSamples:
        """Discard an impossible partial sample and drain resampler state."""
        self._byte_tail = b""
        if self._resampler is None:
            return np.zeros(0, dtype=np.float32)
        return np.ascontiguousarray(self._resampler.flush(), dtype=np.float32)
