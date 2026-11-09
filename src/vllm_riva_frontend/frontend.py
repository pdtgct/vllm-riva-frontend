"""The audio frontend: the one conversion point to 16 kHz mono ``float32``.

Every streaming surface and ``Recognize`` decode client audio here: the
accept-matrix, the G.711 and PCM16 decoders, the whole-sample carry across
messages, the pinned ``scipy-poly-v1`` resampler, RIFF/WAVE header
inference and the in-memory WAV that ``Recognize`` hands to core. The module
knows no transport, lease or request lifecycle, and it never touches the
filesystem.

Everything here is synchronous and runs in the caller's own task on the
event loop: decoding is incremental, in slices bounded by the caller's
pre-submit limit, over pure NumPy tables.
"""

from __future__ import annotations

import io
import struct
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.signal import firwin, lfilter

#: Normalized mono audio: every decoder here produces ``float32``.
FloatSamples = npt.NDArray[np.float32]

#: The four mono cells, as ``(encoding, sample_rate_hz, channels)``.
ACCEPT_MATRIX = frozenset(
    {
        ("LINEAR_PCM", 16_000, 1),
        ("LINEAR_PCM", 8_000, 1),
        ("MULAW", 8_000, 1),
        ("ALAW", 8_000, 1),
    }
)
#: The pinned resampler; part of the fingerprint of any resampled result.
RESAMPLER_ID = "scipy-poly-v1"

_ENCODINGS = frozenset(cell[0] for cell in ACCEPT_MATRIX)
_BYTES_PER_SAMPLE = {"LINEAR_PCM": 2, "MULAW": 1, "ALAW": 1}
_OUTPUT_RATE = 16_000
_SCALE = np.float32(32768.0)
#: Samples requantized per step; its ``float64`` temporary is 512 KiB.
_REQUANTIZE_SLICE = 65_536


# ------------------------------------------------------------ accept-matrix


@dataclass(frozen=True)
class FormatError:
    """An off-matrix or unparseable format, naming the offending fields.

    ``fields`` uses the audio leaf's neutral names (``encoding``,
    ``sample_rate_hz``, ``channels``); each surface renames them to its
    dialect's fields. Every format error projects ``unsupported_format``.
    """

    fields: tuple[str, ...]
    detail: str = ""

    @property
    def code(self) -> str:
        """The catalog code every format error projects."""
        return "unsupported_format"


# @spec ING-FE-016, ING-FE-014
def validate_format(
    encoding: str, sample_rate_hz: int, channels: int = 1
) -> FormatError | None:
    """Check one declared or resolved cell against the accept-matrix.

    One field is named per offence: ``encoding`` when it is none of the
    three accepted encodings (its rate is then not judged), otherwise
    ``sample_rate_hz`` when the rate is not a cell of that encoding (G.711
    is intrinsically 8 kHz, so a 16 kHz G.711 declaration names the rate and
    is never resampled), and ``channels`` when the audio is not mono.
    """
    fields: list[str] = []
    if encoding not in _ENCODINGS:
        fields.append("encoding")
    elif (encoding, sample_rate_hz, 1) not in ACCEPT_MATRIX:
        fields.append("sample_rate_hz")
    if channels != 1:
        fields.append("channels")
    if not fields:
        return None
    return FormatError(
        tuple(fields),
        f"({encoding}, {sample_rate_hz} Hz, {channels} channels) is outside "
        "the mono accept-matrix LINEAR_PCM 16000/8000, MULAW 8000, ALAW 8000",
    )


def duration_from_bytes(
    encoding: str, sample_rate_hz: int, byte_count: int
) -> float:
    """Seconds of whole samples in ``byte_count`` bytes of one cell.

    The fixed-width cells let a caller bound decoded duration before it
    allocates any decoded or resampled audio.
    """
    width = _BYTES_PER_SAMPLE.get(encoding)
    if width is None or sample_rate_hz <= 0:
        raise ValueError(
            f"no fixed sample width for ({encoding}, {sample_rate_hz})"
        )
    return (max(0, byte_count) // width) / sample_rate_hz


# --------------------------------------------------------------- RIFF/WAVE


@dataclass(frozen=True)
class RiffFormat:
    """What a RIFF/WAVE header declares.

    ``encoding`` uses the accept-matrix names when the header declares one
    of them and a descriptive name outside the matrix otherwise, so the
    caller's ``validate_format`` refuses it naming ``encoding``.
    ``data_offset`` is the byte offset of the first sample; ``data_length``
    is the declared ``data`` length, or ``None`` for the sentinels 0 and
    ``0xFFFFFFFF`` (audio to the end of the stream).
    """

    encoding: str
    sample_rate_hz: int
    channels: int
    data_offset: int
    data_length: int | None


_WAVE_FORMATS = {1: ("LINEAR_PCM", 16), 6: ("ALAW", 8), 7: ("MULAW", 8)}
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE
_DATA_LENGTH_SENTINELS = (0, 0xFFFFFFFF)


def _riff_error(detail: str) -> FormatError:
    return FormatError(("encoding",), detail)


def _wave_encoding(fmt: bytes) -> str:
    """Name the encoding a ``fmt `` chunk body declares."""
    code = struct.unpack_from("<H", fmt, 0)[0]
    bits = struct.unpack_from("<H", fmt, 14)[0]
    if code == _WAVE_FORMAT_EXTENSIBLE and len(fmt) >= 26:
        # The sub-format GUID begins with the plain format code.
        code = struct.unpack_from("<H", fmt, 24)[0]
    known = _WAVE_FORMATS.get(code)
    if known is not None and known[1] == bits:
        return known[0]
    return f"WAVE_FORMAT_{code:#06x}_{bits}BIT"


# @spec ING-GRPC-029, ING-GRPC-021, ING-NIMWS-009
def sniff_riff(
    buffer: bytes | bytearray | memoryview, *, max_header_bytes: int
) -> RiffFormat | FormatError | None:
    """Resolve the RIFF/WAVE header at the head of ``buffer``.

    Returns a ``RiffFormat`` for every parseable header, an off-matrix cell
    included (the caller validates the cell), ``None`` while the header may
    still complete with more bytes, and ``FormatError`` when the bytes are
    not RIFF/WAVE, the header is malformed, or no complete header fits in
    ``max_header_bytes``. Chunk sizes are read, never allocated.
    """
    if max_header_bytes <= 0:
        raise ValueError("max_header_bytes must be positive")
    data = bytes(memoryview(buffer)[:max_header_bytes])
    bounded = len(memoryview(buffer)) >= max_header_bytes

    def incomplete() -> FormatError | None:
        if bounded:
            return _riff_error(
                f"no complete RIFF/WAVE header within {max_header_bytes} bytes"
            )
        return None

    head = min(len(data), 4)
    if data[:head] != b"RIFF"[:head]:
        return _riff_error("the audio does not begin with a RIFF/WAVE header")
    if len(data) < 12:
        return incomplete()
    if data[8:12] != b"WAVE":
        return _riff_error("the RIFF container is not WAVE")

    position = 12
    fmt: bytes | None = None
    while True:
        if len(data) < position + 8:
            return incomplete()
        chunk_id = data[position : position + 4]
        size = struct.unpack_from("<I", data, position + 4)[0]
        body = position + 8
        if chunk_id == b"data":
            if fmt is None:
                return _riff_error("the WAVE data chunk precedes its fmt chunk")
            channels, rate = struct.unpack_from("<HI", fmt, 2)
            return RiffFormat(
                encoding=_wave_encoding(fmt),
                sample_rate_hz=rate,
                channels=channels,
                data_offset=body,
                data_length=None if size in _DATA_LENGTH_SENTINELS else size,
            )
        if chunk_id == b"fmt ":
            if size < 16:
                return _riff_error(
                    "the WAVE fmt chunk is shorter than 16 bytes"
                )
            if len(data) < body + min(size, 40):
                return incomplete()
            fmt = data[body : body + min(size, 40)]
        position = body + size + (size & 1)


# ----------------------------------------------------------------- decoders


def _mulaw_expansion() -> FloatSamples:
    code = np.arange(256, dtype=np.int32) ^ 0xFF
    magnitude = (((code & 0x0F) << 3) + 0x84) << ((code & 0x70) >> 4)
    linear = np.where(code & 0x80, 0x84 - magnitude, magnitude - 0x84)
    table = (linear / 32768.0).astype(np.float32)
    table.flags.writeable = False
    return table


def _alaw_expansion() -> FloatSamples:
    code = np.arange(256, dtype=np.int32) ^ 0x55
    segment = (code & 0x70) >> 4
    magnitude = (code & 0x0F) << 4
    magnitude = np.where(
        segment == 0,
        magnitude + 8,
        (magnitude + 0x108) << np.maximum(segment - 1, 0),
    )
    linear = np.where(code & 0x80, magnitude, -magnitude)
    table = (linear / 32768.0).astype(np.float32)
    table.flags.writeable = False
    return table


_MULAW = _mulaw_expansion()
_ALAW = _alaw_expansion()


# @spec ING-FE-002, ING-FE-018
def mulaw_table() -> FloatSamples:
    """The 256-entry ITU-T G.711 mu-law expansion, as 16-bit value / 32768."""
    return _MULAW


# @spec ING-FE-002, ING-FE-018
def alaw_table() -> FloatSamples:
    """The 256-entry ITU-T G.711 A-law expansion, as 16-bit value / 32768."""
    return _ALAW


class _Resampler:
    """``scipy-poly-v1``: 8 kHz to 16 kHz, stateful within a segment.

    The FIR is the one ``scipy.signal.resample_poly`` designs for ``up=2,
    down=1``: 41 taps, Kaiser window with beta 5.0, cutoff at the input's
    Nyquist frequency, gain 2. Zero-stuffing and filtering with carried
    state, discarding the 20-sample delay once per segment and draining the
    20-sample lookahead once at finalization, reproduces ``resample_poly``
    of the whole segment for any split into pieces.
    """

    _HALF = 20
    _TAPS = firwin(2 * _HALF + 1, 0.5, window=("kaiser", 5.0)) * 2.0

    def __init__(self) -> None:
        """Start a segment in the filter's initial state."""
        self.reset()

    def reset(self) -> None:
        """Return to the initial state; pending output is discarded."""
        self._state = np.zeros(len(self._TAPS) - 1)
        self._skip = self._HALF
        self._pending = False

    def push(self, samples: FloatSamples) -> FloatSamples:
        """Resample one piece, carrying the filter history."""
        if not samples.size:
            return np.zeros(0, dtype=np.float32)
        stuffed = np.zeros(2 * samples.size)
        stuffed[0::2] = samples
        out, self._state = lfilter(self._TAPS, 1.0, stuffed, zi=self._state)
        self._pending = True
        return self._emit(out)

    def drain(self) -> FloatSamples:
        """Flush the lookahead once and start a fresh segment."""
        if not self._pending:
            self.reset()
            return np.zeros(0, dtype=np.float32)
        out, _ = lfilter(self._TAPS, 1.0, np.zeros(self._HALF), zi=self._state)
        tail = self._emit(out)
        self.reset()
        return tail

    def _emit(self, out: npt.NDArray[np.float64]) -> FloatSamples:
        if self._skip:
            dropped = min(self._skip, out.size)
            self._skip -= dropped
            out = out[dropped:]
        return np.ascontiguousarray(out, dtype=np.float32)


# @spec ING-FE-002, ING-FE-018, ING-FE-017, ING-FE-020, ING-FE-022
# @spec ING-FE-021, ING-FE-011, ING-FE-009, ING-FE-010
class StreamingAudioFrontend:
    """Decode one stream's accepted cell to 16 kHz mono ``float32`` pieces.

    One instance serves one stream (and ``Recognize`` decodes through the
    same type, so a clip decodes to the same samples on either path). It
    carries a byte residual across messages so a sample that straddles two
    messages is decoded once, and it resamples 8 kHz audio with carried
    filter state, so its output does not depend on how the audio was split.
    """

    def __init__(self, *, encoding: str, sample_rate_hz: int) -> None:
        """Bind one validated accept-matrix cell.

        An off-matrix cell is a programming error here: surfaces refuse it
        with ``validate_format`` before they build a frontend.
        """
        if (encoding, sample_rate_hz, 1) not in ACCEPT_MATRIX:
            raise ValueError(
                f"({encoding}, {sample_rate_hz}) is outside the accept-matrix"
            )
        self.encoding = encoding
        self.sample_rate_hz = sample_rate_hz
        self._width = _BYTES_PER_SAMPLE[encoding]
        self._table: FloatSamples | None = (
            _MULAW
            if encoding == "MULAW"
            else _ALAW
            if encoding == "ALAW"
            else None
        )
        self._resampler = (
            _Resampler() if sample_rate_hz != _OUTPUT_RATE else None
        )
        self._residual = b""

    @property
    def residual_bytes(self) -> int:
        """Bytes of an incomplete trailing sample held for the next message."""
        return len(self._residual)

    @property
    def resampler_identifier(self) -> str | None:
        """``RESAMPLER_ID`` when this cell is resampled, else ``None``."""
        return RESAMPLER_ID if self._resampler is not None else None

    def decode(
        self, data: bytes | bytearray | memoryview, *, max_samples: int
    ) -> Iterator[FloatSamples]:
        """Decode the whole samples of one message, in bounded slices.

        Yields non-empty, C-contiguous 1-D ``float32`` pieces of at most
        ``max_samples`` output samples each; the caller hands each slice to
        its own ``feed`` before taking the next, so the decoded form of a
        large message never exceeds the bound. An incomplete trailing sample
        is kept for the next message at once; the samples are decoded as the
        iterator is consumed.
        """
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        view = memoryview(data).cast("B")
        if self._residual:
            need = self._width - len(self._residual)
            if len(view) < need:
                self._residual += bytes(view)
                return iter(())
            view = memoryview(self._residual + bytes(view)).cast("B")
        usable = len(view) - len(view) % self._width
        self._residual = bytes(view[usable:])
        return self._slices(view[:usable], max_samples)

    def drain(self) -> FloatSamples:
        """The finalization tail: the resampler's lookahead, flushed once.

        A residual that is not a whole sample is discarded.
        """
        self._residual = b""
        if self._resampler is None:
            return np.zeros(0, dtype=np.float32)
        return self._resampler.drain()

    def reset(self) -> None:
        """Discard everything not yet handed out (the Realtime ``clear``).

        The byte residual and the resampler's history, delay compensation
        and lookahead return to their initial state, so the next audio
        starts a new segment.
        """
        self._residual = b""
        if self._resampler is not None:
            self._resampler.reset()

    def _slices(
        self, view: memoryview, max_samples: int
    ) -> Iterator[FloatSamples]:
        ratio = 1 if self._resampler is None else 2
        step = max(1, max_samples // ratio) * self._width
        for start in range(0, len(view), step):
            samples = self._expand(view[start : start + step])
            if self._resampler is not None:
                samples = self._resampler.push(samples)
            for begin in range(0, samples.size, max_samples):
                piece = samples[begin : begin + max_samples]
                if piece.size:
                    yield np.ascontiguousarray(piece, dtype=np.float32)

    def _expand(self, chunk: memoryview) -> FloatSamples:
        if self._table is not None:
            return self._table[np.frombuffer(chunk, dtype=np.uint8)]
        pcm = np.frombuffer(chunk, dtype="<i2").astype(np.float32)
        return pcm / _SCALE


# ---------------------------------------------------------- requantization


# @spec ING-FE-012
def pcm16(samples: npt.ArrayLike) -> npt.NDArray[np.int16]:
    """The one 16-bit requantization: ``clip(rint(x * 32768))`` as int16.

    ``rint`` rounds half to even. A sample already on the 1/32768 grid (every
    PCM16 and G.711 value) round-trips exactly.
    """
    values = np.asarray(samples)
    out = np.empty(values.shape, dtype=np.int16)
    _pcm16_into(values.reshape(-1), out.reshape(-1))
    return out


def _pcm16_into(values: np.ndarray, out: np.ndarray) -> None:
    """Write ``pcm16(values)`` into ``out``, both one-dimensional.

    The ``float64`` arithmetic runs on one slice at a time, so a long clip's
    temporaries stay within the slice allowance of the files design's
    ``Recognize`` memory bound instead of growing with the clip.
    """
    for start in range(0, values.size, _REQUANTIZE_SLICE):
        stop = start + _REQUANTIZE_SLICE
        scaled = np.multiply(values[start:stop], 32768.0, dtype=np.float64)
        np.rint(scaled, out=scaled)
        np.clip(scaled, -32768, 32767, out=scaled)
        out[start:stop] = scaled


# @spec ING-FE-012
def requantize(samples: npt.ArrayLike) -> FloatSamples:
    """``pcm16`` back to ``float32``, for the file-versus-stream comparison.

    Live streams are never requantized; only the equivalence comparison
    applies this to a stream's samples.
    """
    return pcm16(samples).astype(np.float32) / _SCALE


# @spec ING-FE-008
def encode_wav16(samples: npt.ArrayLike) -> bytes:
    """An in-memory 16 kHz mono 16-bit RIFF/WAV with the canonical header.

    The 44-byte header carries the data length; the samples go through
    ``pcm16``, which core's decoder reads back as ``int16 / 32768``. The
    samples are requantized slice by slice straight into the one buffer the
    returned ``bytes`` is made from (``BytesIO.getvalue`` hands its buffer
    over without copying it), so the WAV is held once.
    """
    values = np.asarray(samples).reshape(-1)
    size = 2 * values.size
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        _OUTPUT_RATE,
        _OUTPUT_RATE * 2,
        2,
        16,
        b"data",
        size,
    )
    buffer = io.BytesIO()
    buffer.write(header)
    if size:
        buffer.seek(len(header) + size - 1)
        buffer.write(b"\0")  # sizes the buffer once, to the whole WAV
        with buffer.getbuffer() as view:
            payload = np.frombuffer(view, dtype="<i2", offset=len(header))
            _pcm16_into(values, payload)
            del payload  # the view cannot be released while it is exported
    return buffer.getvalue()
