---
parent: high-level-design
prefix: ING
facets: [FE]
---

# Audio frontend

## Context and Design Philosophy

The model plugin and everything behind it see one audio format: 16 kHz mono `float32`. The audio
frontend is the single conversion point between the formats the compatibility surfaces accept and
that format. It is shared by the Riva gRPC stream and `Recognize`, and the Riva Realtime WebSocket;
the file route hands core's service the client's file and uses core's decoder. It owns the accept-matrix, the
decoders, the sample-boundary carry, the resampler, the bound on audio held before `feed`, and
the in-memory WAV that `Recognize` hands to core.

Principles:

- **A small, explicit matrix.** Four cells, mono, and anything else is an explicit rejection.
  A false declaration is a client error and is never converted into silent garbage.
- **Whole samples only.** Message framing never aligns with sample boundaries; the decoder never
  decodes half a sample.
- **Normalize, do not exact-match.** Resampling shifts waveforms, so the resampler's identity is
  part of any result's fingerprint.
- **The model owns segmentation.** The frontend hands the lease arbitrary whole-sample pieces and
  never computes a cadence chunk size.
- **No filesystem.** Audio is held in memory only.

## Accept-matrix

Every inference surface that decodes audio itself (streaming and unary Riva, Realtime) accepts
exactly these mono cells:

| Encoding | Sample rate | Realtime name |
| --- | --- | --- |
| `LINEAR_PCM` (little-endian int16) | 16 kHz | `pcm16`, 16000 |
| `LINEAR_PCM` | 8 kHz | `pcm16`, 8000 |
| `MULAW` | 8 kHz | `g711_ulaw`, 8000 |
| `ALAW` | 8 kHz | `g711_alaw`, 8000 |

When a request declares or resolves any other encoding, sample rate or channel count, the surface
answers `unsupported_format` naming the offending fields. G.711 is intrinsically 8 kHz, so a
declared 16 kHz G.711 stream is a client error and is never resampled. FLAC, Opus and other
compressed streams, other rates and multichannel audio are rejected explicitly: they carry codec
dependencies with no telephony constituency in the streaming path. (The file route accepts the
containers core's decoder accepts, within core's limits.) General negotiation of formats is left
to the dialects' own definitions.

## Decoders

- **G.711** mu-law and A-law decode through the 256-entry ITU expansion tables, as `float32`
  (the 16-bit value divided by 32768). The tables are pure NumPy.
- **LINEAR_PCM** is interpreted as little-endian int16 and divided by 32768.
- Both produce samples in `[-1, 1)`.

## Carry across messages

For Riva streaming audio messages and Realtime append events, the decoder carries a byte-level
residual across client messages and decodes only complete samples: a sample that straddles two
messages is never decoded as two fragments. A residual left when the stream ends is discarded;
finalization is a signal only after complete samples have been drained. The complete-file paths
(`Recognize` and the file route) follow their bounded request parsing and validation (streaming and
files leaves) and have no residual.

## Resampling

8 kHz audio that the frontend decodes itself (the streaming surfaces and `Recognize`; core decodes
the file route's audio) is resampled to 16 kHz by a pinned polyphase resampler identified by the
named value `scipy-poly-v1`. Its parameters are fixed:

- upsampling ratio 2 (up 2, down 1);
- one low-pass FIR of 41 taps (half-length 20), designed with a Kaiser window of beta 5.0 and a
  cutoff at 4 kHz (the 8 kHz input's Nyquist frequency, half the 16 kHz output's), scaled by the
  ratio 2;
- the filter's 20-sample output delay discarded once at the start of each segment, and its
  20-sample lookahead drained once on flush;
- filter state carried across pieces within a segment, and reset to its initial state by a Realtime
  `clear` (streaming leaf).

These are the parameters `scipy.signal.resample_poly` uses for `up=2, down=1` with its default
window, and they are the reference. A segment is the audio between the start of a stream or a
`clear` and the next `clear` or finalization. For any split into pieces of a segment that ends at
finalization, the concatenated output equals `resample_poly(x, up=2, down=1)` of the whole segment
within an absolute tolerance of 1e-6 (the existing frontend test pins this); a segment that ends at
a `clear` yields a prefix of that result, because `clear` discards everything not yet handed to
`feed`, the resampler's pending output included. Within a segment the output does not depend on how
the input was split into pieces. The identifier is published in the deployment provenance for any result produced
from resampled input (operations leaf), and `resampler_identifier` in configuration must equal the
built-in identifier: it is a fact of the build, not a choice. Parity and golden comparisons stay at
16 kHz native; 8 kHz comparisons are normalized comparisons, not exact matches.

## Hand-off to the lease

After `open`, the frontend hands the lease arbitrary-size, whole-sample, 16 kHz mono `float32`
pieces. Streaming audio messages are decoded and handed on as they arrive; the model plugin alone
owns the accepted residual, cadence segmentation and final-tail arithmetic, and no frontend code
computes or reads the cadence chunk size.

The buffer between a transport read and `feed` is bounded by `pre_submit_max_samples`, the number of
normalized samples handed to `feed` and not yet acknowledged, together with read-side flow control:

- Credit for one `feed` call's samples is released exactly once, from that call's `on_accepted(n)`
  callback. `on_accepted` fires once per `feed`, before any unit is sent, and acknowledges that the
  samples reached the session's packer; the session has no accepted-audio queue to overflow.
- When a `feed` call is awaiting the engine and the owner cannot hand over more audio, the handler
  stops reading that transport (gRPC request iterator; WebSocket receive) until `feed` returns;
  samples the handler holds stay within the bound.
- A message larger than the bound is split into feeds of at most the bound with reads paused while
  feeding (streaming leaf), so a legal message never exceeds it; `buffer_overflow` for the
  pre-submit layer is reserved and not reachable for legal messages, and no surface keeps a cadence
  buffer.
- A lease failure with reason `input_capacity` or `output_capacity` is projected by the errors leaf
  and identifies the model plugin's layer.

Decoding is incremental. A received message is decoded in slices whose decoded (and resampled)
output is at most `pre_submit_max_samples` samples, so the decoded form of a large message never
exceeds the bound and each slice goes to its own `feed` call; the decoder's byte residual and
resampler state carry across slices exactly as across messages.

Deferred-header bytes are separate bounded post-open state (`max_riff_header_bytes`) and are never
fed until the cell resolves (streaming leaf).

At finalization the frontend drains the decoder: the resampler's lookahead is flushed once, and a
byte residual that is not a whole sample is discarded. The drained samples are fed before `flush`
(session leaf).

## In-memory WAV for `Recognize`

`Recognize` converts its accepted audio to an in-memory 16 kHz mono 16-bit RIFF/WAV with no
filesystem use: decode and resample as above (the same decoder instance type, so a clip decodes to
the same samples as on a stream), requantize to 16-bit, and write a canonical 44-byte header with
the data length. Requantization is the single function

`pcm16(x) = clip(rint(x * 32768), -32768, 32767)` as int16,

where `rint` rounds half to even. Core's decoder reads the WAV back as `int16 / 32768`, so a sample
that is already a multiple of 1/32768 (every PCM16 and G.711 value, and every 16 kHz PCM input)
round-trips exactly, and only resampled 8 kHz audio is changed by the requantization.

The same function is available to the stream side as a `float32 -> float32` helper
(`pcm16` followed by division by 32768), so the equivalence comparison between a complete file and
a stream (HLD Success metrics 2) can apply the identical requantization to the stream's samples. It
is used only by the comparison; live streams are not requantized.

## Decisions & Alternatives

| Decision | Chosen | Alternatives Considered | Rationale |
| --- | --- | --- | --- |
| Accepted cells | Four mono cells | Any rate; resample everything | Telephony is the streaming use case; the cost is one lookup table and one resampler; a 16 kHz G.711 declaration is a client mistake |
| Compressed streams | Rejected in the streaming path | Add FLAC/Opus decoders | Codec dependencies with no constituency among Riva streaming clients; the file route has core's decoders |
| Whole-sample carry | Byte-level residual | Decode what arrives | A straddling sample decoded as halves corrupts audio and drifts accuracy for reasons unrelated to the model |
| Resampler | Polyphase FIR, pinned and named | A newer or faster resampler | The identity is part of a result's fingerprint; changing it is a versioned change (`scipy-poly-v1` -> new name) |
| Resampler across `clear` | Reset to the initial state; pending output discarded | Continuity across `clear` | `clear` means discard what is buffered; delayed samples surviving it would reach the engine after the client asked to discard them; the whole-input oracle then applies to each uninterrupted segment |
| Segmentation | Left to the model's packer | Frontend cadence buffer | Two segmenters could cut the same audio differently |
| Pre-submit credit | Released from `on_accepted` | Released when `feed` returns | `feed` returns after units complete, which can take engine time; the credit is about the packer having the samples |
| Stream and file decode | One decoder implementation | Separate file decoder | File-vs-stream equivalence needs the same samples |
| WAV requantization | `rint` to int16 with a published function, also offered to the stream side | Truncation; float WAV | `Recognize` goes through core's WAV decoder, which reads 16-bit integers; the identical function on the stream side makes the comparison exact |
| WAV sample format | 16-bit integer | A 32-bit float WAV, which would need no requantization | Half the bytes (against core's upload limit), and every decoder backend reads 16-bit integers the same way; the only difference from float is the requantization of resampled audio, which the shared function makes explicit |

## Open Questions & Future Decisions

### Deferred
1. Additional streaming encodings, if a client population needs them; each is a new cell with its
   own decode path and a new resampler identifier when resampled.
2. A higher-quality 8 kHz resampler; versioned by identifier.

## References

- ITU-T G.711 (mu-law and A-law expansion).
- `scipy.signal.resample_poly`, `scipy.signal.firwin`, `scipy.signal.lfilter`.
- Core's audio loader: `vllm/multimodal/media/audio.py` on `pdtgct/vllm` (16-bit reads as
  `int16 / 32768`).
- Model plugin: `docs/intent/session/` (`SESS-IN-*`, `SESS-PACK-*`), `docs/intent/transcription/`.
- Streaming leaf (formats and header inference), files leaf, session leaf, errors leaf.
