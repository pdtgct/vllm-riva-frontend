# Audio frontend specs

Design: [audio-design.md](audio-design.md). Prefix `ING`, facet FE.

## Accept-matrix and decoding (FE)

- [ ] **ING-FE-016**: Every inference surface that decodes audio itself (streaming and unary Riva, and the Riva Realtime WebSocket) shall accept exactly the mono accept-matrix `(LINEAR_PCM, 16 kHz)`, `(LINEAR_PCM, 8 kHz)`, `(MULAW, 8 kHz)` and `(ALAW, 8 kHz)`, and when a request declares or resolves any other encoding, sample rate or channel combination the surface shall answer `unsupported_format` naming the offending fields.
- [ ] **ING-FE-014**: The streaming path shall reject FLAC, Opus and other compressed streams, other sample rates and multichannel audio explicitly, and shall never resample a declared 16 kHz G.711 stream.
- [ ] **ING-FE-002**: G.711 mu-law and A-law shall decode via the ITU lookup tables and LINEAR_PCM shall be interpreted as little-endian int16.
- [ ] **ING-FE-018**: Each decoder shall produce `float32` samples equal to the 16-bit value divided by 32768.
- [ ] **ING-FE-017**: 8 kHz audio that the frontend decodes itself shall be resampled to 16 kHz by the pinned resampler (ING-FE-021), a named value with default `scipy-poly-v1`, and the resampler identifier shall be published in the deployment provenance the frontend reports for any result produced from resampled input.
- [ ] **ING-FE-022**: The pinned resampler shall carry its state across pieces within a segment, a segment being the audio between the start of a stream or a Realtime `clear` and the next `clear` or finalization, so that its output does not depend on how a segment's input was split into pieces, and a `clear` shall reset it to its initial state (ING-NIMWS-024).
- [ ] **ING-FE-021**: The `scipy-poly-v1` resampler shall upsample by 2 with the fixed polyphase FIR the audio design gives (41 Kaiser-window taps, beta 5.0, cutoff at the 8 kHz input's Nyquist frequency, gain 2), discard the filter's 20-sample delay once at the start of each segment (ING-FE-022) and drain its 20-sample lookahead once at finalization; for any split of a segment that ends at finalization into pieces, its concatenated output shall equal `scipy.signal.resample_poly(x, up=2, down=1)` of the whole segment within an absolute tolerance of 1e-6, and for a segment that ends at a `clear` the output handed to `feed` shall be a prefix of that result within the same tolerance.
- [ ] **ING-FE-020**: For Riva streaming audio messages and Realtime append framing, the front end shall carry a byte-level residual across client messages and decode only complete samples, so that a sample straddling two messages is never decoded as two fragments.

## Hand-off to the lease (FE)

- [ ] **ING-FE-009**: Streaming audio messages shall hand the lease arbitrary whole-sample 16 kHz mono `float32` pieces with finalization as a signal only after complete samples are drained, the model plugin alone shall own accepted residual, cadence segmentation and final-tail arithmetic (SESS-PACK-002 to SESS-PACK-004), and no frontend code shall compute or read the cadence chunk size.
- [ ] **ING-FE-007**: After `open`, the frontend's buffer between transport read and `feed` shall be bounded by the finite positive `pre_submit_max_samples` value on outstanding samples handed to `feed` and not yet acknowledged, and by read-side flow control, shall release exactly one `feed` call's outstanding-sample credit from that call's `on_accepted(n)` callback (SESS-IN-003), and no surface shall maintain a cadence buffer.
- [ ] **ING-FE-011**: The audio frontend shall decode a received message in slices whose decoded and resampled output is at most `pre_submit_max_samples` samples, giving each slice its own `feed` call, with the byte residual and resampler state carried across slices exactly as across messages.
- [ ] **ING-FE-010**: At finalization the audio frontend shall flush the resampler's lookahead once, discard a byte residual that is not a whole sample, and make the drained samples available to be fed before `flush`.
- [ ] **ING-FE-015**: A `SessionFailed` with reason `input_capacity` or `output_capacity` shall be projected as defined by ING-ERR-013.

## In-memory WAV (FE)

- [ ] **ING-FE-008**: `Recognize` shall convert accepted audio to an in-memory 16 kHz mono 16-bit RIFF/WAV with a canonical 44-byte header and no filesystem use, decoding through the same decoder implementation the stream side uses.
- [ ] **ING-FE-012**: The 16-bit requantization shall be `clip(rint(x * 32768), -32768, 32767)` as int16 with half-to-even rounding, and the same function (followed by division by 32768) shall be available to the stream side for the file-versus-stream equivalence comparison and shall not be applied to live streams.
