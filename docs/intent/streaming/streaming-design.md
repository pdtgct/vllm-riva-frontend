---
parent: high-level-design
prefix: ING
facets: [GRPC, NIMWS, EOU]
---

# Streaming surfaces

## Context and Design Philosophy

This leaf is the wire contract of the two live-stream surfaces and of the Riva gRPC service as a
whole: the `RivaSpeechRecognition` servicer (`StreamingRecognize`, `Recognize`,
`GetRivaSpeechRecognitionConfig`) and the Riva Realtime WebSocket (dispatcher, session bootstrap,
events). `Recognize` is described here as a gRPC wire contract (message handling, limits, header
inference); its execution path is in the files leaf. How a stream is driven (owner, terminal cause,
bounds) is the session leaf; the error catalog and its projections are the errors leaf; decode is
the audio leaf.

Principles:

- **The stock client is the contract.** A Riva client generated from the public protocol, and the
  `nvidia-riva/python-clients` tools, work unmodified. Where a client's defaults send a value the
  model cannot honour, the value is accepted and the deviation is documented, rather than rejected.
- **Honest subset.** Every request field is honoured, accepted as model-intrinsic, or rejected
  naming the field. Nothing is silently ignored.
- **Direct translation.** Each event maps straight to a lease call or lease result.
- **Public sources define the dialect.** The Riva protocol is `nvidia-riva/common`; the Realtime
  dialect is the public NIM Realtime reference and `riva/client/realtime.py` of `nvidia-riva/python-clients`.

## Riva gRPC servicer

The servicer implements all three RPCs of `nvidia.riva.asr.v1.RivaSpeechRecognition`, with message
types from the generated stubs of the public `nvidia-riva-client` package (pinned), not vendored
code. It is registered on the plugin's `grpc.aio.server()` (vehicle leaf).

| RPC | Mapping |
| --- | --- |
| `StreamingRecognize` | session factory (below) |
| `Recognize` | complete file on core's transcription service (files leaf); request handling below |
| `GetRivaSpeechRecognitionConfig` | static description (below) |

### `StreamingRecognize`

The first request message is `streaming_config`; any other message first is `protocol_order`
(`FAILED_PRECONDITION`). The handler validates it completely and calls
`open` before advancing the request iterator; every later message carries `audio_content`; a later
`streaming_config` is terminal `protocol_order` with abnormal lease cleanup. The session opens with
`cadence="560ms"`, the convention shared with the Realtime WebSocket, so a clip transcribed over
either decodes on the same lookahead, and with the validated locale; cadence is server policy
because the Riva protocol has no cadence field.

Validation of the first message covers:

- the audio cell: a concrete accepted `(encoding, sample_rate_hertz, audio_channel_count)` (audio
  leaf), or the explicit deferred-header declaration `ENCODING_UNSPECIFIED` (below);
- `language_code` (session leaf's locale rule), `model` (empty or the served identity),
  `max_alternatives`, and every other `RecognitionConfig` field per the disposition table;
- the request-level `runtime_config` rider;
- `interim_results` (both values honoured).

A single legal message can carry more audio than the pre-submit bound. The handler splits the
decoded samples into `feed` calls of at most `pre_submit_max_samples` samples and pauses reading
from the transport while it feeds, so a legal message never overflows the bound. The Realtime
adapter does the same for an `input_audio_buffer.append`. `buffer_overflow` is therefore not
reachable on the streaming surfaces for legal messages from the pre-submit layer; the code is kept
reserved there. (It remains reachable from the model plugin's session capacity limits, errors
leaf.)

Response messages are `StreamingRecognizeResponse` with one result holding one alternative whose
`transcript` is the cumulative hypothesis. A changed cumulative hypothesis returned by `feed` is
sent with `is_final=false` when `interim_results` is true. Client half-close maps to `flush`
(session leaf) and yields exactly one result with `is_final=true`, after `finish` and `release`
succeed and before the server closes the stream. The same final, then gRPC `OK`, is what a
shutdown finalization produces at drain (vehicle and session leaves). The final result also carries `audio_processed`: the seconds of 16 kHz audio handed to `feed`,
the resampler tail drained at finalization included, all of which the model has processed once `flush` returns. The frontend emits no other
field (no word offsets, stability, channel tag or confidence), and no `audio_processed` on an interim
result. When cancellation, a deadline or an abort during shutdown wins before
the final, the servicer emits no fabricated final response; a cleanup failure never replaces the
already-selected cancellation or deadline status, and on an otherwise normal path follows the
session leaf's cleanup rules. Transcript text is the model's output relayed unchanged.

### `RecognitionConfig` field disposition

Protobuf scalars default to zero, so zero means "unset" for `audio_channel_count` and
`max_alternatives` (normalized to 1) and, with `ENCODING_UNSPECIFIED`, for `sample_rate_hertz` and
`audio_channel_count` as header constraints. A non-zero `sample_rate_hertz` or `audio_channel_count`
under `ENCODING_UNSPECIFIED` is a constraint the resolved header must match; under a concrete
encoding the same fields are the concrete audio configuration and are validated against the
accept-matrix. The generated client's defaults (all zero or empty, plus the values the stock tools
set from a WAV header) must pass.

The disposition covers every field of `RecognitionConfig` at the pinned stub version (2.26.0 lists
fifteen). A contract test walks `RecognitionConfig.DESCRIPTOR`, so a stub upgrade that adds a field
fails until the field is dispositioned. Rejected fields answer `INVALID_ARGUMENT`
(`invalid_config_field`) for scalar or default-shape fields and `UNIMPLEMENTED`
(`unsupported_capability`) for absent capability classes, in both cases naming the field.

| Field | Disposition |
| --- | --- |
| `encoding` | honoured: `LINEAR_PCM`, `MULAW`, `ALAW`; `ENCODING_UNSPECIFIED` defers to header inference |
| `sample_rate_hertz` | honoured: 8000, 16000 (the accept-matrix, audio leaf); with a concrete encoding, 0 is unset and `unsupported_format` naming the field; with `ENCODING_UNSPECIFIED`, 0 means unconstrained |
| `language_code` | honoured: any key of the factory's `locales`; empty means `default_locale` |
| `audio_channel_count` | honoured: 1; 0 means unset and is normalized to 1; larger is rejected (`unsupported_format`) |
| `max_alternatives` | honoured: 1 (greedy decode has one hypothesis); 0 means unset and is normalized to 1; larger is rejected |
| `enable_automatic_punctuation` | model-intrinsic: neither value changes output; punctuation follows the checkpoint |
| `verbatim_transcripts` | model-intrinsic: neither value changes output |
| `profanity_filter` | rejected when true |
| `enable_word_time_offsets` | rejected when true (a candidate with per-utterance finals) |
| `enable_separate_recognition_per_channel` | rejected when true (mono only) |
| `model` | honoured: empty or the served identity; any other value is rejected naming `model` |
| `speech_contexts` | rejected when non-empty (word boosting; `unsupported_capability`) |
| `diarization_config` | rejected when `enable_speaker_diarization` is true or any field is set (`unsupported_capability`) |
| `custom_configuration` | rejected when non-empty |
| `endpointing_config` | rejected when any field is non-zero (`unsupported_capability`); see deferred end-of-utterance |

`StreamingRecognitionConfig.interim_results` is honoured for both values. The request-level
`runtime_config` of a `StreamingRecognizeRequest` (a rider, not a `RecognitionConfig` field) is
rejected as `invalid_config_field` naming `runtime_config` when non-empty, and is not treated as a
`RecognitionConfig` field or silently ignored. On the first message the rejection comes before
`open`; on a later message it is terminal: none of that message's audio is fed, and the open lease
takes `abort` then `release`. (The unary `RecognizeRequest` of the
pinned stubs has no such field.)

`enable_automatic_punctuation` and `verbatim_transcripts` are accepted under either value because
the public `python-clients` tools send them by default (`--no-verbatim-transcripts` defaults to
off, so verbatim is requested; `--automatic-punctuation` defaults to off), and rejecting a
canonical default would break the connectivity the honest subset exists to keep. The deviation is
stated in the servicer's user-facing documentation and in the configuration RPC's description.

### Header inference

The canonical `python-clients` file tools declare no encoding: `RecognitionConfig.encoding` is left
unset and the raw file bytes, RIFF header included, are streamed as audio. So when
`encoding` is `ENCODING_UNSPECIFIED`, the format is resolved from a RIFF/WAVE header at the head of
the audio:

- the header's format code (PCM-16, mu-law, A-law), sample rate and channel count resolve the cell;
  the header bytes are stripped and never decoded as audio;
- the resolved cell is validated against the accept-matrix exactly as a declared one is; a declared
  `sample_rate_hertz` or `audio_channel_count` is a constraint the header must match;
- for streaming this is a fully validated deferred-header mode: the owner completes `open` before
  reading audio, buffers at most `max_riff_header_bytes` after `open`, and calls no `feed` until
  the cell resolves;
- unary `Recognize` resolves and validates its already-complete request before it takes a
  file-gate permit;
- no complete parseable header within the bound, or an off-matrix cell, is `unsupported_format`
  naming the fields;
- a declared non-`UNSPECIFIED` encoding is never overridden by a header;
- the `data` chunk's declared length decides how much of the stream is audio: a length of 0 or
  `0xFFFFFFFF` (the sentinels of a header written before the length was known) means "to the end
  of the stream"; any other length is explicit and truncates the audio at that many bytes; chunks
  after `data` are ignored.

### Message limits

`init_state` requires finite positive `unary_max_encoded_audio_bytes`,
`unary_max_decoded_duration_seconds`, `max_riff_header_bytes` and `grpc_config_envelope_max_bytes`.
`grpc_receive_max_bytes` equals the encoded-audio limit plus the envelope allowance. That framework
message cap applies to every request message of both RPCs: a message above it may be rejected by
gRPC itself with its native `RESOURCE_EXHAUSTED` framing error before the servicer runs, the one
exception to the stable catalog-detail rule.

The size and duration limits below are unary only. A `StreamingRecognize` message is limited only
by the framework cap and is decoded in bounded slices (audio leaf), and a stream's total duration is
unlimited unless `max_session_duration` is set. A resource or decode failure after `open` ends the
session with abnormal lease cleanup (session leaf) and its phase-table code.

### Unary `Recognize` request handling

The handler for `Recognize` runs here up to the point where it hands audio to the files leaf's path.
It processes request audio in bounded memory and performs no filesystem write, local spool or
retained upload artifact on success, rejection, cancellation, deadline or failure; client fixture
reads and test-result writes are outside this prohibition.

For `Recognize`, every encoded-size or decoded-duration overflow the handler can reach returns
`RESOURCE_EXHAUSTED` with code `request_too_large` before the handler takes a file-gate permit.
Before allocating decoded or resampled audio, the handler derives the duration from the bounded
payload bytes and the fixed-width cell of the accept-matrix (PCM16 two bytes per sample; G.711 one
byte), so conversion cannot create an unbounded second copy.

### `GetRivaSpeechRecognitionConfig`

The response is static, derived from the session factory and the accept-matrix. It lists the
served-model identity once for each pair of type and locale, because the stock Riva client scripts
(`nvidia-riva/python-clients`, `scripts/asr/transcribe_file.py` and
`transcribe_file_offline.py`) answer `--list-models` by keeping the entries whose
`parameters["type"]` is `online` (streaming) or `offline` (complete files) and grouping them by the
single `parameters["language_code"]` each entry carries:

- for each locale in the factory's `locales`, in code-point order, an entry with
  `parameters["type"] = "online"`, always;
- then, for each locale in the same order, an entry with `parameters["type"] = "offline"`, when the
  shared files-enabled flag is true (the transcription task is present and the tiny-deployment
  exception has not disabled files);
- `auto` is listed when it is a member of the factory's `locales`, and is never synthesized;
- in each entry: `language_code` set to that entry's locale (no plural `language_codes`), sample rates `8000,16000`, encodings
  `LINEAR_PCM,MULAW,ALAW`, and the model-intrinsic notes for punctuation and verbatim transcripts.

## Riva Realtime WebSocket

### Dispatcher

A pure-ASGI middleware, added by `attach_router` inside core's middleware (vehicle leaf), claims a
connection only when the scope type is `websocket`, the path (after `root_path`) is exactly
`/v1/realtime`, and strict percent and UTF-8 decoding of the query yields the multiset of pairs
exactly `{("intent", "transcription")}`. Every other scope passes unchanged to core: a missing,
blank, duplicate, unsupported or additional query, a malformed encoding, other paths, and every
non-WebSocket scope. The dispatcher does not delete, reorder, replace or interpret core's routes or
events. A claimed scope is served by the adapter below; core's route never sees it.

### Receive cap

The dispatcher enforces `ws_receive_max_bytes` on each claimed connection by counting the bytes of
every received message itself; it does not depend on a server setting. It requires
`ws_receive_max_bytes` to be at least `ws_event_envelope_max_bytes` (startup check). Two sizes
apply to a received message:

- above `ws_receive_max_bytes`: the connection is closed with 1009 (an `error` carrying
  `request_too_large` precedes the close when the connection can still send);
- above `ws_event_envelope_max_bytes` but within the receive cap: before any JSON or base64
  decode, one `error` carrying `request_too_large`, then close 1009.

A frame the server's own limit rejects first may produce the server's native close 1009 without
the stable event; the dispatcher's caps hold whichever limit is lower. The bound on an
`input_audio_buffer.append` is the envelope cap: an append above `ws_event_envelope_max_bytes` is
refused as above (`request_too_large`, close 1009) before any decode, `feed` or allocation, so an
append the adapter decodes carries at most the samples that the cap's base64 payload can encode.
Before resample allocation the adapter derives that decoded sample count from the payload length
and splits the append into feeds of at most `pre_submit_max_samples` samples, which remains the
separate bound on samples outstanding at `feed`.

### Bootstrap

`POST /v1/realtime/transcription_sessions` returns the default transcription-session object shaped
per the public NIM Realtime reference (the stock client posts `{}` and echoes the result back as its
first update). It carries the served-model identity, `modalities: ["text"]`,
`input_audio_format: "pcm16"`, `input_audio_params` of 16 kHz mono, `recognition_config` with
`max_alternatives: 1`, disabled `speaker_diarization` and `word_boosting`, an empty
`endpointing_config`, `client_secret: null`, and `input_audio_transcription.language` set to the
factory's `default_locale`. When `default_locale` is `None`, the language is empty and the client
must name one in its update. The route creates no owner and starts no timer.

### Refusal before configuration

A claimed connection refused by the ingress cap or by a draining plugin is handled in this order:
the adapter accepts the WebSocket, sends one `error` carrying the catalog code (`busy` with its
`authority`, or `service_unavailable`), and closes (1013 for both; see the errors leaf). It sends no
`conversation.created`, creates no owner or lease, and reads no client event.

### Events

On an admitted claimed connection the adapter sends `conversation.created` before processing client events.
Client and server events map directly to the lease:

| Event | Mapping |
| --- | --- |
| `transcription_session.update` | first update: validate, `open`, then `transcription_session.updated`; later: only locale is mutable, through the serialized `update_locale`, acknowledged after success; other changed fields answer `config_change_rejected` (session leaf) |
| `input_audio_buffer.append` | audio leaf decode, the bounded pre-submit buffer and read-side flow control, then serialized `feed` |
| `input_audio_buffer.commit` | advisory: acknowledged with `input_audio_buffer.committed`; never forces a sub-cadence engine step |
| `input_audio_buffer.clear` | see below; acknowledged with `input_audio_buffer.cleared` |
| `input_audio_buffer.done` | the only event that triggers final-tail flush: tail, `flush`, `finish`, `release` (session leaf) |
| changed cumulative hypothesis | `conversation.item.input_audio_transcription.delta` carrying the true incremental suffix |
| final after release | exactly one `conversation.item.input_audio_transcription.completed` with `is_last_result: true`, then close 1000; a shutdown finalization sends the same `completed`, then close 1001 (vehicle leaf) |
| transcription failure | `conversation.item.input_audio_transcription.failed` |
| protocol or session failure | `error` |

Delta events carry only the new text, and the adapter emits no interim `completed` events: one
terminal `completed` per session. Malformed JSON, a non-object or malformed envelope, an unknown
event type, or structurally invalid required fields or field types on a known event mutate no
state; the adapter emits one field-specific `error` with catalog code `invalid_event` when a response
can be serialized safely, then closes 1008. A structurally valid event carrying semantically
unsupported configuration instead takes the specific catalog code of the configuration rules below.
After `input_audio_buffer.done` the owner keeps a bounded receive observer until the final's send
begins or the shutdown trigger cancels it (session leaf). A malformed or unknown envelope it sees
keeps `invalid_event` (1008), and a message above the receive cap keeps `request_too_large` (1009);
every other client message (`append`, `clear`, `commit`, `done`, `transcription_session.update`)
fails the session with `session_terminal` and suppresses the final if its send has not begun. Once
the final's send has begun it is irrevocable and later traffic is not read.

Each Realtime owner processes accepted commands through one FIFO. Every accepted
`input_audio_buffer.append` ahead of a `clear` completes its bounded lease handoff before `clear`
takes effect and therefore survives it. At that point `clear` discards all audio not yet handed
to `feed`: raw-byte residual, deferred-header bytes, decoded pieces still local to the frontend and
the resampler's pending output. It resets the resampler (FIR history, delay compensation and
lookahead) to its initial state (audio leaf), acknowledges `input_audio_buffer.cleared`, and never
cancels or rewinds audio handed to `feed`, audio the lease
accepted or model state. Clearing a partial deferred RIFF header restarts the bounded scan from
empty.

### Audio format and configuration

`input_audio_format` and `input_audio_params` are validated jointly against the accept-matrix:
`(pcm16, 16000, mono)`, `(pcm16, 8000, mono)`, `(g711_ulaw, 8000, mono)` and `(g711_alaw, 8000,
mono)`. The G.711 names are the OpenAI Realtime enum values. Every other combination, including
16 kHz G.711 or non-mono, answers `unsupported_format` naming the fields.

`input_audio_format: "none"` is the declaration the stock file-mode client sends with raw container
bytes. It is a fully validated deferred-header mode, handled as `ENCODING_UNSPECIFIED` above:
`open` first, at most `max_riff_header_bytes` buffered after it, RIFF/WAVE header resolved and
stripped, declared `input_audio_params` enforced as constraints, no `feed` until resolution,
`unsupported_format` naming the fields on failure. A declared non-`"none"` format is never
overridden by a header. `"none"` is a documented compatibility declaration of the stock client and
does not join the honoured OpenAI enum.

The first update's sections are dispositioned field for field like the gRPC table:
`recognition_config`, `speaker_diarization`, `word_boosting`, `endpointing_config`,
`custom_configuration`. `input_audio_transcription.model` is empty or the served identity;
`input_audio_transcription.language` follows the locale rule. While the factory's `default_locale`
is set, the default object round-trips without rejection; when it is `None`, the default object's
language is empty and the client must name a locale. A section becomes non-default only through
its enable toggle, non-zero windows or non-emptiness. A non-empty `input_audio_transcription.prompt` and `modalities` other than
`["text"]` are `invalid_config_field` naming `prompt` or `modalities`: they are scalar or
default-shaped fields, not capability sections. `id`, `object`, `client_secret` and
`provenance` are echo keys, not configuration. Unknown keys answer `invalid_config_field` naming the
dialect-native field. Rejections name the dialect's own field names.

Initial configuration is all-or-nothing: if any field is invalid, the adapter emits every field
error, calls no `open`, emits no `transcription_session.updated` and stays unconfigured, so a later
complete corrected update can succeed. The first `transcription_session.updated` means only
"configured"; it may carry the deployment's provenance object and never uses admission vocabulary.

## Deferred: end-of-utterance

When end-of-utterance is enabled, utterance segmentation implements the public protocol's
documented window and threshold semantics (`start_history` and `start_threshold`, `stop_history` and
`stop_threshold`, `stop_history_eou` and `stop_threshold_eou` of `EndpointingConfig`) over the
per-unit blank and non-blank mask that the model plugin reports (a report the model plugin does not yet specify).
On an end-of-utterance the session emits a segment final and requests the decoder-state reset
through the model plugin's segment-completion reporting. Until that lands, non-default
`endpointing_config` is rejected per the disposition tables; when it lands, the six fields become
honoured values and this paragraph is superseded. Behaviour is validated by unit tests derived from
the documented semantics (synthetic silence and speech patterns with known window outcomes) and, as
an advisory lane, by comparison with the behaviour described in the public Speech NIM reference;
never by golden parity, since the NeMo reference model has no endpointing.

## Decisions & Alternatives

| Decision | Chosen | Alternatives Considered | Rationale |
| --- | --- | --- | --- |
| One leaf for the gRPC facet | `Recognize`'s wire handling here, its execution path in the files leaf | Split the facet between streaming and files | A facet belongs to exactly one leaf, and the servicer, field table, limits and header inference are shared by both RPCs; the files leaf owns what is specific to complete files |
| Stubs | `nvidia-riva-client` generated stubs, pinned | Vendored codegen | The canonical client and the servicer then share one proto artifact; one fewer build step |
| Streaming cadence | `560ms` fixed | Client field (none exists); model default | Server policy; one cadence for both streaming surfaces |
| Configuration listing shape | One entry per type and locale, types `online` and `offline`, singular `language_code` | One entry per type listing every locale in `language_codes`; type `streaming` | The stock scripts select `online` or `offline` entries and group them by the singular `language_code`; the online filter excludes `streaming`, and a plural `language_codes` leaves the grouping key empty |
| `verbatim_transcripts`, punctuation | Accepted under either value, documented | Reject the non-default | The canonical tools send a value by default; rejecting a default breaks the stock client |
| `Recognize` request-level rider | `runtime_config` rejected on streaming requests | Ignore | Riders are request-level options; the honest subset covers them |
| Receive cap | The dispatcher counts bytes and closes 1009 | Read the ASGI server's limit and fail if smaller; set the server limit | A plugin cannot set the ASGI server's frame limit and the server does not expose it uniformly; counting makes the cap independent of server settings. The server's own limit still bounds what arrives before the count |
| WebSocket `none` format | Deferred-header mode, outside the honoured enum | Honour as an OpenAI enum value | It is a client-specific declaration; keeping it separate keeps the OpenAI enum exact |
| Incremental deltas | True suffix, one terminal `completed` | Cumulative deltas and interim completions | Deltas carry new text only, so a client concatenating them gets the transcript; one terminal event matches the stock client's `is_last_result` handling |
| `clear` and resampler state | Discard all audio not yet handed to `feed` and reset the resampler to its initial state | Continuity across `clear` (keep FIR history and delayed samples) | `clear` means discard what is buffered; delayed samples surviving it would reach the engine after the client asked to discard them |
| Traffic after `done` | A bounded receive observer until the final's send begins; observed traffic fails with `session_terminal` | Stop reading at `done`; let traffic race the close | Reading only between calls makes the outcome depend on timing; an observer gives one testable result |
| Message limits | Framework cap for both RPCs; size and duration limits for `Recognize` only | Apply the unary limits to streams | A stream's duration is open-ended by design; `max_session_duration` is the option that bounds it |
| `commit` | Advisory | Force a step | Cadence is the model's; a sub-cadence step would break equivalence with files |
| Large legal messages | Split into feeds of at most the pre-submit bound, reads paused while feeding | Reject a message above the bound with `buffer_overflow` | A client may legally send a large message; rejecting it would turn framing choices into errors; `buffer_overflow` stays reserved at this layer |
| RIFF `data` length | 0 and `0xFFFFFFFF` mean to end of stream; an explicit length truncates; later chunks ignored | Trust the declared length; reject unknown length | Writers that stream a WAV cannot know the length; a declared length is a promise about the file |
| Refusal on WebSocket | Accept, `error`, close 1013, no `conversation.created` | Deny the upgrade; send `conversation.created` first | The client must receive the catalog code, and a session that was never admitted should not be announced |
| Malformed `intent` queries | Passed to core | Deny with 1008 | The dispatcher claims one exact shape and leaves everything else to core |
| Bootstrap default locale | The factory's `default_locale` | Always `auto` | The default is the model's to define |
| End-of-utterance | Deferred with the model plugin | Frontend-side segmentation | Segmentation needs the per-unit mask and decoder reset that only the model plugin has |
| `audio_processed` and `confidence` | `audio_processed` on the final result only; `confidence` unset | `audio_processed` on interims as the audio fed so far; a confidence derived from the transcript | The proto defines `audio_processed` as the audio processed so far: the frontend knows what it fed, and the model has processed all of it only after `flush`, so an interim value would overstate it; an interim therefore reads 0.0, which the proto, unlike for `confidence`, does not define as unset. The proto makes 0.0 the sentinel for an unset `confidence`, and the session interface returns text only |

## Open Questions & Future Decisions

### Deferred
1. **Per-utterance finals** (end-of-utterance) and word time offsets, designed together with the
   model plugin's endpoint binding and segment reporting.
2. Whether `GetRivaSpeechRecognitionConfig` should list unsupported features explicitly, if stock
   clients need it.
3. A Realtime `provenance` object in `transcription_session.updated`: its contents are the
   deployment's own identifiers (operations leaf); whether stock clients need it is unconfirmed.
4. An exact interim `audio_processed`, which needs the session interface to report the samples the
   model has processed.

## References

- Riva protocol: `nvidia-riva/common` (`riva/proto/riva_asr.proto`); generated stubs in
  `nvidia-riva-client` 2.26.0 (`riva/client/proto`).
- Stock clients: `nvidia-riva/python-clients` (`riva/client/asr.py`, `riva/client/argparse_utils.py`,
  `riva/client/realtime.py`, `scripts/asr/`), checked at commit `5bb8bd6`.
- Public Speech NIM Realtime and ASR documentation.
- Model plugin: `pdtgct/vllm-nemotron-asr-streaming-plugin` (`docs/intent/session/`).
- Session leaf (owner, bounds, finalization), audio leaf (decode), errors leaf (catalog), files
  leaf (`Recognize` execution), vehicle leaf (listener, dispatcher placement).
