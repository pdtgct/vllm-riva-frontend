# Compatibility surfaces

This plugin mounts three compatibility surfaces alongside vLLM-Omni's
native `/v1/realtime` API. All three share the same engine, the same
session factory, and the same admission/load-shed accounting; none of them
starts a second inference engine.

## Identity rule

Every surface validates the served-model identity the same way: it must be
the exact value passed to `--served-model-name` (or the canonical model
value, if that flag was omitted). A client that instead sends the
checkpoint's canonical path or Hugging Face id is rejected — the frontend
does not treat the canonical identity as an additional valid selector, and
does not advertise secondary aliases as compatibility selectors. Where a
surface accepts an optional `model` selector (Riva gRPC's
`RecognitionConfig.model`, Speech NIM HTTP's `model` field, and the
Realtime WebSocket's `input_audio_transcription.model`), it is optional —
omit it and select by locale instead — but if present it must match exactly.

## Riva gRPC

- **Endpoint**: `nvidia.riva.asr.v1.RivaSpeechRecognition` on the
  configured `grpc_bind` (default `0.0.0.0:50051`), plus standard gRPC
  health checking and server reflection.
- **Dialect**: the public Riva ASR proto, as consumed by
  [`nvidia-riva/python-clients`](https://github.com/nvidia-riva/python-clients)
  and any other client generated from those protos.
- **Methods**: `StreamingRecognize`, `Recognize`, and
  `GetRivaSpeechRecognitionConfig`. `GetRivaSpeechRecognitionConfig`
  reports the served model, streaming/offline support, the two supported
  sample rates (`8000`, `16000`), the three supported encodings
  (`LINEAR_PCM`, `MULAW`, `ALAW`), and the model's supported locales.
- **Supported subset**: mono `LINEAR_PCM` at 8 or 16 kHz, mono `MULAW`/`ALAW`
  at 8 kHz; `language_code` against the served model's locale set;
  `enable_automatic_punctuation` and `verbatim_transcripts` are accepted but
  model-intrinsic (see the README's Compatibility scope). RIFF/WAV audio
  with an unspecified encoding is auto-detected from its header.
  `max_alternatives > 1`, `audio_channel_count > 1`, `speech_contexts`,
  `profanity_filter`, `enable_word_time_offsets`,
  `enable_separate_recognition_per_channel`, `diarization_config`,
  `custom_configuration`, `endpointing_config`, and `runtime_config` are all
  rejected if set to a truthy/non-empty value.
- **Error behavior**: every rejection maps to a gRPC status via
  `context.abort(...)` (see the error catalog below) with a message of the
  form `"<code>: <fields>"` naming which request field(s) triggered it.
  `StreamingRecognize` also enforces `preconfiguration_timeout` (config
  before first audio), `session_idle_timeout` (no accepted audio),
  `session_finalization_timeout` (terminal flush), and — if configured —
  `max_session_duration`, each surfaced as the matching timeout code.

## Riva Realtime WebSocket

- **Endpoint**: `/v1/realtime?intent=transcription` on the host's own HTTP
  port (the same port passed to `vllm-omni serve --port`), claimed
  exactly by that path and query string; every other WebSocket scope,
  including plain `/v1/realtime`, passes through unchanged to vLLM-Omni's
  native realtime route. Session bootstrap (returning an echoable session
  object with defaults) is available at
  `POST /v1/realtime/transcription_sessions` on the same port.
- **Dialect**: the Speech NIM realtime JSON event protocol —
  `transcription_session.update` / `.updated`, `input_audio_buffer.append`
  / `.commit` / `.clear` / `.cleared` / `.done`, and
  `conversation.item.input_audio_transcription.delta` / `.completed`.
- **Supported subset**: the first client event must be
  `transcription_session.update`; `input_audio_format` is `"pcm16"`, a
  supported G.711 encoding, or `"none"` to defer format detection to a
  RIFF/WAV header sniffed from the first appended audio. Fields under
  `speaker_diarization`, `word_boosting`, and `endpointing_config` are
  accepted only if left disabled — enabling any of them is rejected as an
  unsupported capability. `recognition_config` supports
  `max_alternatives` (must be `1`), `enable_automatic_punctuation`,
  `enable_verbatim_transcripts` (both model-intrinsic), and
  `enable_word_time_offsets` and `enable_profanity_filter` (both rejected
  if enabled). Once a session is configured, its
  format, params, and recognition/capability fields are immutable — a
  later `transcription_session.update` may only change `language`, and
  only before a locale-lookup rejection would otherwise apply.
- **Behavior note**: this adapter emits true incremental deltas (never
  cumulative resends) and exactly one terminal
  `conversation.item.input_audio_transcription.completed` event per
  session. This intentionally does not reproduce cumulative-delta or
  repeated-interim-completion behavior some other Riva-family deployments
  exhibit.
- **Error behavior**: protocol/config/format errors are sent as a JSON
  event (`type: "error"`, or the catalog's dialect-specific event type for
  a code — see the error catalog below). Most rejections then close the
  WebSocket with a standard close code: `1008` for a protocol or format
  violation, `1009` for an oversized event, `1011` for an internal or
  finalization failure, `1013` when the session is rejected before
  admission for being over the concurrency cap. A format rejection sent
  before the session has finished its first configuration does not close
  the connection, so the client can retry with a corrected
  `transcription_session.update`.

## Speech NIM HTTP transcription

- **Endpoint**: `POST /v1/audio/transcriptions` on the host's own HTTP
  port, as `multipart/form-data`.
- **Dialect**: the Speech NIM HTTP transcription request/response shape —
  a single `file` part plus optional text fields.
- **Supported subset**: exactly one `file` part (RIFF/WAV, auto-detected
  encoding/sample-rate/channels from its header, same encoding/rate
  support as the other surfaces); optional `model` (must match the served
  model if present) or `language` (must be a supported locale; `auto` is
  rejected as a value for this field — provide `language` only if you mean
  an explicit locale) — at least one of the two is required;
  `response_format` is `json` (default, returns `{"text": "..."}"`) or
  `text` (returns the transcript as `text/plain`); `temperature` is
  accepted but model-intrinsic, since this frontend decodes deterministically.
- **Error behavior**: multipart framing/size problems return `400`
  (malformed) or `413` (too large) with an `{"error": {...}}` body naming
  the reason; semantic rejections (unsupported format, unknown locale,
  unsupported/invalid field, audio too long) return `400` with a
  `{"detail": "<code>: <reason>"}` body; a request exceeding the
  configured `http_request_timeout` returns `504`; an unready host returns
  `503`; an unhandled failure returns `500`.

## Error catalog

Every rejection across all three surfaces is drawn from one shared catalog
of stable codes, each projected into that surface's own dialect (a gRPC
status code, a WebSocket/NIM event type, and an HTTP status code). Not
every code is necessarily reachable from every surface — for example, only
the WebSocket and HTTP surfaces project `internal` and `invalid_audio` into
a NIM-specific `conversation.item.input_audio_transcription.failed` event
type; gRPC projects the same code as a plain `error`-shaped abort.

| Code | gRPC status | HTTP status | Meaning |
| --- | --- | --- | --- |
| `busy` | `RESOURCE_EXHAUSTED` | 503 | The local concurrency cap (`load_shed_max_sessions`) is reached; retry later. |
| `admission_wait_timeout` | `DEADLINE_EXCEEDED` | 504 | Reserved in the catalog for a bounded wait on host admission. |
| `idle_timeout` | `ABORTED` | 504 | No audio was accepted within `session_idle_timeout`; the session was aborted. |
| `finalization_timeout` | `DEADLINE_EXCEEDED` | 504 | The session's terminal flush/finish did not complete within `session_finalization_timeout`. |
| `service_unavailable` | `UNAVAILABLE` | 503 | The host's own admission is closed (not ready, or draining). |
| `configuration_timeout` | `DEADLINE_EXCEEDED` | 504 | The client did not complete the configuration/open handshake within `preconfiguration_timeout`. |
| `request_timeout` | `DEADLINE_EXCEEDED` | 504 | The request's own deadline (gRPC deadline, `http_request_timeout`, or `max_session_duration`) was exceeded. |
| `malformed_request` | `INVALID_ARGUMENT` | 400 | The request could not be parsed (bad multipart framing, bad JSON, etc.). |
| `request_too_large` | `RESOURCE_EXHAUSTED` | 413 | A configured size bound was exceeded (audio, multipart envelope, or WebSocket event). |
| `session_terminal` | `FAILED_PRECONDITION` | 409 | The client tried to use a session that had already reached a terminal state. |
| `internal` | `INTERNAL` | 500 | An unexpected adapter or provider failure. |
| `protocol_order` | `FAILED_PRECONDITION` | 400 | An event or frame arrived out of the required order (e.g. audio before configuration). |
| `invalid_config_field` | `INVALID_ARGUMENT` | 400 | A request field is unsupported, unrecognized, or invalid for this deployment. |
| `unsupported_capability` | `UNIMPLEMENTED` | 400 | The request asked for a capability this deployment does not implement (diarization, word boosting, word offsets, endpointing). |
| `unknown_locale` | `INVALID_ARGUMENT` | 400 | The requested language/locale is not in the served model's locale set. |
| `config_change_rejected` | `OK`* | 400 | The client tried to change an immutable session field after the session was already configured. |
| `unsupported_format` | `INVALID_ARGUMENT` | 400 | The audio encoding, sample rate, or channel count is not supported. |
| `invalid_audio` | `INVALID_ARGUMENT` | 400 | The audio payload itself is invalid (e.g. truncated RIFF data). |
| `buffer_overflow` | `RESOURCE_EXHAUSTED` | 400 | More audio was submitted than `pre_submit_max_samples` allows before being accepted. |
| `invalid_event` | `INVALID_ARGUMENT` | 400 | A WebSocket event was malformed, of an unknown type, or had an invalid field type. |

\* `config_change_rejected` is a WebSocket/HTTP-native rejection (an
immutable-session-field change is rejected in-band as a non-fatal event on
the WebSocket surface); its catalog `gRPC status` entry of `OK` reflects
that it is not raised as a gRPC abort in the current gRPC surface.
