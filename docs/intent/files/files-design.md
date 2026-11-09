---
parent: high-level-design
prefix: ING
facets: [FILE, HTTP]
---

# Complete files

## Context and Design Philosophy

Complete-file transcription has two entry points on this server: the Riva `Recognize` RPC and
`POST /v1/audio/transcriptions`. Both run on the same service, core's transcription serving
(`OpenAIServingTranscription`), which sends a file to the model as one ordinary request. The model
plugin makes that request walk the whole file through the streaming transition on its own state
slot (`FILE-*` of the model plugin). This leaf is what the frontend adds in front of that service:
the request guard, the in-flight file gate, locale transport, the bounded in-memory body handling
for the HTTP route, and the `Recognize` adapter.

Principles:

- **Core's service over re-implementation.** The frontend routes to core's service, keeps core's
  response, error shape and server-load accounting, and adapts only the differences at the
  boundary. Clients of the HTTP route get core's OpenAI-shaped response.
- **Files must not displace streams.** Files are admitted by a frontend gate before they enter
  core's queue, because a queued file counts against the engine's active-request limit and could
  make streams refuse admission.
- **Unsupported is refused, not ignored.** A request feature the file path cannot honour is
  answered explicitly before a permit is taken.
- **Audio never touches the filesystem.** The frontend holds a request's audio in memory only,
  within configured bounds.
- **Bounded and owned.** A file request is an owner task; its reads and waits are finite; a client
  disconnect releases everything it held.

The service is core's internal serving class, which RFC 2 documents as not yet a stable contract.
The frontend pins the parts it uses (the constructor-built state attribute, `create_transcription`,
the request model, `read_upload_with_limit`, the request-ID derivation and the engine client's `abort`) with tests
that fail when core changes them, including: a client-supplied `X-Request-Id` does not reach core;
a file is one engine input (the model declares `max_audio_clip_s=None` and no explicit language
detection, so core issues no chunk or language-detection requests with other IDs).

## Capability and route replacement

Complete files need the served model's `transcription` task. Core builds
`state.openai_serving_transcription` and attaches `POST /v1/audio/transcriptions` only for such a
model, so both the route and the state attribute are the capability signal:

- `attach_router` replaces core's `POST /v1/audio/transcriptions` with the frontend's file route
  when core's is present (RFC 2 route override), and replaces nothing when it is absent.
  `/v1/audio/translations` stays core's; the model refuses translation. The frontend does not mount,
  replace or shadow it.
- `init_state` finds the service at `state.openai_serving_transcription` and derives one **files
  enabled** flag: true when the service is present and the gate sizing allows files (below). That
  one flag decides `Recognize`'s capability and the `type=offline` listing (streaming leaf).
  Without the service, `Recognize` answers `UNIMPLEMENTED`, the file route does not exist (core
  attached none), and no `offline` entry is listed.
- When the service is present but files are disabled by the tiny-deployment rule (below), the
  frontend's route stays registered, because `attach_router` already replaced core's; it refuses
  every request with `unsupported_capability`, projected as HTTP 501, as its first step (before the
  ingress cap and before reading a body, creating no owner), and
  `Recognize` answers `UNIMPLEMENTED`.
- Without the frontend loaded, core's route is unguarded; that configuration is unsupported for
  this model.

## One gate

One counting gate of `max_concurrent_files` covers `Recognize` and the file route together.

- Waiters wait in the frontend, as owner tasks, not in core's queue. A wait is bounded by
  `file_gate_wait_timeout`; expiry answers `busy` with `authority: load_shed`.
- A permit is held from the moment the request has passed every earlier check until the response
  body has been fully sent (including `stream=true`), and releases then. For `Recognize` it is held
  until the response is returned. gRPC writes that unary response after the handler returns, so
  the frontend does not bound its delivery; the server's stop at the drain deadline ends it.
- The gate caps files in flight; it does not reserve capacity for streams. A finished file's slot
  stays occupied until the engine retires it, so for a short time slightly more than
  `max_concurrent_files` slots can be held by files. Streams keep the engine's admission, and
  `capacity` or `active_limit` refusals remain possible under combined load.
- `init_state` fails unless `max_concurrent_files` is below both the state-slot count and the
  engine's active-request limit (vehicle leaf, Capacity), so even a full gate leaves slots for
  streams. That rule needs at least two of each, so smaller servers have files disabled instead (below).
- Gate waiters, body readers and executing files are file owners and count against the
  sub-cap `max_file_owners` inside the ingress cap (vehicle leaf), so a file backlog never refuses a
  stream. A file arriving at a full sub-cap answers `busy` (`authority: load_shed`).
- When `min(slots, active limit)` is below 2 and `max_concurrent_files` is not configured, complete
  files are disabled at startup with a warning: the files-enabled flag is false (`Recognize`
  answers `UNIMPLEMENTED`, the route refuses with 501, no `offline` entry is listed). An explicitly configured size that fails the
  check fails startup.
- Closing admission at drain refuses new waiters; waiters are owners and are cancelled with the
  rest.

## The request guard

Before taking a permit, the file route and `Recognize` refuse any request that asks for something
the model's single-slot file path cannot do:

- `use_beam_search` true. Core's beam path resubmits fresh request ids with `max_tokens=1`, which
  cannot keep one acoustic slot.
- any completion limit: `max_completion_tokens` present. The model guarantees that a file ends with
  its end token (not a `length` ending) by checking a worst-case budget against `max_model_len`
  before submission, and that guarantee needs no smaller limit.
- any non-default value of a sampling field that core forwards into the engine's sampling
  parameters: `temperature`, `top_p`, `top_k`, `min_p`, `frequency_penalty`, `presence_penalty`,
  `repetition_penalty`, `seed` and `watermarking`. A value equal to core's field default (for
  `temperature`, `frequency_penalty` and `presence_penalty` 0; for `top_p` 1; `top_k` 0; `min_p` 0;
  `repetition_penalty` 1; `seed` and `watermarking` absent) is accepted; the model decodes
  greedily, so any other value would be silently meaningless. The set is the list in core's
  `TranscriptionRequest.to_sampling_params`, and a test fails when core adds a forwarded field;
- a client `vllm_xargs` with any key other than `locale` (route; below).

The route answers 400 (core's error shape, `invalid_config_field` naming the field) and `Recognize`
never sets these fields. Refusing a token limit in the generation configuration, or a foreign end
token, at startup is the model plugin's check; the frontend guard covers per-request limits only.

**Explicit versus inherited values.** Core's request model has fields that default to `None` and
then inherit the server's generation settings. The guard distinguishes three states: an omitted
field (always accepted), an explicit client value equal to core's default (accepted), and an
explicit non-default client value (refused). The effective inherited value, taken from the served
model's generation settings, is the operator's configuration and is not inspected per request;
`init_state` logs a warning when those settings carry any of the guarded sampling keys, because the
greedy decode ignores them.

**Accepted but ignored fields.** On the non-beam path core accepts some fields it never forwards
to the engine: `n` and `length_penalty` (beam search only), `include_stop_str_in_output`, and the
stream-option fields outside `stream=true`. They are dispositioned: an explicit `n` other than 1,
`length_penalty` other than 1.0, and `include_stop_str_in_output` true are refused with 400
`invalid_config_field` before a permit is taken; the same explicit-versus-default rule applies. Fields
the model itself refuses (`prompt`, `hotwords`, `to_language`) are not ignored: they pass to core
and get its 400.

**Field allowlist.** Core's request model accepts extra fields. The route validates the text
parts against an explicit allowlist of the field names and aliases of core's `TranscriptionRequest`
(including the alias `timestamp_granularities[]`); a part whose name is not in the list is refused
with 400 `invalid_config_field` naming it, and a test fails when core adds or renames a field.

## HTTP route

### Ingress

The route runs in this order, each step before the next:

0. **Files-enabled check**: while files are disabled and the service is present, the route
   answers 501 `unsupported_capability` at once, before the ingress cap and before any read, and
   creates no owner.
1. **Ingress cap** (vehicle leaf): atomically admit the owner. A draining plugin answers 503
   `service_unavailable`; a full cap answers 503 `busy` (`authority: load_shed`). Both create no
   owner or permit.
2. **Content type**: `multipart/form-data` with exactly one syntactically valid ASCII boundary
   parameter. JSON, URL-encoded forms, missing or duplicate boundaries, malformed or non-ASCII
   boundary syntax fail before body parsing as `malformed_request` (400). A syntactically valid
   `Content-Type` or boundary above its configured byte cap is `request_too_large` (413).
3. **Body read**: the body is read in chunks under `http_multipart_envelope_max_bytes` and the
   total read time `http_body_read_timeout`; each chunk goes to the parser as it arrives and is not
   retained, so the encoded body is never held whole. A `Content-Length` above the cap is rejected before
   any read; a missing or chunked length is supported under the same running count. A body above
   the cap answers 413 `request_too_large` before core is called; a read that exceeds the time
   (a stalled upload) answers `request_timeout` with HTTP 408. A disconnect during the read
   discards only bounded memory.
4. **Parse**: the raw ASGI body chunks feed `python-multipart.MultipartParser` callbacks directly.
   The route calls no `request.form()`, uses no file-object `FormParser` and no helper permitted to
   spool, so nothing reaches the filesystem. The one file part goes to a bounded in-memory buffer,
   text parts to separately bounded buffers.
5. **Validate and guard**: field set, locale, core's upload limit, then the guard.
6. **Permit** (the gate), then **call** core's service.

### Multipart limits

While parsing, the handler enforces total bytes, part count, part-header bytes, text-field bytes and
file bytes before extending a buffer. Every ASGI chunk fragmentation, including a split inside a
boundary, parses equivalently. Parser truncation or disconnect is never success and discards only
bounded in-memory state. Malformed framing, a truncated body, a duplicate field, a second `file`
part and a missing `file` part are `malformed_request` (400). Filename, boundary and part content
type are untrusted metadata: never a path or a persisted value.

Limits are configuration of the frontend (vehicle leaf): `http_multipart_envelope_max_bytes`,
`http_content_type_max_bytes`, `http_multipart_boundary_max_bytes`, `http_multipart_max_parts`,
`http_multipart_max_header_bytes`, `http_text_field_max_bytes`, `http_body_read_timeout`.
`init_state` requires `http_multipart_envelope_max_bytes` to be finite, positive and at least
core's upload limit (`VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` MiB) plus the framing allowance of 262,144
bytes, so that an ordinary single-file request reaches the semantic audio limit before the envelope
limit.

### Request construction and the upload limit

The text parts are the form fields of core's `TranscriptionRequest`. The frontend validates them
with core's model so core's semantics apply: a repeated name is allowed only for a field core
declares as a list, and any other duplicate is `malformed_request`; unknown or invalid fields and
values answer core's validation error. The `file` part becomes an in-memory upload object (a
`BytesIO`-backed upload; nothing is spooled), which core's request type requires.

Core's upload limit applies to the file part as core's route would: the frontend runs the file part
through core's own `read_upload_with_limit`, so an oversized file produces core's validation error
with core's message and status.

### Locale on the route

The route does not apply exact-locale transport. It forwards the client's `language` field to core
unchanged, and the model maps it to a locale (the model plugin's language rules). Absent `language`
is the model's default. A client `vllm_xargs` is accepted only when it contains the single key
`locale`: the value is validated against the factory's `locales` (`unknown_locale`, 400) and
forwarded as given, which the model takes ahead of its language mapping; any other key is refused
with 400. The `model` field is validated by core against every served name, as core's own route
does.

### Locale transport for `Recognize`

Exact-locale transport applies to `Recognize`, whose clients name locales. `Recognize`:

1. takes `language_code`, or `default_locale` when empty; when that is also `None`, the client must
   name one (`invalid_config_field` naming `language_code`);
2. validates it against the factory's `locales`; an unknown value is `unknown_locale`;
3. sends it exactly as `vllm_xargs={"locale": <locale>}`, which the model takes ahead of its own
   language mapping;
4. sets `language` to the locale's language only when that language is in the model's
   `supported_languages` (the model class attribute of the service), else `None`; the same applies
   to `auto`, which has no language. A locale's language is the lowercase letters before the first
   `-` or uppercase letter when there are two or three of them (the model plugin's definition,
   which a test pins).

### Calling core

After the guard and the permit, the route calls `create_transcription(audio, request, raw_request)`
and returns core's response as core's route does: a JSON response for a complete result, an error
response with its status for an `ErrorResponse`, or an event stream for `stream=true`.

**Request ID and abort ownership.** Core derives the engine request ID from the request: in
`_create_speech_to_text` it is `f"{task_type}-{_base_request_id(raw_request)}"` with task type
`transcribe`, and `_base_request_id` (`serve/engine/serving.py`) returns the `X-Request-Id` header
of the raw request when present and a random id otherwise. A file is one input, so the engine
request ID equals that string. The route fixes the ID by generating a UUID and giving core a new
Starlette `Request` built over a copy of the original scope whose header list has every client
`x-request-id` entry removed and the frontend's lowercase `(b"x-request-id", <uuid>)` entry added;
it never mutates the original request, whose headers Starlette caches on first access. So the
engine ID is `transcribe-<uuid>`, known to the frontend. `Recognize` has no HTTP request: it builds
a minimal Starlette `Request` from a synthetic scope `{"type": "http", "headers": [(b"x-request-id",
<uuid>)]}`. Core reads that request's headers for the request ID and for tracing headers, and writes
per-request metadata to its `state` (which Starlette creates on demand); it reads no body and polls
no disconnect, so the minimal scope is sufficient.

The frontend does not rely on core's cancellation handling or on closing generators to stop
inference. Core aborts the engine request when its awaited call is cancelled, but the owner never
cancels that call: core decodes audio on an executor thread, so a cancelled await would release the
frontend's accounting while the thread still holds the audio; a request can still be submitted to
the engine after a cancelled await; and for `stream=true` the call has already returned an event
generator, whose remaining work no cancellation of the call reaches. So on client disconnect,
deadline, shutdown or any failure the route's owner calls the engine client's
`abort(<engine request ID>)` itself, within the bound below. This holds equally while a pull of a
streaming response's generator is outstanding. Cancelling the child that made the call, or closing
the outer generator, is never treated as ending the engine request.

**Structured ownership.** The route does not rely on core's `with_cancellation` either: it spawns
a handler task and a disconnect-listener task and cancels the pending one without awaiting it. The
route's owner task owns the core call (a child task) and the disconnect watcher (a second child).
The owner never cancels the core call (only `stop` does, below).

A file request passes four milestones: the permit; the return of core's service call, once core's
preprocessing (decode and prompt construction) is done, which for `stream=true` returns an event
generator; the exhaustion of that generator (`stream=true` only); and the delivery of the response.
The owner alone advances the event generator, one pull at a time. Pulls never overlap, so the
generator is never closed while a pull is running.

**One terminal cause.** The owner records exactly one terminal cause at its operation boundaries:
the permit, the call's return, each generator pull and each response send. A client disconnect or
RPC cancellation observed before any cause is selected wins and produces no response. Otherwise the
first of shutdown, a deadline (compared by the errors leaf's deadline rule) or a failure stays
primary and decides the response. A disconnect observed after a cause was selected suppresses the
response and leaves the recorded cause unchanged. A later abort, close or settlement failure never
changes the response.

**Abort and settlement.** When a cause is selected after the permit and before the call has
returned or, for `stream=true`, before the generator is exhausted, the owner aborts the engine
request ID and repeats the abort every 250 ms until the call has returned and any outstanding pull
has finished (a decode in progress finishes, the request reaches the engine, and the repeated abort
ends it). Core submits the request to the engine lazily, on the generator's first pull, so an abort
sent before submission finds nothing and a pull can submit the request between two retries.
Therefore, once the call has returned and any outstanding pull has finished, the owner sends one
final awaited abort for the ID, before it closes the generator and before any accounting is
released. It then cancels the disconnect watcher and, for a returned event generator, closes it with
`aclose`. Aborting an ID the engine does not know is harmless. The cleanup deadline starts when the
cause is selected and covers the abort retries, the final abort, the generator's close and the
children's settlement: `session_cleanup_timeout`, or during drain the smaller of that and the time
left to the drain deadline. The bound limits the wait, not the settlement, which can outlast it.
The same contract applies to `Recognize`, whose engine request ID is fixed by its synthetic request.

**Cleanup continuation.** A file request's cleanup is a cleanup continuation registered in the owner
registry with the request's record. It owns the abort retries, the final abort, the generator's
close, the children's settlement and the exactly-once release of the permit, the file-owner slot
and the server-load count: accounting is never released while work it covers still runs. Normally
the owner runs the continuation to its end before it returns. When the cleanup bound expires first,
the handler sends its response, if any, and transfers ownership of the remaining cleanup to the
continuation; the record stays registered, with the permit, slot and count, until the continuation
finishes, and `stop` cancels the continuation and the children it holds. A request cancelled at
the drain's settle point is the exception: its handler answers `service_unavailable` at once and
hands the whole cleanup to the continuation, so the client learns of the shutdown while the
listener is still up. The stuck permit is
visible rather than hidden. A disconnect before the permit is taken discards only multipart memory,
with no abort. A cause selected after the service completed normally (the call returned for a
complete response, the generator exhausted for `stream=true`), including a disconnect during the
response send, suppresses the response only, with no abort.

**Load accounting.** The route does its own server-load accounting instead of calling core's
`load_aware_call`: when `enable_server_load_tracking` is set, the owner increments
`server_load_metrics` when it is admitted, and the cleanup continuation decrements it exactly once,
after settlement and after the response send finished or failed, whether the owner ran the
continuation itself or handed it over. Core's decorator catches only `Exception`, so cancellation
could leak its counter, and its decrement rides on a response background task that does not run
when a send fails; a decrement owned by the continuation is exactly once on normal completion,
cancellation, failed sends and a handler that answered before settlement.

**Deadline and streams.** `file_execution_timeout` is an absolute deadline from the permit to the
return of core's service call, and for `stream=true` to the exhaustion of core's event generator,
not around `create_transcription` alone. Delivery after that point is governed only by
`response_send_timeout` for each send, so a complete response whose inference finished within the
deadline is not turned into a timeout by a slow send. For `stream=true` the deadline includes
delivery pacing: each pull of the generator waits until the previous event has been sent, so a slow
event-stream reader consumes `file_execution_timeout`. Before response headers are sent, any failure is an
ordinary HTTP error in core's shape. After headers are sent, a failure ends the event stream with
core's streaming error event shape when it can be serialized, or closes the connection; an HTTP
status is never sent after headers. On shutdown the route answers `service_unavailable` (503
before headers; stream termination after, errors leaf). The permit is released after the engine
request is aborted, the children have settled and the send has finished or failed.

The route bounds each response send: a reader that stalls longer than `response_send_timeout`
(default 60 s) is cancelled. While core's service call is still executing, including an event
stream whose generator is not exhausted, the engine request is aborted; after the service completed
normally, delivery is suppressed and nothing is aborted. In both cases the permit is released after
settlement.

Core's own errors pass through unchanged: a returned `ErrorResponse` is returned as core's route
returns it, and a raised exception is re-raised from the route handler (the owner carries it out of
the core-call child), so that core's application exception handlers shape it as they shape core's
own route.

## `Recognize`

`Recognize` is wire-unary. After request validation it:

1. converts the request's audio to an in-memory 16 kHz mono 16-bit WAV (audio leaf; no filesystem
   use);
2. takes a gate permit;
3. calls core's transcription service with those bytes, the exact locale (the transport above, with
   `language_code` as the locale), `response_format="json"`, no streaming, no beam search and no
   completion limit, and the minimal synthetic request that fixes its engine request ID (see Request
   ID and abort ownership), under the same abort, settlement and accounting rules as the route;
4. returns one `RecognizeResponse` with one result and one alternative carrying the transcript,
   only after the service completes normally.

It uses no session factory, no full-context algorithm of its own and no second segmenter.

**Validation before the permit.** Before taking a permit, `Recognize` validates the complete format
(or resolves and validates a RIFF header, streaming leaf), the locale, the encoded-request bound,
the RIFF-header bound and the decoded-duration bound. At each validation, permit and terminal
boundary, a cancellation or an already-expired RPC deadline wins if already observed; otherwise the
specific validation error wins. `init_state` requires `unary_max_decoded_duration_seconds` to be at
most core's decode-duration limit (`VLLM_MAX_AUDIO_DECODE_DURATION_S`), so a request the frontend
accepts is not rejected by core's decoder.

**After handing the request to core.** Core's execution is bounded by `file_execution_timeout`, for
`Recognize` and the route alike; its default is `max(300 s, 2 * unary_max_decoded_duration_seconds)`
(1200 s at the defaults), generous so legitimate long files are not cut. The RPC's absolute deadline
applies to `Recognize` from servicer entry, through validation, the gate wait and execution; where
it and `file_execution_timeout` both apply, the earliest wins under the errors leaf's precedence
rule. Expiry of the bound is
`request_timeout`; cancellation or deadline expiry aborts the engine request and settles under the
rules above, the permit being released after settlement, and emits no response. Engine death while a file runs is core's to
handle through its own engine watchdog; the frontend does not escalate for it (vehicle leaf).

**Server-load accounting.** `Recognize` has no HTTP request, so it is outside the server-load
counter, which reports HTTP load; only the route is counted.

**Errors.** Core's service reports expected failures in two ways, and the errors leaf projects
each separately: a returned `ErrorResponse` object, and an exception raised from the call (the
model's `ValueError` from `get_generation_prompt`, decoder `ValueError`s such as an invalid file or
an exceeded duration limit, and `VLLMValidationError`). The errors leaf's classification of raised
exceptions is authoritative. A file that exceeds the model's token budget (the model's `ValueError`, which core returns as 400)
projects `request_too_large` for `Recognize` (`RESOURCE_EXHAUSTED`); on the route the response stays
core's own 400. The frontend identifies the refusal by the fixed prefix `file exceeds max_model_len: ` with which
the model plugin begins that message (its FILE-BUD-003), pinned by tests on both sides. There is no startup relation between the duration
cap and `max_model_len`: the budget depends on the model's emission bound, which the frontend cannot
compute, so a file the frontend accepts can still be refused by the model.

**Memory.** Request audio is held in bounded memory; no filesystem write, local spool or retained
upload artifact is created on success, rejection, cancellation, disconnect, shutdown or failure.
The same holds for the file route's whole request path. Client fixture reads and test-result writes
are outside this prohibition.

## Memory per owner

A file owner holds these simultaneous copies, all bounded by configuration. These are the counted
allocations; core's decode memory is not counted.

- **Route:** the parsed file part (up to the file bound), the in-memory upload object built from it
  (the part buffer is handed over, not copied), the `bytes` object core's upload helper joins from
  it, and core's own decoded array (core's limits, not counted). The encoded body is not retained
  beyond the part. Frontend-held: at most `3 x http_multipart_envelope_max_bytes` (about 97 MiB at
  the defaults).
- **`Recognize`:** the encoded request (up to `unary_max_encoded_audio_bytes`), the decoded
  `float32` samples (at most 600 s x 16 kHz x 4 bytes, 38 MB), the resampler's per-slice state and
  temporary arrays (decode runs in slices, below 1 MB each; a 4 MiB allowance), and the WAV (19 MB
  at 600 s). At most `unary_max_encoded_audio_bytes + 6 x 16000 x unary_max_decoded_duration_seconds
  + 44` bytes plus the 4 MiB allowance (about 91 MiB at the defaults).

The per-owner bound is the larger of the two formulas, so it grows with the configured upload and
duration limits. At the default configuration both paths stay under 128 MiB, which is the
acceptance limit: a test sends one maximal request on each path with core's service replaced by a
stub that reads the bytes and returns a transcript, and measures the increase of `tracemalloc`'s
traced peak over the baseline taken before the request (numpy buffers are traced). The default
total is `max_file_owners` times the bound: 3 to 6 owners (the derived gate of 1 to 4 permits plus
2 waiters), at most 768 MiB. Core's own decode memory is bounded by core's limits and is not part
of this budget.

## Equivalence

A complete file's transcript equals the transcript of a factory session fed the same audio at the
cadence core's file path uses, the deployment guide's `num_lookahead_tokens` (ING-EP-058); the factory
accepts any served cadence, so the comparison opens its session at that cadence. It is not the
output of the frontend's 560 ms streaming surface, which is a different lookahead arm unless the
file path's cadence is also 560 ms (it is 1120 ms in the reference deployment). Under matched conditions: the same decoded samples, with the `Recognize` conversion's 16-bit requantization
applied to the stream side too (audio leaf); the same cadence, locale, precision and endpointing
setting (the model plugin's file design states the model's side). The frontend's contribution is
that the decoded samples it hands to core are exactly the samples the stream side would hand to
`feed`, and that the locale core sees is the locale a stream would open with. The end-to-end
comparison is an integration test under the model plugin's conditions: the file's request ends
with EOS, the comparison session is flushed successfully with a fixed locale, and endpointing is
disabled on both. Core returns success for a file without checking for an abnormal finish reason
(Deferred 1), so neither this test nor the gate establishes that a failed file is reported as a
failure.

## Decisions & Alternatives

| Decision | Chosen | Alternatives Considered | Rationale |
| --- | --- | --- | --- |
| Complete files | Core's transcription service through one gate | A stream at a long cadence; a second algorithm | The model's file path exists for this; a second code path would drift from streams |
| HTTP route | RFC 2 route override with in-memory parsing, gate and guard, calling core's service | Middleware that buffers and replays the body to core's route | Core's form parser spools large uploads to disk and the body would be parsed twice; middleware also cannot refuse before core reads the form |
| File-route response | Core's OpenAI shape | A Speech NIM HTTP shape | The tenet: core's service satisfies the contract; a NIM HTTP response is a non-goal |
| Parser | Bounded in-memory `MultipartParser` with the frontend's limits as configuration | Core's form parsing; fixed constants | Core's parser can spool; the limits are operator-visible bounds on frontend memory |
| Request time | `http_body_read_timeout` for the read, `file_execution_timeout` (derived, generous) for core, `response_send_timeout` for each send | Only a read timeout; one total timeout | Every wait of an owner is bounded, so a stalled engine call or reader cannot hold a permit forever; separate bounds keep each generous for its own phase |
| Gate wait expiry | `busy` with `authority: load_shed` | `admission_wait_timeout` | The error catalog reserves that code for a bounded wait on an admission queue; the gate is local load control and clients retry it as `busy` |
| Gate size | Below both the slot count and the active-request limit; files disabled (route stays registered, refuses with 501) when capacity is below 2 and no cap is configured | Equal to the limit | A full gate must leave slots for streams |
| Ownership of the core call | The route's owner task owns the call and the disconnect watcher; it never cancels the call, aborts the engine ID repeatedly until the call returns, and holds accounting until then | Core's `with_cancellation`; cancelling the call child | The decorator does not await the cancelled child, and cancelling the await leaves core's executor-thread decode holding audio outside the frontend's accounting |
| Stream deadline | Absolute through generator exhaustion | A timeout around `create_transcription` | For `stream=true` the call returns before the work is done |
| End of the execution deadline | The core call's return, or generator exhaustion for `stream=true`; each send bounded separately | The end of the response | A complete response whose inference finished in time should not become a timeout because a send was slow; the send bound covers its delivery. For `stream=true`, pulls wait on sends, so the deadline includes delivery pacing and a slow reader consumes it |
| File terminal cause | One cause at operation boundaries; a disconnect observed before any cause wins; otherwise the first of shutdown, deadline or failure; a later disconnect suppresses the response and keeps the recorded cause | Priority by kind; the last cause wins | Racing causes must yield one testable response; a client that left receives nothing either way |
| Event generator | Advanced and closed by the owner alone, one pull at a time; aborts repeat until any outstanding pull has finished | Close it from the cancelling task | An asynchronous generator cannot be closed while a pull is running, and the pull ends only when the abort reaches the engine |
| Stopping inference | The route fixes the engine request ID through `X-Request-Id` and calls `abort` itself | Close the generator; cancel the call child | Core aborts on cancellation of its awaited call, which the owner never cancels (its decode runs on an executor thread outside the frontend's accounting); a request can still be submitted after a cancelled await; and a returned event generator is out of that call's reach |
| Accounting on unsettled children | A registered cleanup continuation keeps the permit, file-owner slot and load count until they settle and releases each exactly once | Hand off to a bounded cleanup capacity; release in the handler's `finally` | Releasing accounting while work still runs would let new files overcommit slots, and a handler that has answered can no longer run a `finally` for work that settles later |
| Final abort | One awaited abort after the call returned and any pull finished | Stop at the call's return | Core submits lazily, so earlier aborts can all precede submission |
| Server-load counter | The owner increments when admitted; the owner or its cleanup continuation decrements exactly once after settlement and the send | Wrap `load_aware_call` | The core decorator misses cancellation and failed sends |
| Ignored fields | Explicit non-default `n`, `length_penalty`, `include_stop_str_in_output` refused | Accept silently | Unsupported features are refused, never ignored |
| Request fields | Explicit allowlist; explicit non-default sampling refused; inherited values not inspected | Pass unknown fields to core; refuse inherited values | Extra fields would reach the model unvalidated; inherited values are operator configuration |
| File owners | Counted in `max_file_owners` inside the ingress cap | A shared cap | A backlog of waiting files must not refuse streams |
| Equivalence reference | A factory session at the cadence core's file path uses (the deployment guide's `num_lookahead_tokens`, ING-EP-058) | The 560 ms streaming surface | The file runs at that cadence and the factory serves any served cadence, so the comparison is exact |
| Over-budget file | `request_too_large` for `Recognize`, no startup relation to `max_model_len` | `INVALID_ARGUMENT`; a startup check | The file is too large for the configured model length; the frontend cannot compute the model's budget |
| `Recognize` accounting | Outside the server-load counter | Count it too | The counter reports HTTP load; the synthetic request exists only to fix the engine request ID |
| Gate placement | In the frontend, before `generate` | Core's queue | A queued file counts against the active-request limit and refuses streams |
| Guard | `use_beam_search`, completion limits, non-default sampling fields and foreign `vllm_xargs` keys refused with 400 | Ignore; pass through | Beam search cannot keep one slot; a completion limit invalidates the model's budget guarantee; client extras would reach the model |
| Locale on `Recognize` | `language_code` is the locale; exact in `vllm_xargs`; `language` set only for a supported language | Pass `language` and let the model map it | A Riva client that names a locale must get that locale; a guessed region changes transcripts silently |
| Locale on the route | `language` forwarded unchanged; client `vllm_xargs` allowed with only `locale` | Exact-locale transport as for `Recognize` | OpenAI-style clients send languages, which the model maps; an explicit locale remains reachable through `locale` |
| Upload limit | Core's `read_upload_with_limit` on the in-memory upload | A frontend limit | The same limit, message and status as core's route |
| Upload object | In-memory (`BytesIO`) | `UploadFile` backed by a spooled file | Spooling writes large uploads to disk |
| Permit release | When the response body is sent | When the service returns | A streaming response is still using the slot after the service returns |
| Decode-duration check | Startup check against core's limit | Let core reject | A request the frontend accepts and core rejects has already cost a permit |

## Open Questions & Future Decisions

### Deferred
1. Partial transcripts after an engine error: core's serving returns text for an `error` or `abort`
   ending today, for every model. A core change will make those failures; until it lands, a file
   whose request ends that way can return a partial transcript as success. The frontend does not
   inspect finish reasons.
2. Exact-locale transport on the route for clients that cannot name a locale in `vllm_xargs`: they
   rely on the model's language mapping, including `language_locales`.
3. A request-chosen cadence for files (the model plugin's deferred item).
4. Equivalence depends on core's audio decoder returning the integer-exact 16-bit values; a decode
   backend that rescales or dithers would break it. A CPU test fixes the expectation.

## References

- Core serving: `vllm/entrypoints/speech_to_text/transcription/api_router.py`,
  `.../transcription/protocol.py`, `.../base/serving.py`, `.../base/utils.py`,
  `vllm/entrypoints/serve/utils/api_utils.py`, `vllm/envs.py` on `pdtgct/vllm`
  (`endpoint-plugin-lifecycle` and `persistent-state-primitive-v2` merged).
- RFC 2 route override and compatibility note: `docs/design/endpoint_plugins.md`.
- Model plugin: `docs/intent/transcription/` (`FILE-*`), `docs/intent/registration/`.
- `python-multipart` `MultipartParser`.
- Vehicle leaf (capacity, ingress cap, configuration), session leaf (locale rule), streaming leaf
  (header inference, `Recognize` limits), audio leaf (WAV encode), errors leaf.
