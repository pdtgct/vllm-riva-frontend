---
parent: high-level-design
prefix: ING
facets: [ERR]
---

# Error catalog

## Context and Design Philosophy

The frontend serves three wire dialects, and a client of each expects failures in that dialect's
shape. One catalog of stable code strings, projected mechanically into each dialect, keeps the
meaning of a failure the same everywhere and lets a client branch on a string that does not change
between releases. The catalog also decides how failures that the model plugin's session reports
(its `SessionRefused` and `SessionFailed` reasons) become client-visible codes.

Principles:

- **Every failure is visible and stable.** No failure, rejection or deviation is silent; every
  error a dialect shows carries a catalog code. A refusal that comes before any dialect is engaged
  (authentication, a WebSocket handshake the route does not serve) is visible as HTTP status and
  carries no catalog code (Outside the catalog, below).
- **Distinct meanings stay distinct.** Load, inactivity, a bounded wait and a missed finalization
  are different facts and have different codes.
- **Name the cause.** A rejection of a field, format or capability names the offending field.
- **Cleanup never rewrites the cause.** The first selected cause is what the client sees.
- **Map each seam reason on its own.** A refusal is not automatically a capacity problem.
- **Core's error shape on the frontend's file route.** The namespaced file route's own errors use
  the HTTP error shape of core's transcription route; core's errors pass through.

## Projection rules

**Riva gRPC.** Every error is a gRPC status via `context.abort`, and the stable code string leads
the status details: `<code>: <detail> (fields: <names>)`. The status enum is the projection; the
details string is the one text channel wide enough to carry the code and the field names. Exception:
a message above the framework cap may be rejected by gRPC with its native framing error before the
servicer runs.

**Riva Realtime WebSocket.** Transcription failures use
`conversation.item.input_audio_transcription.failed`; protocol and session failures use `error`.
An `error` event carries the code and the field names, and `authority` for `busy`. Catalog entries
that require a close mark the adapter terminal; every pending event is sent before the close
while the transport accepts each send within `response_send_timeout` (session leaf); a send that
exceeds it ends the owner by the transport-failure path and abandons the connection with no
further send, including the close frame. The close codes are:

| Close | Used for |
| --- | --- |
| 1000 | normal completion after the one `completed` event |
| 1008 | protocol or format violation (`protocol_order`, `invalid_event`, `unsupported_format` after configuration, `invalid_audio`, `session_terminal`) |
| 1009 | oversized message or event (`request_too_large`) |
| 1011 | internal or finalization failure, and timeouts (the phase table below says which event precedes it) |
| 1001 | the server is draining and its shutdown finalization finalized the stream: sent after the final |
| 1012 | the server is draining and the stream was aborted (`service_unavailable`) |
| 1013 | refused before configuration: `busy`, `service_unavailable` |

During drain, a stream finalized at the settle point ends with gRPC `OK` after its final and
WebSocket close 1001; a stream aborted because its finalization failed or missed the deadline ends
with gRPC `UNAVAILABLE` (`service_unavailable`) and WebSocket close 1012 (vehicle leaf). A
client-selected finalization that the shutdown trigger finds running keeps its own outcome: on
success it ends with gRPC `OK` and WebSocket close 1000 after its final (session leaf).

A format or configuration rejection before the first configuration completes does not close the
connection, so the client can retry with a corrected `transcription_session.update`.

Three refusals of a Realtime connection are distinct. An **admission refusal** (ingress cap,
draining) happens before frontend admission, on a connection whose query the route serves: the
adapter accepts it, sends `error` with `busy` or `service_unavailable`, and closes 1013, so a
stock client can read the code and retry; it creates no owner. An **engine-capacity refusal** is
discovered by `open`, after the owner has registered, accepted the WebSocket, sent
`conversation.created` and received its configuration (session leaf): that existing owner ends
without receiving a lease, with `busy`, `authority: provider`, and close 1013. A **query refusal**
concerns a connection whose query is not exactly `intent=transcription`: the route closes it
before accepting it, the ASGI server answers the handshake with HTTP 403, and no event, close code
or catalog code reaches the client (streaming leaf); it creates no owner.

**HTTP.** The file route's own errors use core's HTTP error shape,
`{"error": {"message": "<code>: <detail>", "type": ..., "param": <field or null>, "code": <http
status>}}`, with `type` following core's convention (for example `BadRequestError`,
`RequestTooLargeError`). Core's own errors pass through unchanged. The operational routes use
NIM-shaped bodies (operations leaf).

### Outside the catalog

These refusals and errors carry no catalog code:

| Case | Response |
| --- | --- |
| an HTTP request under `/plugins/riva/` while authentication initialization is unresolved (vehicle leaf, path-scoped authentication; unreachable after a completed startup) | 503 with the body `{"error": "Service Unavailable"}` |
| a WebSocket handshake under `/plugins/riva/` while authentication initialization is unresolved | refused before accept: 503 with that body where the ASGI server offers the WebSocket denial-response extension, otherwise HTTP 403 with no body |
| an HTTP request under `/plugins/riva/` without a valid bearer token while authentication is on (vehicle leaf, path-scoped authentication) | core's 401 `{"error": "Unauthorized"}`, the response core's own middleware sends |
| a WebSocket handshake under `/plugins/riva/` without a valid bearer token while authentication is on | refused before accept (vehicle leaf, WebSocket refusal): core's 401 and body where the ASGI server offers the WebSocket denial-response extension, otherwise HTTP 403 with no body |
| a Realtime WebSocket query other than exactly `intent=transcription` | handshake rejected with HTTP 403 before accept; no body |
| an HTTP request for an unknown path under the namespace (after authentication) | the router's 404 |
| a WebSocket handshake for a path under the namespace with no WebSocket route (after authentication) | Starlette closes before accept, which the ASGI server answers with HTTP 403 |
| a gRPC message above the framework cap | gRPC's native `RESOURCE_EXHAUSTED` framing error |
| core's own errors on the file route | core's shape, unchanged |

They precede or lie outside the dialect whose catalog projection would carry a code: an
authentication refusal is answered as core answers its own prefixes, so a client sees one
authentication contract for the whole server, and a refused handshake opens no session that could
carry an event. Core's own routes (`/v1/realtime`, `POST /v1/audio/transcriptions`) are core's and
outside this leaf.

## Catalog

Not every code is reachable from every surface; `n/a` marks a code the HTTP file route never
produces. `authority` is carried by `busy` only.

| Code | gRPC status | Realtime | HTTP |
| --- | --- | --- | --- |
| `service_unavailable` | `UNAVAILABLE` | `error`, close 1013 | 503 |
| `busy` | `RESOURCE_EXHAUSTED` | `error`, close 1013 | 503 |
| `admission_wait_timeout` | `DEADLINE_EXCEEDED` | `error` | n/a (reserved: never emitted) |
| `idle_timeout` | `ABORTED` | `error`, close 1011 | n/a |
| `finalization_timeout` | `DEADLINE_EXCEEDED` | `error`, close 1011 | n/a |
| `configuration_timeout` | `DEADLINE_EXCEEDED` | `error`, close 1011 | n/a |
| `request_timeout` | `DEADLINE_EXCEEDED` | `error`, close 1011 | 408 (body read), 504 |
| `malformed_request` | `INVALID_ARGUMENT` | `error` | 400 |
| `request_too_large` | `RESOURCE_EXHAUSTED` | `error`, close 1009 | 413 |
| `session_terminal` | `FAILED_PRECONDITION` | `error`, close 1008 | n/a |
| `protocol_order` | `FAILED_PRECONDITION` | `error`, close 1008 | n/a |
| `invalid_event` | `INVALID_ARGUMENT` | `error`, close 1008 | n/a |
| `invalid_config_field` | `INVALID_ARGUMENT` | `error`; session continues if no state changed | 400 |
| `unsupported_capability` | `UNIMPLEMENTED` | `error`; session continues if no state changed | 501 (files disabled only) |
| `unknown_locale` | `INVALID_ARGUMENT` | `error`; the existing locale stays on an update | 400 |
| `config_change_rejected` | `OK` (never raised as an abort) | `error`; session continues unchanged | n/a |
| `unsupported_format` | `INVALID_ARGUMENT` | `error` and close, or retry before configuration | n/a (core decodes file-route audio) |
| `invalid_audio` | `INVALID_ARGUMENT` | `...transcription.failed` | n/a (core decodes file-route audio) |
| `buffer_overflow` | `RESOURCE_EXHAUSTED` | `error`, close 1011 | n/a (the pre-submit layer is reserved; reachable from session capacity limits) |
| `internal` | `INTERNAL` | `...transcription.failed` | 500 |

## Phase-specific mapping (authoritative)

This table is the one place that fixes, for a streaming session, which code, gRPC status, Realtime
event and close follow from the phase and cause. Other leaves refer to it; where another leaf
states one of these values, this table governs on any conflict.

**Deadline precedence.** Every deadline an owner holds (a phase bound, the gRPC RPC's absolute
deadline, the drain deadline) keeps its original absolute value and identity; clipping shortens only
the wait, and a wait that ends because it reached the drain deadline expires as the drain deadline.
The earliest expired deadline wins. A phase's row requires that phase's own unclipped deadline to
have expired; when two unclipped deadlines fall due at the same instant, the phase deadline wins. An
abnormal cause already selected is kept; a provisional normal or shutdown finalization is not a
selected cause and does not block a later deadline.

| Phase and cause | Code | gRPC | Realtime event, close |
| --- | --- | --- | --- |
| refused before admission: ingress cap, draining | `busy`, `service_unavailable` | `RESOURCE_EXHAUSTED`, `UNAVAILABLE` | `error`, 1013 |
| refused by `open`: engine capacity (the owner ends without a lease) | `busy`, `authority: provider` | `RESOURCE_EXHAUSTED` | `error`, 1013 |
| first message wrong; later `streaming_config` | `protocol_order` | `FAILED_PRECONDITION` | `error`, 1008 |
| configuration plus `open` exceeded its bound | `configuration_timeout` | `DEADLINE_EXCEEDED` | `error`, 1011 |
| first-audio or idle bound; engine `expired` | `idle_timeout` | `ABORTED` | `error`, 1011 |
| `max_session_duration` | `request_timeout` | `DEADLINE_EXCEEDED` | `error`, 1011 |
| a `feed` or `update_locale` call exceeded `lease_call_timeout` | `service_unavailable` | `UNAVAILABLE` | `error`, 1011 |
| finalization bound of a client-selected finalization | `finalization_timeout` | `DEADLINE_EXCEEDED` | `error`, 1011 |
| undecodable audio | `invalid_audio` | `INVALID_ARGUMENT` | `...transcription.failed`, 1008 |
| `flush` fails; unexpected failure; `finish` fails; cleanup bound expires on the normal path; `release` fails (other than `engine_dead`) | `internal` | `INTERNAL` | `...transcription.failed`, 1011 |
| `engine_dead` in any call, including `release` | `service_unavailable` | `UNAVAILABLE` | `...transcription.failed`, 1011 |
| a client message the receive observer sees after client-selected finalization was accepted and before the final's send began, other than a malformed or unknown envelope (`invalid_event`) or an oversized message (`request_too_large`) | `session_terminal` | `FAILED_PRECONDITION` (unreachable: no message follows a gRPC half-close) | `error`, 1008 |
| oversized message or event | `request_too_large` | `RESOURCE_EXHAUSTED` | `error`, 1009 |
| shutdown finalization succeeded | none | `OK` after the final | `completed`, 1001 |
| after the owner observed the trigger and selected provisional shutdown mode, a failure or a lease-call, finalization or cleanup deadline expired (including while settling the outstanding call), or the drain deadline arrived first (also for a client-selected finalization the trigger found running), unless an abnormal cause was selected before the trigger | `service_unavailable` | `UNAVAILABLE` | `error`, 1012 |

A client disconnect produces no response. A cleanup failure never changes the row selected by the
primary cause. During shutdown finalization the shutdown abort row replaces the rows that a
lease-call timeout, an `internal` failure or `engine_dead` would otherwise select, because the
session is being shut down.

## Raised exceptions from core's service

Core's transcription service reports expected failures either as a returned `ErrorResponse` or as
an exception raised from the call. For `Recognize` the two are projected separately:

| Raised | Code | gRPC |
| --- | --- | --- |
| `VLLMValidationError` for a size limit (`audio_filesize_mb` and the like) | `request_too_large` | `RESOURCE_EXHAUSTED` |
| other `VLLMValidationError` | `invalid_config_field` naming its `parameter` | `INVALID_ARGUMENT` |
| `ValueError` from the decoder reporting an exceeded duration or decode-size limit | `request_too_large` | `RESOURCE_EXHAUSTED` |
| `ValueError` from the decoder for an invalid or unsupported file | `invalid_audio` | `INVALID_ARGUMENT` |
| `ValueError` from the model's `get_generation_prompt` (unsupported field, language) | `invalid_config_field` | `INVALID_ARGUMENT` |
| engine-dead errors | `service_unavailable` | `UNAVAILABLE` |
| `ValueError` that matches none of the cases above | `invalid_config_field` | `INVALID_ARGUMENT` |
| any other exception | `internal` | `INTERNAL` |

The frontend sees only the exception type and message. It identifies core's size-limit and
decoder cases by core's own published types and message texts, each pinned by a frontend test that
runs against core's code. The model refuses no file for its length (files leaf, Model length), so
no row projects a length refusal. An
unmatched `ValueError` is `invalid_config_field` (core's own `ValueError` handler returns 400), not
`internal`. A returned `ErrorResponse` is projected by its HTTP status
as in the file-route section. On the route, the owner re-raises them from the route handler, so
they go to core's own application exception handlers and keep core's shape.

## Code meanings

- `service_unavailable`: the plugin is starting, draining or not ready; a lease call exceeded its
  per-operation bound; the engine is dead or paused; a drained stream was aborted.
- `busy`: the local ingress cap or the file-gate wait is full (`authority: load_shed`), or the
  engine refused admission for capacity (`authority: provider`). Retryable.
- `admission_wait_timeout`: reserved. The session seam has no admission queue, so no surface emits
  it, and no surface conflates capacity or load shedding, a bounded wait, inactivity and a missed
  finalization.
- `idle_timeout`: no audio accepted within the first-audio or idle bound, or the engine ended an
  idle stream (`expired`).
- `finalization_timeout`: accepted finalization reached its bound.
- `configuration_timeout`: the initial configuration plus `open` exceeded
  `preconfiguration_timeout`.
- `request_timeout`: a streaming session exceeded `max_session_duration`; or a file request exceeded
  its body-read bound; or a gRPC absolute deadline won under the deadline precedence rule above.
- `malformed_request`: invalid multipart content type, boundary, structure, required file, or
  framing on the file route.
- `request_too_large`: the bounded unary body, encoded audio, decoded duration, WebSocket envelope
  or file-route body (or its content type, boundary, part, header or text field) exceeded its limit;
  the message names which.
- `session_terminal`: after client-selected finalization was accepted and before the final's send
  began, the receive observer saw a well-formed client message (audio, `clear`, `commit`, another
  `done` or an update). No surface inherits core's silent second-commit behaviour.
- `protocol_order`, `invalid_event`: wrong message order; malformed or unknown Realtime events.
- `invalid_config_field`, `unsupported_capability`: a rejected scalar or default-shaped field, or
  an absent capability class (diarization, word boosting, endpointing).
- `unknown_locale`: the locale is not in the factory's `locales`.
- `config_change_rejected`: a Realtime update changed a field that is immutable after
  configuration.
- `unsupported_format`: the cell is off-matrix, or header inference failed.
- `invalid_audio`: the payload cannot be decoded (for example truncated RIFF data).
- `buffer_overflow`: more audio than the pre-submit bound, or a session capacity limit; the message
  identifies the layer.
- `internal`: an unexpected engine, factory, plugin, transport or cleanup failure with no more
  specific primary error.

## Seam reason mapping

Each reason the model plugin's session reports is mapped on its own.

| Raised | Reason | Code | Notes |
| --- | --- | --- | --- |
| `SessionRefused` | `locale` | `unknown_locale` | client input |
| `SessionRefused` | `cadence` | `internal` | server configuration; startup should have failed |
| `SessionRefused` | `capacity`, `active_limit` | `busy`, `authority: provider` | |
| `SessionRefused` | `engine_paused` | `service_unavailable` | the engine is not taking new streams |
| `SessionRefused` | `request_id_in_use` | `internal` | the factory mints ids; a collision is a bug |
| `SessionFailed` | `expired` | `idle_timeout` | the engine ended an idle stream |
| `SessionFailed` | `engine_dead` | `service_unavailable` | also an escalation trigger |
| `SessionFailed` | `rejected_input`, `abnormal_end`, `state` | `internal` | |
| `SessionFailed` | `cancelled` | the owner's selected cause; `internal` when the frontend selected no cause | a cancelled lease with no frontend cause (for example a torn-down factory) is not the client's doing |
| `SessionFailed` | `input_capacity`, `output_capacity` | `buffer_overflow` | naming the layer (the session's input or transcript limit) |
| any other exception | no `session_error` | `internal` | |
| any exception with an unrecognized `reason` | | `internal` | |

An exception without `session_error` projects `internal`. A `release` failure whose reason is
`engine_dead` projects `service_unavailable`; any other `release` failure projects `internal`. A
failure with reason `engine_dead` also
escalates to a requested shutdown (vehicle leaf), whatever code it shows the client.

## File-route and `Recognize` errors

The file route's own errors:

| Failure | Code | HTTP |
| --- | --- | --- |
| guard refusals (`use_beam_search`, completion limit, sampling fields, ignored fields, `vllm_xargs` keys other than `locale`, unknown form fields) | `invalid_config_field` | 400 |
| missing, duplicate or malformed fields; bad multipart structure | `malformed_request` | 400 |
| unknown `locale` in a client `vllm_xargs` | `unknown_locale` | 400 |
| body, content type, boundary, part or text field above its bound | `request_too_large` | 413 |
| body read timeout (a stalled upload) | `request_timeout` | 408 |
| core execution timeout | `request_timeout` | 504 before response headers; after headers the event stream is terminated |
| response send timeout | `request_timeout` (recorded) | no further send; headers are the first send, so no status follows |
| files disabled on this server | `unsupported_capability` | 501 |
| cancelled by shutdown | `service_unavailable` | 503 before headers; after headers the event stream is terminated |
| draining; ingress cap full; gate wait expired | `service_unavailable`; `busy`; `busy` | 503 |

Core's errors (its validation errors, the model's `ValueError`, upload-size errors) pass through
the route unchanged. For `Recognize`, an `ErrorResponse` from core is projected by its HTTP
status: 400 or 422 to `INVALID_ARGUMENT` with code `invalid_config_field` naming the error's
`param` (or `audio` when core gives none), 413 to `RESOURCE_EXHAUSTED` with `request_too_large`,
501 to `UNIMPLEMENTED` with `unsupported_capability`, 503 to `UNAVAILABLE` with
`service_unavailable`, other 5xx to `INTERNAL` with `internal`. A raised exception is classified by the table above.

## Cleanup and primary cause

An unexpected failure with no more specific primary catalog error projects `internal`. A cleanup
failure never replaces an already-selected cancellation, deadline, validation, transport or engine
error. On an otherwise normal path the phase table decides the code of a `finish` failure, a `release`
failure or a cleanup bound that expires; none emits a final.

## Decisions & Alternatives

| Decision | Chosen | Alternatives Considered | Rationale |
| --- | --- | --- | --- |
| One catalog, three projections | Codes projected mechanically per dialect | Per-surface error strings | Clients branch on one stable string; the dialect shape is only the carrier |
| Code in gRPC details | `code: detail (fields: ...)` | A binary status detail message | The stock client prints the details string; a text prefix is readable everywhere |
| Seam reasons | One mapping row per reason | All refusals to capacity | `locale` is a client error, `cadence` a configuration error, `capacity` is load; collapsing them hides the cause |
| `engine_dead` projection | `service_unavailable` (and escalate) | `internal` | The client's right move is to retry elsewhere or later, which is what `UNAVAILABLE` says; the process is about to be restarted |
| Phase mapping | One authoritative table in this leaf; siblings refer to it | Each leaf states its own codes | Several leaves describe the same phase; codes stated in each could disagree, while one table referred to from every leaf cannot conflict with itself |
| Stalled upload | HTTP 408 | 504 | The client was slow, not the server; 504 stays for core execution |
| Files disabled | `unsupported_capability`, HTTP 501 | 404; 503 | The capability is absent from this server by configuration, not temporarily unavailable |
| Raised exceptions | Classified separately from returned `ErrorResponse` | One mapping by status | Core raises expected failures as exceptions, which have no HTTP status to map |
| Per-operation bound expiry | `service_unavailable` | `idle_timeout` | The server did not answer in time; the client may retry |
| `engine_paused` | `service_unavailable` | `busy` | The engine is paused, not full; retrying against it will fail until it resumes |
| `output_capacity` | `buffer_overflow` naming the layer | `request_too_large` | The transcript limit is a session resource limit like the input limit; the client did not send a too-large request. `request_too_large` stays for sizes of what the client sent |
| Core's errors through `Recognize` | Mapped by HTTP status, field named by `param` | New catalog codes for core's cases | The frontend cannot tell core's causes apart beyond status and `param` without parsing messages; an exact mapping can be added when core exposes typed errors |
| `admission_wait_timeout` | Reserved, never emitted | Use it for the file-gate wait | The seam has no admission queue; the file gate is local load control and clients treat it as `busy` |
| File-route error shape | Core's | A Speech NIM error body | The frontend's route is a second entry to core's transcription service and keeps core's request and response contract, so its errors take the shape clients of core's route already parse |
| `config_change_rejected` on gRPC | `OK`, never raised | A gRPC abort | This frontend rejects a changed configuration on a Riva stream as terminal `protocol_order`; the code exists for the Realtime dialect's in-band rejection |
| `release` failure | `engine_dead` is `service_unavailable`, others `internal` | `internal` for all | A dead engine is retryable elsewhere; any other release failure is a bug on our side |
| `cancelled` with no frontend cause | `internal` | The owner's cause | No cause was selected, so the cancellation did not come from the client or a timer |
| Close after a failure event | Closed per the phase table (1011, or 1008 for undecodable audio) | Keep the connection open | The session is over; leaving it open invites traffic into a dead lease |
| Realtime closes | 1000, 1001, 1008, 1009, 1011, 1012, 1013 | One close code | Clients of the WebSocket dialect distinguish retryable (1012, 1013) from protocol errors (1008), and a drained-and-complete stream (1001) from an aborted one |
| Unserved Realtime query | Handshake rejected before accept (HTTP 403), outside the catalog | Accept, send `error`, close 1008; reuse the admission refusal (1013) | No session dialect is open to carry an event; 1013 tells a client to retry, which a wrong query never fixes; refusing before accept creates no owner and no gauge count |
| Authentication failures | Core's 401 response, outside the catalog; a WebSocket handshake refused before accept, with core's 401 only where the ASGI denial-response extension exists | A catalog code such as `unauthenticated` in each dialect's shape | The middleware applies core's check on the namespace, so it answers as core does on `/v1`; one authentication contract for the server. The gRPC listener is outside `--api-key` |

## Open Questions & Future Decisions

### Deferred
1. A finer mapping of core's `ErrorResponse` for `Recognize` once core exposes typed errors.
2. Whether `admission_wait_timeout` should be emitted for the file-gate wait; it needs a decision to
   amend the reservation.

## References

- Riva status codes: `grpc.StatusCode`; protocol `nvidia-riva/common`.
- Stock clients: `nvidia-riva/python-clients` (`riva/client/realtime.py` handles `error` and
  transcription events).
- Core's error type: `vllm/entrypoints/serve/engine/protocol.py` (`ErrorResponse`, `ErrorInfo`),
  `vllm/entrypoints/serve/engine/serving.py` on `pdtgct/vllm`.
- Model plugin: `docs/intent/session/` (`SESS-ERR-*`).
- Core's authentication response: `vllm/entrypoints/serve/middleware/authenticate.py` on
  `pdtgct/vllm`.
- Session leaf (terminal cause, cleanup), vehicle leaf (escalation, path-scoped authentication),
  streaming leaf (close behaviour, WebSocket route and its query refusal), files leaf (guard,
  route, model length).
