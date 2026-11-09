# Implementation layout

The module layout the tests are written against. The designs and specs under `docs/intent/` state
the behaviour; this file names where it lives so tests and code agree. Changing a name here means
changing the tests that use it. Where a leaf design names a module or symbol, that name is used
here; where today's module keeps its role, its name is kept, so the port rewrites modules in place.

Core is vLLM with RFC 1 and RFC 2 merged (`pdtgct/vllm`, the `rfc12-integration-v2` merge). Core
paths below are relative to that tree.

## Package `vllm_riva_frontend`

| Module | Public names |
| --- | --- |
| `__init__` | `plugin()` (entry point `riva_frontend = vllm_riva_frontend:plugin` in group `vllm.endpoint_plugins`): a zero-argument factory returning `RivaFrontendPlugin`; `__version__` |
| `version` | `__version__` (the distribution version `/v1/version` reports) |
| `clock` | `Clock` (the structural `Protocol` of the clock seam: `monotonic()`, `async sleep(seconds)`, `async wait_for(awaitable, deadline)`, which awaits until the absolute `deadline` and otherwise cancels the awaitable and raises `TimeoutError`, as `asyncio.wait_for` does); `RealClock` (`time.monotonic` and asyncio), the clock `RivaFrontendPlugin(clock=None)` uses |
| `lifecycle` | `RivaFrontendPlugin(*, clock=None, entry_points=None)` (`clock=None` is `clock.RealClock()`) with class attributes `name = "riva_frontend"`, `required_tasks = ("realtime",)`, `owns_non_http_listener = True`, and the lifecycle methods `attach_router(app)`, `async init_state(engine_client, state, args)`, `async start(state)`, `ready(state) -> bool`, `async drain(state, timeout)`, `async stop(state)`; attributes tests read after `init_state`: `config` (a `FrontendConfig`), `served_model` (the identity string), `deadlines` (a `StreamDeadlines`), `capacity` (a `Capacity`), `files_enabled`, `factory`, `registry` (an `OwnerRegistry`), `ingress` (an `IngressCap`), `file_gate` (a `FileGate` or `None`), `catalog` (an `ErrorCatalog`), `servicer` (a `RivaServicer`); after `start`: `server` (the `grpc.aio.Server`), `health` (the gRPC `HealthServicer`), `admission_open`, `fault_collector`, `listener_monitor` (tasks); `torn_down`, `fault` (the recorded fault or `None`), `drain_deadline` (`D`, set while draining). `OwnerRegistry()` with `register(kind, task) -> OwnerRecord`, `unregister(record)`, `records` (a mapping), `attach_continuation(record, task)`, `cancel_all() -> list[Task]`, and what owners need from the plugin without holding it (Registry contracts, below); `OwnerRecord(kind, task, continuation=None, children=set(), owner=None)` (`children` is a mutable `set[Task]`; `owner` is the record's shutdown observer); `ShutdownTrigger(deadline)`, the object `drain` raises for stream owners at the settle point; `engine_reserve(timeout)`, `settle_point(now, D)` (the two formulas of the vehicle leaf); `FaultCollector(factory, period=1.0, on_fault)`, `ListenerMonitor(server, on_exit)`; `StartupError` (the exception `init_state` and `start` raise, with `failures: list[str]`) |
| `config` | `FrontendConfig` (one attribute per option of the vehicle leaf's table, including the derived `max_concurrent_files`, `max_file_owners` and `file_execution_timeout`); `parse_config(additional_config) -> FrontendConfig` (reads key `riva_frontend`; raises `ConfigError` listing every failure; derives `file_execution_timeout`, which depends on configuration only, and leaves `max_concurrent_files` and `max_file_owners` `None` when unset, since they depend on the engine's capacity); `ConfigError(failures)`, a subclass of `lifecycle.StartupError`; `StreamDeadlines(never_fed_timeout, parked_idle_timeout, draining_timeout)`, `read_stream_deadlines(additional_config) -> StreamDeadlines`, `DEFAULT_STREAM_DEADLINES` (15, 60, 40); `check_bounds(config, deadlines) -> list[str]`; `Capacity(slots, active_limit)`, `read_capacity(vllm_config) -> Capacity`; `derive_file_caps(config, capacity, service_present) -> tuple[int, int, bool]` (`max_concurrent_files`, `max_file_owners`, files enabled); `DeploymentProvenance` and `build_deployment_provenance(config)` (the four typed fields of the metadata route); `PLUGIN_KEY = "riva_frontend"`, `STREAM_ADMISSION_KEY = "stream_admission"` |
| `errors` | `ErrorCatalog` (the catalog of the errors leaf: `codes`, `grpc_status(code)`, `http_status(code)`, `realtime_close(code)`); `catalog()`; `FrontendError(code, detail="", fields=(), authority=None)` (the exception every surface projects; `grpc_details()` renders `<code>: <detail> (fields: <names>)`, `http_body(status)` renders core's error shape); `seam_error(exc) -> tuple[str | None, str | None]` (`getattr` of `session_error` and `reason`); `project_seam_error(exc) -> FrontendError` (the seam reason mapping); `classify_core_exception(exc) -> FrontendError` and `project_error_response(response) -> FrontendError` (the two file-path tables); `MODEL_LEN_PREFIX = "file exceeds max_model_len: "`; `ESCALATES = frozenset({"engine_dead"})` |
| `frontend` | `ACCEPT_MATRIX`, `RESAMPLER_ID = "scipy-poly-v1"`, `validate_format(encoding, sample_rate_hz, channels) -> FormatError | None`, `FormatError(fields, detail)`, `RiffFormat(encoding, sample_rate_hz, channels, data_offset, data_length)`, `sniff_riff(buffer, *, max_header_bytes) -> RiffFormat | FormatError | None` (`None` means more bytes are needed), `mulaw_table()`, `alaw_table()`, `StreamingAudioFrontend(*, encoding, sample_rate_hz)` with `decode(data, *, max_samples) -> Iterator[np.ndarray]` (whole samples, resampled, sliced to at most `max_samples`), `drain() -> np.ndarray` (the resampler lookahead; the byte residual is discarded), `reset()` (the Realtime `clear`), `residual_bytes`; `pcm16(samples) -> np.ndarray[int16]`, `requantize(samples) -> np.ndarray[float32]` (the stream-side helper), `encode_wav16(samples) -> bytes` (44-byte header, 16 kHz mono); `duration_from_bytes(encoding, sample_rate_hz, byte_count) -> float` |
| `lease` | `SessionFactory` and `SessionLease` (the structural `Protocol`s of the session leaf); `SESSION_FACTORIES_GROUP = "vllm_riva_frontend.session_factories"`, `STREAM_CADENCE = "560ms"`; `discover_session_factory(engine_client, *, selected, entry_points) -> object` (constructs every entry point with `(engine_client, engine_client.model_config)`; raises `StartupError` listing names on zero or several accepts); `resolve_locale(client_locale, factory, *, field) -> str` (raises `FrontendError`); `StreamOwner(*, lease_opener, surface, config, clock, registry, record, catalog)`: the per-stream owner with `open(cadence, locale)`, `feed(samples)`, `update_locale(locale)`, `attach_frontend(frontend)` and `frontend` (the stream's `StreamingAudioFrontend` or `None`), `finalize() -> str` (the frontend's drained tail, `flush`, `finish`, `release`), `async fail(cause)`, `observe_shutdown(trigger)`, `async receive(awaitable)` (one bounded client read; Stream owner, below), `rpc_deadline` (settable), `finalization` (`"client"`, `"shutdown"` or `None`), `shutdown_observed`, `drain_deadline` (`D` once the trigger arrived), `state` (`awaiting_configuration`, `configured`, `streaming`, `finalizing`, `terminal`), `cause` (the one selected terminal cause or `None`), `lease_state` (the owner's view: `open`, `flushed`, `failed`, `aborted`, `released`, `torn_down`), `calls` (the lease calls it made, in order), `bounds` (its `Bounds`), `outstanding_samples` (read-only: the pre-submit credit in use); `Cause(code, phase, deadline, error=None)` (`error` keeps a projected seam error's detail, fields and authority); `ShutdownObserved` (what `receive` raises once the trigger reached the owner); `Bounds` (the owner's timers: `preconfiguration`, `first_audio`, `idle`, `lease_call`, `duration`, `finalization`, `cleanup`, each with `deadline` and `suspended`) |
| `admission` | `IngressCap(*, max_sessions, max_file_owners, serving=None)` (constructed closed; `serving` is the plugin's predicate that core has not begun draining) with `open()`, `close()`, `is_open`, `try_admit(kind) -> Admission | None` (atomic check of serving state and count; `None` means closed or full), `count`, `file_count`, `refusal(kind) -> FrontendError` (`busy` with `authority: load_shed`, or `service_unavailable` when closed); `Admission(kind)` with `release()` (idempotent); `OwnerKind` (a `str` `Enum`: `stream`, `realtime`, `recognize`, `file_route`); `FILE_KINDS` |
| `files` | `FileGate(size, *, clock)` with `async acquire(deadline) -> Permit` (raises `FrontendError("busy")` on expiry), `close()`, `in_flight`, `waiting`; `Permit.release()` (idempotent); `RequestGuard` with `check(request, *, locales) -> FrontendError | None` (beam search, completion limit, sampling fields, ignored fields, `vllm_xargs` keys) and `FORWARDED_SAMPLING_FIELDS`, `IGNORED_FIELDS`, `REQUEST_FIELD_ALLOWLIST` (names and aliases of core's `TranscriptionRequest`); `locale_language(locale) -> str | None`; `recognize_language(locale, supported_languages) -> str | None`; `FileRequestOwner(*, service, engine_client, config, clock, registry, record, catalog, gate=None, accounting=None, gone=None, lifecycle=None, state=None)` (the file gate, what the request holds, a future the client's leaving completes, the plugin and core's app state) with `async run(audio, request, raw_request, *, send=None, deadline=None)` (`send` is the route's ASGI send, `deadline` the RPC's absolute deadline) and `abort_ids`, `cause`, `continuation`; `TranscriptionCall(request_id, ...)`: the child task that calls `create_transcription` and pulls the event generator; `build_raw_request(scope=None) -> Request` (the request-ID fix: a copied scope with one lowercase `x-request-id` header); `file_route(plugin) -> Callable` (the handler installed at `POST /v1/audio/transcriptions`); `recognize_file(plugin, audio_wav, locale, *, context, deadline, admission, record)`: the `Recognize` execution path, which takes over the servicer's admission and record (Admission and registry, below) |
| `multipart` | `MultipartLimits(envelope_max_bytes, content_type_max_bytes, boundary_max_bytes, max_parts, max_header_bytes, text_field_max_bytes)`; `parse_content_type(value: str, limits) -> bytes` (the boundary; raises `FrontendError`); `parse_content_length(headers, limits)`; `MultipartForm(fields: list[tuple[str, str]], file: io.BytesIO | None, filename, content_type)`; `async parse_multipart(receive, boundary, limits, *, deadline, clock) -> MultipartForm` (feeds ASGI chunks to `python_multipart.MultipartParser` callbacks; retains nothing beyond the parts; raises `FrontendError` or `ClientDisconnect`); `make_upload(form) -> UploadFile` (over the form's own `BytesIO`) |
| `grpc` | `RivaServicer(plugin)` implementing `StreamingRecognize` (an async generator), `Recognize` and `GetRivaSpeechRecognitionConfig` (coroutines); `register_aio_services(server, servicer) -> HealthServicer` (servicer, health, reflection); `build_server(config, servicer) -> tuple[grpc.aio.Server, HealthServicer]` (options from `grpc_receive_max_bytes` and `grpc_keepalive_seconds`; the server is unbound: `start` binds `grpc_bind` and starts it; the health servicer is `grpc_health.v1.health.aio.HealthServicer`); `validate_recognition_config(config, *, served_model, locales, default_locale) -> list[FrontendError]`; `FIELD_DISPOSITIONS` (one entry per `RecognitionConfig` field of the pinned stubs); `ENCODING_NAMES`; `RIVA_SERVICE_NAME`; `config_response(plugin) -> RivaSpeechRecognitionConfigResponse` |
| `nim_ws` | `RealtimeDispatcher(app, *, plugin)`: the pure-ASGI middleware `attach_router` adds; `dispatch_decision(scope) -> DispatchDecision` (an `Enum`: `CLAIM` or `PASS`); `bootstrap_session(model_name, default_locale) -> dict`; `RealtimeAdapter(*, plugin, scope, receive, send)` with `async run()`; `project_event(event) -> dict`; `validate_session_update(update, *, served_model, locales, default_locale, first) -> list[FrontendError]`; `EVENT_TYPES` (the client event names); `BOOTSTRAP_PATH = "/v1/realtime/transcription_sessions"`, `REALTIME_PATH = "/v1/realtime"`; `suffix(previous, current) -> str` |
| `surfaces` | What `grpc` and `nim_ws` share (no test imports it): `owner_error(owner)`, `selected_error(owner, error)` (the owner's one selected cause wins on the wire), `async settle(owner, error, phase)`, `child(record, awaitable)` (a task kept in `record.children`), `unregister(registry, record)` (tolerates a record already removed), `locale_error(value, *, field, locales, default_locale)` and `header_problem(header, declared_rate, declared_channels)` (a resolved RIFF header against the declared constraints and the accept-matrix) |
| `operational` | `OPERATIONAL_PATHS` (`/v1/health/ready`, `/v1/health/live`, `/v1/version`, `/v1/metadata`); `operational_router(plugin) -> APIRouter`; `readiness_body(ready) -> tuple[int, dict]`, `liveness_body()`, `version_body()`, `metadata_body(served_model, provenance)` |

`deployment.py` (the Omni-era manifest renderer and its `vllm-riva-frontend-render` script) has no
owner in the designs and is not part of this layout.

### Routing constraints the tests observe

- Every lease call a surface makes goes through `StreamOwner`; the owner serializes them, selects
  the one terminal cause and runs the cleanup table. A test observes the bound of a lease call, the
  suspension of idle timing and the call order on `StreamOwner.calls` and the fake lease's record,
  never on a surface's private state.
- Every lease call and every wait for client input is bounded through the plugin's `clock`
  (`Clock.monotonic`, `Clock.sleep`, `Clock.wait_for`), the one place a test replaces time.
- Every admitted owner is created through `OwnerRegistry.register` after `IngressCap.try_admit`
  returned an `Admission`, and a refusal registers nothing. The fault collector and the listener
  monitor are the only tasks outside the registry.
- Complete files reach core only through `FileRequestOwner` (both surfaces), which calls
  `create_transcription(audio, request, raw_request)` of the service found at
  `state.openai_serving_transcription`, with the request built as the files leaf states, and calls
  `engine_client.abort(request_id)` itself.
- The audio frontend is the only decoder of streaming and `Recognize` audio; `encode_wav16` is the
  only WAV writer; `pcm16` is the only requantization.
- Decoding, resampling and WAV writing run on the event loop, in the handler's own task: the audio
  frontend's `decode`, `drain` and `reset` and `encode_wav16` are synchronous, and no surface hands
  them to an executor or a thread. Ground: audio design, Hand-off to the lease (decoding is
  incremental, in slices of at most `pre_submit_max_samples` decoded samples, over pure NumPy
  tables); the clock seam (`FakeClock.settle` runs the loop until it is idle, which reaches every
  handler's next wait only when no work runs on another thread).
- `discover_session_factory` is the only place that constructs factories; it receives the
  `entry_points` callable from the plugin, so no test patches `importlib.metadata`.
- The dispatcher claims a scope only through `dispatch_decision`; everything it does not claim is
  passed to the wrapped app unchanged.

## Shapes

Where the designs name a behaviour without fixing the Python shape, this section fixes it. Each
entry names its ground: a design section, a spec, a core file or the public client
(`riva/client/realtime.py` of `nvidia-riva/python-clients`, installed as `nvidia-riva-client`).

### Servicer

- `StreamingRecognize(request_iterator, context)` is an async generator (`async def` with `yield`),
  the stream-stream form of `grpc.aio`; it yields `StreamingRecognizeResponse` messages and ends a
  failed stream with `await context.abort(status, details)`, which raises. `Recognize(request,
  context)` and `GetRivaSpeechRecognitionConfig(request, context)` are coroutines returning their
  response message. Ground: the design's "Every error is a gRPC status via `context.abort`"
  (errors leaf, Projection rules); one handler form so one test driver fits every leaf.
- `FrontendError.grpc_details()` renders `<code>: <detail>` and appends ` (fields: <a>, <b>)` only
  when the error names fields; a `busy` error's detail begins with `authority: <authority>`, so
  the gRPC details and the HTTP `message` both begin `busy: authority: load_shed` (or `provider`).
  `FrontendError.fields` is a tuple of names in order. Ground: ING-ERR-029, ING-ERR-051.
- `ErrorCatalog.grpc_status(code)` returns a `grpc.StatusCode`; `FrontendError.http_body(status)`
  returns core's error shape as a plain `dict` (`ErrorResponse(...).model_dump()`), which the route
  sends with that status. Ground: errors design, Projection rules (HTTP).
- `max_alternatives` above 1 is `invalid_config_field` naming `max_alternatives`
  (`INVALID_ARGUMENT`), a scalar field of the disposition table; `audio_channel_count` above 1 is
  `unsupported_format` naming `audio_channel_count`. Ground: streaming design, field disposition
  table and ING-GRPC-017.
- Each RPC reads its deadline once, at servicer entry, as the absolute value `clock.monotonic() +
  context.time_remaining()` (no deadline when `time_remaining()` is `None`), and keeps that value
  and its identity for the rest of the call. Ground: ING-ERR-056 (a deadline keeps its original
  absolute value), ING-FILE-065 (the deadline applies from servicer entry).
- `Recognize` observes that deadline at each validation, permit and terminal boundary; when it has
  expired, the owner selects `request_timeout` and aborts with `DEADLINE_EXCEEDED` (the client may
  already be gone; the abort is still made). The gate wait and core's execution are clipped to it
  and expire as it. Ground: ING-FILE-002, ING-FILE-065, ING-ERR-046.

### Realtime events

- Every server event is a JSON object with `type` and `event_id`. The `error` event is
  `{"type": "error", "event_id": ..., "error": {"code": <catalog code>, "message": "<code>: <detail>",
  "param": "<field>,<field>"?, "authority": <authority>?}}`: `param` is present only when the error
  names fields (comma-joined, in order) and `authority` only for `busy`. The
  `conversation.item.input_audio_transcription.failed` event carries the same `error` object plus
  `item_id` and `content_index: 0`. Ground: the public client reads `event["error"]["message"]`
  (`realtime.py`, the `error` branch of its receive loop); ING-ERR-051 (code, field names,
  `authority`); OpenAI Realtime's `error` object shape (`code`, `message`, `param`).
- `transcription_session.update` is `{"type": "transcription_session.update", "session": {...}}`;
  its acknowledgement is `{"type": "transcription_session.updated", "event_id": ..., "session":
  {...}}` with the honoured session object. Ground: the public client sends `session` and reads
  `response_data["session"]`.
- `conversation.created` is `{"type": "conversation.created", "event_id": ..., "conversation":
  {"id": ..., "object": "realtime.conversation"}}`.
- `conversation.item.input_audio_transcription.delta` carries `delta` (the new text only),
  `item_id`, `content_index: 0`; `...completed` carries `transcript` (the final transcript),
  `is_last_result: true`, `item_id`, `content_index: 0`. `input_audio_buffer.committed` and
  `input_audio_buffer.cleared` carry `item_id`. Ground: the public client reads `delta`,
  `transcript` and `is_last_result`.
- The bootstrap object (`bootstrap_session(model_name, default_locale)`) has exactly the keys `id`
  (`sess_<hex>`), `object` (`realtime.transcription_session`), `modalities` (`["text"]`),
  `input_audio_format` (`pcm16`), `input_audio_params` (`{"sample_rate_hz": 16000, "num_channels":
  1}`), `input_audio_transcription` (`{"model": <served identity>, "language": <default_locale or
  "">}`), `recognition_config` (`{"max_alternatives": 1}`), `speaker_diarization`
  (`{"enable_speaker_diarization": false}`), `word_boosting` (`{"enable_word_boosting": false,
  "word_boosting_list": []}`), `endpointing_config` (`{}`) and `client_secret` (`null`). A client may
  send the public shape's further default-valued keys (`prompt: ""`, the `recognition_config`
  toggles at `false`, `max_speaker_count: 0`); they are dispositioned as default values and accepted.
  Ground: ING-NIMWS-020, ING-NIMWS-013, ING-NIMWS-022.
- The Realtime audio names are `pcm16`, `g711_ulaw` and `g711_alaw`; `nim_ws` maps them to the
  audio frontend's encodings (below). An unknown `input_audio_format` (other than `"none"`) is
  `unsupported_format` naming `input_audio_format`. Ground: audio design, Accept-matrix;
  ING-NIMWS-007. A zero `input_audio_params` value under `"none"` is unconstrained, as a zero
  `RecognitionConfig` field is under `ENCODING_UNSPECIFIED`. Ground: ING-NIMWS-009 by ING-GRPC-018.
- `dispatch_decision(scope) -> DispatchDecision`, an `Enum` in `nim_ws` with members `CLAIM` and
  `PASS`. Ground: the dispatcher section of the streaming design.
- A `base64` payload that does not decode is `invalid_audio` (the `failed` event, close 1008): the
  `audio` field has the right type, so the event is structurally valid and its payload is what
  cannot be decoded. Ground: ING-ERR-022, ING-NIMWS-023.
- Realtime rejections name the dialect's own keys, unqualified: `sample_rate_hz`, `num_channels`,
  `input_audio_format`, `language`, `model`, `prompt`, `modalities`, `enable_profanity_filter`,
  `max_alternatives`; a section rejected as a whole is named by its key (`speaker_diarization`,
  `word_boosting`, `endpointing_config`, `custom_configuration`), an unknown key by itself. The
  format rule is the audio leaf's (below): `input_audio_format` for an unknown name,
  `sample_rate_hz` for a known format at a rate off its cells (G.711 at 16 kHz), `num_channels`
  for non-mono. Ground: streaming design, "Rejections name the dialect's own field names".
- `input_audio_transcription.model` other than empty or the served identity, a non-empty `prompt`,
  `modalities` other than `["text"]`, `enable_word_time_offsets` true and `max_alternatives` above
  1 are `invalid_config_field` (scalar or default-shape fields, as on gRPC); `unsupported_capability`
  is for the capability sections (diarization, word boosting, endpointing). Ground: streaming
  design, field disposition table and its class rule; ING-GRPC-022.
- Every later update ends with exactly one `transcription_session.updated` reflecting the honoured
  subset, after its rejection events; when nothing is honoured it reflects the unchanged session.
  Ground: session design, Locale updates and configuration changes (Realtime); ING-LIFE-009.
- `ErrorCatalog.realtime_close(code)` is the close of the code's catalog row (`None` where the row
  has none). The adapter takes a streaming session's close from the phase table, which overrides
  the catalog where they differ: `service_unavailable` closes 1013 only for a refusal before
  configuration, 1011 after a lease-call timeout or `engine_dead`, and 1012 for the shutdown abort
  row; `internal` closes 1011 and `invalid_audio` 1008. Ground: errors design, Phase-specific
  mapping (authoritative).
- The adapter sends no `delta` for `flush`'s transcript: the final transcript arrives only in the
  `completed` event, and after `release` returns the adapter's first send is that `completed`
  event. The entry of that send is where "the final's send begins" (ING-ERR-057): until then,
  while `release` runs included, the receive observer keeps reading, and a message it sees there
  suppresses the final (`session_terminal`, close 1008) without cancelling the running `release`
  and with no lease call after it. Ground: streaming design, the event mapping ("final after
  release": exactly one `completed`; deltas for changed hypotheses only); ING-NIMWS-023,
  ING-ERR-057, ING-LIFE-028.
- After the first successful update and `open`, a later update's changed non-locale field is
  `config_change_rejected` whatever its value; `unsupported_capability` and `invalid_config_field`
  for section fields arise on the first update only (an unknown key is `invalid_config_field` on
  any update). Ground: ING-LIFE-008, ING-ERR-033.

### Admission and registry

- `OwnerKind` is a `str`-valued `Enum`: `OwnerKind.stream == "stream"`, `realtime`, `recognize`,
  `file_route`; `OwnerKind("stream")` is `OwnerKind.stream`. `FILE_KINDS` is
  `frozenset({OwnerKind.recognize, OwnerKind.file_route})`.
- `IngressCap(*, max_sessions, max_file_owners)` is constructed closed: `is_open` is false and
  `try_admit` returns `None` until `open()` is called. `start` calls `open()` as its last step and
  `drain` calls `close()`; `plugin.admission_open` is `ingress.is_open`. Before `start`, every
  streaming handler refuses with `service_unavailable` (`UNAVAILABLE`; `error` and close 1013).
  Ground: vehicle design, Listener ("opens admission as its last step, so no owner is admitted
  before `start` completes"); ING-EP-035; ING-ADM-011.
- The gRPC health service carries two names: `""` (the server) and `RIVA_SERVICE_NAME`; `start` sets
  both to `SERVING` when admission opens and `drain` sets both to `NOT_SERVING`. Ground: ING-EP-005,
  ING-EP-035.
- `OwnerRegistry()` takes no arguments. `register(kind, task)` returns an `OwnerRecord` with
  `continuation=None` and `children=set()`; `children` is a mutable `set[asyncio.Task]` the owner
  adds to directly; `attach_continuation(record, task)` sets `continuation`. `cancel_all()` cancels
  tasks, continuations and children and returns them. Ground: ING-EP-060, ING-FILE-071.
- `StartupError(failures)` is the exception `init_state` and `start` raise; `ConfigError` is a
  subclass of it raised by `parse_config`, so a configuration failure propagates from `init_state`
  as a `StartupError` whose `failures` list every option. `parse_config` derives
  `file_execution_timeout` (`max(300, 2 * unary_max_decoded_duration_seconds)`, from configuration
  alone) and leaves `max_concurrent_files` and `max_file_owners` at `None` when unset, because they
  depend on the engine's capacity; `init_state` stores the derived-complete `FrontendConfig` on
  `plugin.config`. Ground: vehicle design, configuration table; ING-EP-018, ING-EP-023,
  ING-EP-048, ING-ADM-014.
- `Recognize` admits (`OwnerKind.recognize`) and registers its owner at servicer entry, after the
  capability check and before any decode; a refusal registers nothing. It passes the `Admission`
  and the `OwnerRecord` to `files.recognize_file`, which releases both exactly once, from the
  cleanup continuation when that outlasts the call; until the hand-over the servicer releases them.
  Ground: ING-ADM-017 ("for `Recognize`, at servicer entry before any decode"), ING-ADM-014,
  ING-FILE-071.
- Registry contracts. `OwnerRegistry` carries what an owner needs from the plugin without holding
  it: `torn_down` (owners then call the lease's synchronous `teardown` only), `drain_deadline` and
  `shutdown_trigger` while draining, `on_fault` and `escalate(error)` (an `engine_dead` reason
  escalates to a requested shutdown). `attach_owner(record, owner)` sets the record's shutdown
  observer (a `StreamOwner` attaches itself) and delivers a trigger already raised;
  `deliver_shutdown(trigger)` raises the trigger for every stream owner and cancels every file
  owner (its cancellation path, ING-EP-038); `cancel_owners()` cancels owner tasks only; `pending()`,
  `settled(kinds=None)` and `changed()` let `drain` wait for owners on the clock. Ground: ING-EP-037,
  ING-EP-038, ING-EP-043, ING-EP-060.
- Handlers read the plugin's attributes at request time, never at construction, because
  `attach_router` creates them before `init_state`: the file route reads `files_enabled`,
  `ingress`, `registry`, `file_gate`, `config`, `clock`, `catalog`, `drain_deadline` and
  `torn_down`; `Recognize` also reads `transcription` (core's `openai_serving_transcription`),
  `engine_client` and `state` (core's app state), which `init_state` keeps; the Realtime adapter
  reads `ingress`, `registry`, `factory`, `served_model`, `config`, `clock`, `catalog` and
  `drain_deadline`. Ground: vehicle design, the lifecycle table (`attach_router` runs in
  `build_app`, before `init_state`).
- `--shutdown-timeout` is read from `engine_client.vllm_config.shutdown_timeout`, the value core's
  launcher reads (`vllm/entrypoints/launchers/launcher.py:274`) before it passes the deadline
  `time.monotonic() + shutdown_timeout` to `drain_endpoint_plugins` (`launcher.py:275-276`), which
  hands each plugin's `drain` the seconds left to that deadline
  (`vllm/plugins/endpoint_plugins/interface.py:266-272`). Ground: ING-EP-027.
- `init_state` reads the served model's generation settings (ING-FILE-054) from
  `state.openai_serving_transcription.default_sampling_params`, the mapping core's serving builds
  from `model_config.get_diff_sampling_param()` (`vllm/entrypoints/speech_to_text/base/serving.py:114`).
- `stop`'s 1 s wait for the cancelled tasks runs through `clock.wait_for`. Ground: ING-EP-059; the
  clock seam.

### Stream owner

- `StreamOwner(*, lease_opener, surface, config, clock, registry, record, catalog)`:
  `lease_opener` is the factory's `open` bound method (an async callable `(*, cadence, locale)`),
  `surface` is `"riva"` or `"realtime"`. Ground: session design, Structural protocol.
- Both surfaces read client input only through `owner.receive(awaitable)`, which bounds the read by
  the owner's current client-wait timer (`preconfiguration`, then `first_audio` or `idle` with
  `duration`, then `finalization` for the receive observer) and `rpc_deadline`; expiry selects the
  earliest deadline's cause, cleans up and raises `FrontendError`; once the shutdown trigger reached
  the owner it cancels the pending read and raises `ShutdownObserved`, which the surface answers
  with `finalize()`. `owner.finalization` and `owner.shutdown_observed` choose the Realtime close
  (1001 after a shutdown finalization, 1012 for the shutdown abort row). Ground: session design,
  Bounds; ING-LIFE-040, ING-LIFE-053, ING-ERR-056.
- `owner.state` and `owner.lease_state` compare equal to their names as strings (plain `str` or a
  `str`-valued `Enum`); `owner.calls` is the list of lease-method names called, in order;
  `owner.bounds` is the owner's `Bounds`.
- `owner.outstanding_samples` is the owner's read-only count of samples handed to the lease's
  `feed` whose `on_accepted` has not yet fired: `feed(samples)` adds `len(samples)` when it calls
  the lease, and that call's `on_accepted(n)` callback subtracts `n` when it fires, which the
  model's lease does before `feed` returns. It never waits for `feed` to return, and it is the
  credit `pre_submit_max_samples` bounds. Ground: ING-FE-007; audio design, Hand-off to the lease
  (credit is released from `on_accepted`, not when `feed` returns); the model plugin's session
  design (`on_accepted` fires once, before any unit is sent).
- `owner.frontend` is the stream's `StreamingAudioFrontend`, `None` until the surface attaches it
  (`attach_frontend(frontend)`) once the audio cell is known. `finalize()` drains it (`drain()`)
  and feeds a non-empty tail under the finalization bound before `flush`; with no frontend or an
  empty tail it makes no tail `feed`. Ground: session design, Finalization step 0; ING-LIFE-017.
- A `flush` that returns a non-string marks `lease_state` `flushed` (the call returned) and selects
  `internal`; cleanup is `release` only, with neither `finish` nor `abort`. Ground: ING-LIFE-017,
  ING-LIFE-027 (a flushed lease is never aborted).
- `async fail(cause)`: selects `cause` if no abnormal cause is selected yet (an abnormal cause is
  immutable; a provisional normal or shutdown finalization yields), cancels and awaits the
  settlement of the current `open`, `feed`, `update_locale`, `flush` or `finish` call (never a
  running `release`), then runs the shielded cleanup; it returns after cleanup. The owner method
  whose call it interrupted (`open`, `feed`, `update_locale`, `finalize`) raises `FrontendError`
  with the selected cause's code in its caller, so the surface projects that one cause. Ground:
  session design, One cause and Cleanup order; ING-LIFE-035, ING-LIFE-056.
- A cause selected after `finish` has settled is recorded (`owner.cause`) and suppresses the final:
  `finalize` raises its `FrontendError` instead of returning the transcript. It does not cancel a
  running `release`, which keeps its own bound, and cleanup goes on to `release` only; `fail`
  returns once that `release` has ended. Ground: ING-LIFE-028, ING-LIFE-025.
- "`release` has begun" (ING-LIFE-037) means the owner has created the `release` call as a task
  and let it run to its first suspension before it cancels it. A cleanup budget already spent when
  `abort` is cancelled therefore still starts `release` (it is not handed to `clock.wait_for` with
  the expired deadline, which would cancel it before it starts) and cancels it after that first
  suspension, so the cancellation hands the session to the factory's cleanup (the model plugin's
  SESS-REL-003) rather than discarding a call that never ran. Ground: ING-LIFE-037; the clock seam
  (a coroutine handed to `wait_for` with an expired deadline never starts).
- A send or close that exceeds `response_send_timeout` ends the owner by the transport-failure
  path: no further event and no close frame is sent. Ground: errors design, Riva Realtime
  WebSocket; session design, Bounds (`response_send_timeout`: "no further sends").
- `Cause(code, phase, deadline)`: `phase` is one of the timer names `preconfiguration`,
  `first_audio`, `idle`, `lease_call`, `duration`, `finalization`, `cleanup`, the deadlines
  `rpc_deadline` and `drain`, or the non-deadline phases `transport` (cancellation, disconnect, a
  send bound), `protocol` (a validation or protocol result), `input` (undecodable audio, overflow),
  `engine` (a lease failure), `factory` (an `open` refusal or failure) and `shutdown`; `deadline` is
  the absolute `clock.monotonic()` deadline for a deadline phase and `None` otherwise. Ground: the
  Bounds table and the cause classification of the session design; ING-ERR-056.
- Suspended client-idle timers keep their remaining time (`Bounds.<timer>.suspended`) and resume
  when the call returns; accepted audio resets the idle timer. Ground: ING-LIFE-040, ING-LIFE-041.
- An `open` pending when the shutdown trigger arrives is settled under `preconfiguration_timeout`
  with the wait clipped to `D`: a lease returned before `D` is finalized (final, then `OK` or 1001);
  `D` first cancels `open` under the cancelled-`open` rule and ends the stream with the shutdown
  code. Ground: ING-LIFE-032, ING-LIFE-054.

### Files

- `RequestGuard()` takes no arguments; `check(request, *, locales)` is its one method.
- `parse_content_type(value: str, limits) -> bytes` takes the decoded header value and returns the
  boundary token. It raises `FrontendError("malformed_request")` or
  `FrontendError("request_too_large")`.
- `parse_multipart` raises `FrontendError` (`malformed_request`, `request_too_large`,
  `request_timeout`) for its own refusals and `starlette.requests.ClientDisconnect` when the client
  disconnects mid-body; the route's owner treats the latter as a disconnect (no response, bounded
  memory discarded). Ground: ING-HTTP-007, ING-FILE-007.
- `MultipartForm.file` is an `io.BytesIO` (positioned at 0) that the parser wrote the file part
  into, or `None` when no file part arrived; `make_upload(form)` wraps that same object in core's
  `UploadFile`, so no copy is made. Ground: ING-FILE-041.
- `FileGate.acquire` on a closed gate raises `FrontendError("service_unavailable")`. Ground:
  ING-FILE-053 with ING-ADM-011 (a draining plugin answers `service_unavailable`).
- `catalog.http_status(code)` returns one status per code; for `request_timeout` it is 408 (the
  body-read timeout). The core-execution timeout is the one place that passes 504 to
  `FrontendError.http_body(504)`. Ground: ING-ERR-055.
- A text part that core's `TranscriptionRequest` rejects answers core's validation error: HTTP 400
  in core's error shape, as core's `RequestValidationError` handler answers
  (`vllm/entrypoints/serve/exception_handling/handlers/validation.py`). Ground: ING-FILE-042.
- After response headers, a failure that can be serialized ends the event stream as core's own
  generator ends a failed stream: core's streaming error event (`data: ` and core's error shape, as
  `create_streaming_error_response` writes it), then `data: [DONE]`, and the body is completed
  (`vllm/entrypoints/speech_to_text/base/serving.py:847-852`). Ground: ING-FILE-035.
- `seed` and `watermarking` default to absent, so an explicit value of either is refused whatever
  it is (`seed=0`, `watermarking=false`). Ground: ING-FILE-020, files design, Guard.

### Audio

- `StreamingAudioFrontend(*, encoding, sample_rate_hz)` takes the encoding names `"LINEAR_PCM"`,
  `"MULAW"` and `"ALAW"`; `ACCEPT_MATRIX` is `frozenset({("LINEAR_PCM", 16000, 1),
  ("LINEAR_PCM", 8000, 1), ("MULAW", 8000, 1), ("ALAW", 8000, 1)})` of `(encoding, sample_rate_hz,
  channels)`. `grpc.ENCODING_NAMES` maps the proto enum values to these names and `nim_ws` maps the
  Realtime names. Ground: audio design, Accept-matrix.
- `validate_format(encoding, sample_rate_hz, channels)` names, in this order, `encoding` when it is
  not one of the three names, else `sample_rate_hz` when the rate is not a cell of that encoding,
  and `channels` when not mono; `FormatError.fields` is a tuple. gRPC names the same fields
  `encoding`, `sample_rate_hertz`, `audio_channel_count`. Ground: audio design, Accept-matrix
  ("any other encoding, sample rate or channel count ... naming the offending fields"; G.711 is
  intrinsically 8 kHz).
- `sniff_riff` returns a `RiffFormat` for every parseable RIFF/WAVE header, an off-matrix cell
  included; the caller validates the cell with `validate_format`. It returns `FormatError` only when
  the bytes are not a RIFF/WAVE header, the header is malformed, or no complete header fits in
  `max_header_bytes`; `None` while more bytes are needed. `RiffFormat.data_length` is `None` for the
  sentinel lengths 0 and `0xFFFFFFFF` (audio to the end of the stream) and the declared integer
  otherwise. `RiffFormat.encoding` uses the names above. Ground: streaming design, Header inference;
  ING-GRPC-021.

- `pcm16` and `encode_wav16` requantize in fixed-size slices, and `encode_wav16` writes the samples
  straight into the one buffer its `bytes` is made from, so `Recognize` holds the decoded `float32`
  samples and one WAV, and it drops its copy of the encoded audio once that is decoded. Ground:
  files design, Memory per owner; ING-FILE-043.

### Driving `drain` in tests

Core's `drain_endpoint_plugins` caps the whole drain in real time (`asyncio.timeout`), so a test
under `FakeClock` does not call it. It does what core does around the plugin: set
`state.draining = True`, then await `plugin.drain(state, timeout)` with `timeout` in seconds
(`support.app.drain_plugin`); the plugin converts `timeout` to `clock.monotonic()` deadlines
(`D = now + timeout - engine_reserve(timeout)`), so virtual time drives every wait. Tests of core's
wiring with a real clock use `support.app.drain_plugins`.

## Test seams

| Seam | How a test injects it |
| --- | --- |
| Session factory | `RivaFrontendPlugin(entry_points=...)`: a callable with the signature of `importlib.metadata.entry_points(group=...)` returning objects with `name` and `load()`. `tests/support/fake_session.py` provides `fake_entry_points(**constructors)` and `FakeSessionFactory` (its leases record `fed`, `fed_samples` and `accepted`, count `outstanding` and `peak_outstanding` samples handed to `feed` and not yet acknowledged, report the call in progress as `running`, and run `after(name, callback)` hooks when a call succeeds; `on_accepted` always fires before `feed` returns, as the model's session design requires: at once by default, and under `defer_acceptance` only when the test calls `acknowledge()`, with `feed` still pending, after which `feed` performs its scripted behaviour and returns; a cancelled `release` hands the session to the factory's cleanup task, which `drain(deadline)` awaits until the absolute deadline on the factory's `clock` and `owns(lease)` reports; a failed cleanup task's exception is recorded as a fault, which `drain` returns exactly once); a constructor returns the fake or `None`, or raises. Nothing patches `importlib.metadata`. |
| Core transcription service | The plugin reads `state.openai_serving_transcription` in `init_state`, exactly as core's route does (`vllm/entrypoints/speech_to_text/transcription/api_router.py`). `tests/support/fake_transcription.py` installs `FakeTranscriptionServing` there; the fake mirrors `create_transcription(audio_data, request, raw_request=None)` and derives the request ID the way core's `_create_speech_to_text` does. The served identity comes from `state.openai_serving_models.model_name()` (`FakeServingModels`). |
| Clock | `RivaFrontendPlugin(clock=...)`: a `Clock` with `monotonic()`, `async sleep(seconds)` and `async wait_for(awaitable, deadline)`. The plugin passes it to owners, the file gate and the fault collector; absolute deadlines (`factory.drain`, `ShutdownTrigger`) are `clock.monotonic()` values. `tests/support/clock.py` provides `FakeClock` (virtual time; `advance(seconds)` runs the loop until it is idle, moves time, wakes due sleepers and runs the loop to idle again), a `RealClock` like the production one, and the test's own loop waits `until(condition)` (bounded real time) and `settle(rounds)` (the loop run until idle, at least `rounds` turns). The two clocks' `wait_for` agree: at or after its deadline a done awaitable returns its result and anything else is cancelled before it runs further, so a coroutine handed over with an expired deadline never starts; code that must start a call before cancelling it (ING-LIFE-037) does so itself. A test that needs wall time (a real gRPC listener) uses `RealClock` with short configured bounds. |
| gRPC | Two modes. Every leaf that drives a stream calls the servicer directly through `tests/support/riva.py`: `StreamingCall(servicer, clock, ...)` (the one driver of `StreamingRecognize`: `configure`, `audio`, `half_close`, `wait`, `finish`, `status` (the gRPC status), `code` (the catalog code), `finals` and `interims` (transcripts), `final_responses`), `recognize(servicer, request, context)` (a `RecognizeOutcome`: `response`, `status`, `code`, `details`, `transcript`) and the message builders (`recognition_config`, `config_request`, `audio_request`, `recognize_request`), over `tests/support/grpc_harness.py`'s `FakeServicerContext` (observable `abort`, `cancel()` that fires done callbacks, and `time_remaining()` counting down to the absolute `deadline` fixed at construction on the call's clock, as a real context counts down in wall time) and `RequestStream` (a test-driven request iterator). Vehicle leaf tests of `start`, `ready`, `drain` and `stop` run a real `grpc.aio` server on a free port (`free_port()`, since the design refuses port 0) and talk to it through `aio_channel`. |
| HTTP and WebSocket | `tests/support/app.py` builds a FastAPI app with core's real routers for `/health`, `POST /v1/audio/transcriptions` and `/v1/audio/translations` (present only with the `transcription` task), a recording stand-in for core's `/v1/realtime` WebSocket route, then attaches the plugin through core's real `attach_endpoint_plugins` (discovery replaced by the given plugin list), core's exception handlers and core's middleware (`init_entrypoints_middleware`, so `--api-key` and the middleware order are core's). State is filled as core's `init_app_state` fills it, and `init_endpoint_plugins_state`, `start_endpoint_plugins`, `endpoint_plugins_ready`, `drain_endpoint_plugins` and `stop_endpoint_plugins` are core's real functions; `serve_plugin(plugin, engine, ...)` runs build, state init and `start` in one call and returns a `Served` handle, and `drain_plugin(plugin, state, timeout)` drives one plugin's `drain` under `FakeClock` (Shapes, above). HTTP is exercised with `httpx.ASGITransport`, with the raw ASGI driver `asgi_http` of `tests/support/asgi.py` for chunked bodies and disconnects, or with `tests/support/http.py`'s `HttpExchange` (test-controlled chunk timing, disconnect, blocked and failing sends) and its multipart builders (`multipart_body`, `file_request`, `assert_core_error`). WebSockets use `tests/support/asgi.py`'s in-loop ASGI client `AsgiWebSocket` (`consumed` counts the messages the app has read, `push_json` delivers one without yielding; `BlockingSendWebSocket` stalls the app's sends and keeps the stalled messages in `parked`), and every leaf drives the Realtime dialect through `tests/support/realtime.py`'s `RealtimeClient` (`connect`, `next`, `until_type`, `open_session`, `update`, `append`, `done_event`, `assert_closes`, `drain_events`) and its builders and readers (`default_session`, `update_event`, `append_event`, `error_code`, `error_fields`, `error_authority`). Starlette's `TestClient` runs the app on another loop and is not used. |
| Engine | `tests/support/fake_engine.py`'s `FakeEngineClient` carries `vllm_config` (`additional_config`, `cache_config.num_gpu_blocks`, `scheduler_config.max_num_seqs` and `max_num_active_seqs`, `parallel_config.data_parallel_size`, `shutdown_timeout`), `model_config`, `errored`, `dead_error`, `abort` and `check_health`, with the attribute names of core's `EngineClient` and config classes. |
| Audio | `tests/support/audio.py` generates PCM16, μ-law, A-law and RIFF/WAV bytes in code. A test observes decoding and resampling by wrapping `frontend.StreamingAudioFrontend.decode`, `drain` and `reset` on the class (`monkeypatch.setattr`): the audio frontend is the only decoder and resampler of streaming audio (Routing constraints), so the wrappers see every decode and resample a surface makes, and a refusal that must precede decoding (ING-NIMWS-015, ING-NIMWS-018) shows no `decode` or `drain` call once the session is configured (constructing the frontend and `reset` initialise buffers and do not count). |

## Tests

| Leaf | Directory | Covers |
| --- | --- | --- |
| vehicle | `tests/vehicle/` | declaration, `init_state` order and checks, listener, `ready`, `drain`, `stop`, faults, configuration |
| session | `tests/session/` | discovery, seam conformance (against `FakeSessionFactory`), the stream owner, bounds, cleanup |
| streaming | `tests/streaming/` | the Riva servicer's wire contract, the Realtime dispatcher and adapter |
| files | `tests/files/` | gate, guard, multipart, the HTTP route, `Recognize`, core pins |
| audio | `tests/audio/` | accept-matrix, decoders, carry, resampler, WAV |
| errors | `tests/errors/` | catalog, projections, seam and core mappings |
| operations | `tests/operations/` | the four routes, coexistence, authentication |

Each directory has an `__init__.py` and its own `conftest.py`; shared fixtures live in
`tests/conftest.py` (`clock`, `fake_factory` (on `clock`), `owner_parts` and `make_owner` (a
`StreamOwner` built directly from the layout's constructor arguments over `fake_factory`,
registered for the current task), `riva_options`, `fake_engine` (with `grpc_bind` on a
free port and `riva_options`), `fake_models`, `fake_transcription`, `core_state`, `entry_points`,
`plugin` (the frontend on `clock` and `entry_points`), `plugins`, `supported_tasks`, `test_app`,
`app_state`, `started` (core's `start` and `stop` around the test); a leaf overrides `plugins`
with `[plugin]`), and the fakes and drivers in `tests/support/` (importable as `support.*`;
`pythonpath = ["tests"]`). A leaf's `conftest.py` holds only what is specific to that leaf (its
plugin fixture, its options, its oracles); anything two leaves need is in `support`.
`tests/test_support.py` tests the support code itself and imports no frontend module.

Run from the repository root with the Riva stubs and the core tree on `PYTHONPATH`:

```text
python -m pytest -q tests
```

Markers: `gpu` (an installed engine on a CUDA device; skipped without one), `engine` (imports the
core tree; skipped when it does not import), `real_factory` (the seam conformance suite against the
model plugin's factory; skipped when it is not installed). Unmarked tests are CPU tests over the
fakes. Asyncio mode is `auto` (`pytest-asyncio`): `async def` tests need no marker, and each test
gets its own event loop.
