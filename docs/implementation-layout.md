# Implementation layout

The module layout the tests are written against. The designs and specs under `docs/intent/` state
the behaviour; this file names where it lives so tests and code agree. Changing a name here means
changing the tests that use it. Where a leaf design names a module or symbol, that name is used
here; where today's module keeps its role, its name is kept, so the port rewrites modules in place.

Core is vLLM with RFC 2 (endpoint plugins) merged (`pdtgct/vllm`). The frontend needs nothing else
of core; the session seam belongs to the model plugin. Core
paths below are relative to that tree.

## Package `vllm_riva_frontend`

| Module | Public names |
| --- | --- |
| `__init__` | `plugin()` (entry point `riva_frontend = vllm_riva_frontend:plugin` in group `vllm.endpoint_plugins`): a zero-argument factory returning `RivaFrontendPlugin`; `__version__` |
| `version` | `__version__` (the distribution version `/plugins/riva/v1/version` reports) |
| `clock` | `Clock` (the structural `Protocol` of the clock seam: `monotonic()`, `async sleep(seconds)`, `async wait_for(awaitable, deadline)`, which awaits until the absolute `deadline` and otherwise cancels the awaitable and raises `TimeoutError`, as `asyncio.wait_for` does; `timer(deadline) -> Future[None]`, a future resolved at the absolute `deadline` by a timer of the event loop, with no task behind it, already resolved when the deadline has passed, and cancelling the future cancels the timer); `RealClock` (`time.monotonic` and asyncio), the clock `RivaFrontendPlugin(clock=None)` uses |
| `lifecycle` | `RivaFrontendPlugin(*, clock=None, entry_points=None)` (`clock=None` is `clock.RealClock()`) with class attributes `name = "riva_frontend"`, `required_tasks = ("realtime",)`, and the lifecycle methods `attach_router(app)`, `async init_state(engine_client, state, args)`, `async start(state)`, `ready(state) -> bool`, `async drain(state, timeout)`, `async stop(state)`; attributes tests read after `init_state`: `config` (a `FrontendConfig`), `served_model` (the identity string), `deadlines` (a `StreamDeadlines`), `capacity` (a `Capacity`), `files_enabled`, `factory`, `registry` (an `OwnerRegistry`), `ingress` (an `IngressCap`), `file_gate` (a `FileGate` or `None`), `catalog` (an `ErrorCatalog`), `servicer` (a `RivaServicer`), `provenance` (the one `DeploymentProvenance` every surface reads), `transcription` (core's transcription service or `None`), `state` (core's app state), `engine_client`, `auth` (an `AuthState`); after `start`: `server` (the `grpc.aio.Server`), `health` (the gRPC `HealthServicer`), `admission_open`, `fault_collector`, `listener_monitor` (tasks); throughout: `clock`, `draining`, `torn_down`, `fault` (the recorded fault or `None`), `drain_deadline` (`D`, set only while a `drain` call runs its graceful work), `drain_announced() -> Future` (resolved at the next change of the send bounds: when `drain` sets `D`, when the send bounds come to count as passed (the settlement begins, the graceful steps are skipped, or `stop` runs) and when the `drain` call returns and they stop counting; already resolved when it is asked for while they count as passed), `send_bounds_passed` (true from `stop` on and, in a `drain` call, from the settlement or the skipped graceful steps until that call returns), `listener_stopping`. `OwnerRegistry()` with `register(kind, task) -> OwnerRecord`, `unregister(record)`, `records` (a mapping), `attach_continuation(record, task)`, `unowned_sends` (the refusal sends of no record), `adopt_send(task) -> bool` (False once closed), `close_sends()`, `sends_closed`, `outstanding()`, `cancel_owners() -> list[Task]` (cancels each owner once, through `OwnerRecord.cancel_by_plugin()`), `discard_ended()`, and what owners need from the plugin without holding it (Registry contracts, below); `OwnerRecord(kind, task, continuation=None, children=set(), owner=None, held_by_handler=False, unsettled_sends=set(), cancelled_by_plugin=False)` (`children` is a mutable `set[Task]`; `owner` is the record's shutdown observer; `held_by_handler` and `unsettled_sends`: Registry contracts, below; `cancel_by_plugin()` sets `cancelled_by_plugin` and then cancels `task`, so an owner tells the plugin's cancellation from its transport's by that mark); `ShutdownTrigger(deadline)`, the object `drain` raises for stream owners at the settle point; `engine_reserve(timeout)`, `settle_point(now, D)` (the two formulas of the vehicle leaf); `FaultCollector(factory, period=1.0, on_fault=None, *, clock=None, lock=None)` with `async run()` and `request_stop()` (no argument: the collector waits on `clock.timer` and a stop signal, ends when the signal arrives and is never cancelled; a collection call in progress completes first), `ListenerMonitor(server, on_exit, expected=None)`; `StartupError` (the exception `init_state` and `start` raise, with `failures: list[str]`) |
| `config` | `FrontendConfig` (one attribute per option of the vehicle leaf's table, including the derived `max_concurrent_files`, `max_file_owners` and `file_execution_timeout`); `parse_config(additional_config) -> FrontendConfig` (reads key `riva_frontend`; raises `ConfigError` listing every failure; derives `file_execution_timeout`, which depends on configuration only, and leaves `max_concurrent_files` and `max_file_owners` `None` when unset, since they depend on the engine's capacity); `ConfigError(failures)`, a subclass of `lifecycle.StartupError`; `StreamDeadlines(never_fed_timeout, parked_idle_timeout, draining_timeout)`, `read_stream_deadlines(additional_config) -> StreamDeadlines`, `DEFAULT_STREAM_DEADLINES` (15, 60, 40); `check_bounds(config, deadlines) -> list[str]`; `Capacity(slots, active_limit)`, `read_capacity(vllm_config) -> Capacity`; `derive_file_caps(config, capacity, service_present) -> tuple[int, int, bool]` (`max_concurrent_files`, `max_file_owners`, files enabled); `DeploymentProvenance` and `build_deployment_provenance(config)` (the four typed fields of the metadata route); `PLUGIN_KEY = "riva_frontend"`, `STREAM_ADMISSION_KEY = "stream_admission"` |
| `errors` | `ErrorCatalog` (the catalog of the errors leaf: `codes`, `grpc_status(code)`, `http_status(code)`, `realtime_close(code)`); `catalog()`; `FrontendError(code, detail="", fields=(), authority=None)` (the exception every surface projects; `grpc_details()` renders `<code>: <detail> (fields: <names>)`, `http_body(status)` renders core's error shape); `seam_error(exc) -> tuple[str | None, str | None]` (`getattr` of `session_error` and `reason`); `project_seam_error(exc) -> FrontendError` (the seam reason mapping); `classify_core_exception(exc) -> FrontendError` and `project_error_response(response) -> FrontendError` (the two file-path tables); `ESCALATES = frozenset({"engine_dead"})` |
| `frontend` | `ACCEPT_MATRIX`, `RESAMPLER_ID = "scipy-poly-v1"`, `validate_format(encoding, sample_rate_hz, channels) -> FormatError | None`, `FormatError(fields, detail)`, `RiffFormat(encoding, sample_rate_hz, channels, data_offset, data_length)`, `sniff_riff(buffer, *, max_header_bytes) -> RiffFormat | FormatError | None` (`None` means more bytes are needed), `mulaw_table()`, `alaw_table()`, `StreamingAudioFrontend(*, encoding, sample_rate_hz)` with `decode(data, *, max_samples) -> Iterator[np.ndarray]` (whole samples, resampled, sliced to at most `max_samples`), `drain() -> np.ndarray` (the resampler lookahead; the byte residual is discarded), `reset()` (the Realtime `clear`), `residual_bytes`; `pcm16(samples) -> np.ndarray[int16]`, `requantize(samples) -> np.ndarray[float32]` (the stream-side helper), `OUTPUT_RATE = 16_000` (the rate of every sample handed on); `encode_wav16(samples) -> bytes` (44-byte header, 16 kHz mono); `duration_from_bytes(encoding, sample_rate_hz, byte_count) -> float` |
| `lease` | `SessionFactory` and `SessionLease` (the structural `Protocol`s of the session leaf); `SESSION_FACTORIES_GROUP = "vllm_riva_frontend.session_factories"`, `STREAM_CADENCE = "560ms"`; `discover_session_factory(engine_client, *, selected, entry_points, discard) -> object` (`discard` is required; constructs every entry point with `(engine_client, engine_client.model_config)`, whether or not `selected` names one, and raises `StartupError` if any constructor raises; `selected` picks which result is used; raises `StartupError` listing names on zero or several accepts, on an absent or declining `selected`, or when the chosen factory does not serve `STREAM_CADENCE`; it never settles a factory itself: it hands every constructed factory it does not return to `discard`, and the plugin settles each one, ING-EP-069); `resolve_locale(client_locale, factory, *, field) -> str` (raises `FrontendError`); `StreamOwner(*, lease_opener, surface, config, clock, registry, record, catalog)`: the per-stream owner with `open(cadence, locale)`, `feed(samples)`, `update_locale(locale)`, `attach_frontend(frontend)` and `frontend` (the stream's `StreamingAudioFrontend` or `None`), `finalize() -> str` (the frontend's drained tail, `flush`, `finish`, `release`), `async fail(cause)`, `observe_shutdown(trigger)`, `async receive(awaitable)` (one bounded client read; Stream owner, below), `trigger_taken` (a future, `None` before the shutdown trigger, resolved once the owner has acted on it: `drain` waits for it, bounded by `D`, before it stops the listener), `handed_over` (a cancelled `release` left the session to the factory), `escalations` and `on_escalation` (the `engine_dead` failures the owner met and the plugin's callback for them), `cleanup_errors` (secondary failures recorded during cleanup), `rpc_deadline` (settable), `finalization` (`"client"`, `"shutdown"` or `None`), `shutdown_observed`, `drain_deadline` (`D` once the trigger arrived), `state` (`awaiting_configuration`, `configured`, `streaming`, `finalizing`, `terminal`), `cause` (the one selected terminal cause or `None`), `lease_state` (the owner's view: `open`, `flushed`, `failed`, `aborted`, `released`, `torn_down`), `calls` (the lease calls it made, in order), `bounds` (its `Bounds`), `outstanding_samples` (read-only: the pre-submit credit in use), `samples_fed` (read-only: the 16 kHz samples of successful `feed` calls, the drained tail included); `Cause(code, phase, deadline, error=None)` (`error` keeps a projected seam error's detail, fields and authority); `ShutdownObserved` (what `receive` raises once the trigger reached the owner); `Bounds` (the owner's timers: `preconfiguration`, `first_audio`, `idle`, `lease_call`, `duration`, `finalization`, `cleanup`, each with `deadline` and `suspended`) |
| `admission` | `IngressCap(*, max_sessions, max_file_owners, serving=None)` (constructed closed; `serving` is the plugin's predicate that core has not begun draining) with `open()`, `close()`, `is_open`, `try_admit(kind) -> Admission | None` (atomic check of serving state and count; `None` means closed or full), `count`, `file_count`, `refusal(kind) -> FrontendError` (`busy` with `authority: load_shed`, or `service_unavailable` when closed); `Admission(kind)` with `release()` (idempotent); `OwnerKind` (a `str` `Enum`: `stream`, `realtime`, `recognize`, `file_route`); `FILE_KINDS` |
| `files` | `FileGate(size, *, clock, serving=None)` (`serving` is the plugin's predicate that core has not set `state.draining`) with `async acquire(deadline, *, spawn=None) -> Permit` (raises `FrontendError("busy")` on expiry and `FrontendError("service_unavailable")` on a closed gate; `spawn` starts the wait's timer task under the caller's record), `close()`, `closed` (true once closed, or while `serving()` is false, so a gate refuses new waiters while core drains even with no plugin `drain` call), `in_flight`, `waiting`, `size`; `Permit.release()` (idempotent); `RequestGuard` with `check(request, *, locales) -> FrontendError | None` (beam search, completion limit, sampling fields, ignored fields, `vllm_xargs` keys) and `FORWARDED_SAMPLING_FIELDS`, `IGNORED_FIELDS`, `REQUEST_FIELD_ALLOWLIST` (names and aliases of core's `TranscriptionRequest`); `locale_language(locale) -> str | None`; `recognize_language(locale, supported_languages) -> str | None`; `FileRequestOwner(*, service, engine_client, config, clock, registry, record, catalog, gate=None, accounting=None, gone=None, lifecycle=None, state=None)` (the file gate, what the request holds, a future the client's leaving completes, the plugin and core's app state) with `async run(audio, request, raw_request, *, send=None, deadline=None)` (`send` is the route's ASGI send, `deadline` the RPC's absolute deadline) and `abort_ids`, `cause` (a `FileCause(kind, error, status, exception, deadline)`), `continuation`, `request_id`, `result`, `suppressed`, `cancelled_by_task`; the owner takes a cancellation of its task as the plugin's only from the plugin's own signals: `record.cancelled_by_plugin`, a torn-down plugin, send bounds counting as passed and, for `Recognize`, `listener_stopping`; core's `state.draining`, the plugin's drain flag, `D` alone and closed admission do not make a cancellation the plugin's (ING-FILE-089); `TranscriptionCall(request_id, ...)`: the child task that calls `create_transcription` and pulls the event generator; `build_raw_request(scope=None) -> Request` (the request-ID fix: a copied scope with one lowercase `x-request-id` header); `file_route(plugin) -> Callable` (the handler installed at `routes.FILE_ROUTE_PATH`, `POST /plugins/riva/v1/audio/transcriptions`, beside core's own `POST /v1/audio/transcriptions`, which it leaves unchanged); `wav_seconds(audio_wav) -> float` (the WAV's duration from its header); `recognize_file(plugin, audio_wav, locale, *, context, deadline, admission, record)`: the `Recognize` execution path, which takes over the servicer's admission and record (Admission and registry, below) |
| `multipart` | `MultipartLimits(envelope_max_bytes, content_type_max_bytes, boundary_max_bytes, max_parts, max_header_bytes, text_field_max_bytes)`; `parse_content_type(value: str, limits) -> bytes` (the boundary; raises `FrontendError`); `parse_content_length(headers, limits)`; `MultipartForm(fields: list[tuple[str, str]], file: io.BytesIO | None, filename, content_type)`; `async parse_multipart(receive, boundary, limits, *, deadline, clock, spawn=None) -> MultipartForm` (feeds ASGI chunks to `python_multipart.MultipartParser` callbacks; retains nothing beyond the parts; the read and its timer run as two tasks started by `spawn`, so an owner keeps both under its record; raises `FrontendError` or `ClientDisconnect`); `make_upload(form) -> UploadFile` (over the form's own `BytesIO`) |
| `grpc` | `RivaServicer(plugin)` implementing `StreamingRecognize` (a coroutine that writes with `context.write`), `Recognize` and `GetRivaSpeechRecognitionConfig` (coroutines); `register_aio_services(server, servicer) -> HealthServicer` (servicer, health, reflection); `build_server(config, servicer) -> tuple[grpc.aio.Server, HealthServicer]` (options from `grpc_receive_max_bytes` and `grpc_keepalive_seconds`; the server is unbound: `start` binds `grpc_bind` and starts it; the health servicer is `grpc_health.v1.health.aio.HealthServicer`); `validate_recognition_config(config, *, served_model, locales, default_locale) -> list[FrontendError]`; `FIELD_DISPOSITIONS` (one entry per `RecognitionConfig` field of the pinned stubs); `ENCODING_NAMES`; `RIVA_SERVICE_NAME`; `config_response(plugin) -> RivaSpeechRecognitionConfigResponse` |
| `nim_ws` | `RealtimeRoute(plugin)`: the ASGI app of the Realtime WebSocket route `routes.REALTIME_PATH`, which serves a connection whose query is exactly `intent=transcription` and refuses any other query before accepting it; `bootstrap_endpoint(plugin)`: the endpoint of the bootstrap `POST`; `bootstrap_session(model_name, default_locale) -> dict`; `RealtimeAdapter(*, plugin, scope, receive, send)` with `async run()`; `project_event(event) -> dict`; `validate_session_update(update, *, served_model, locales, default_locale, first) -> list[FrontendError]`; `EVENT_TYPES` (the client event names); `suffix(previous, current) -> str` |
| `surfaces` | What `grpc` and `nim_ws` share (tests import it only to exercise `bounded_send` and to check helper signatures; the surfaces' behaviour is observed through `grpc` and `nim_ws`): `owner_error(owner)`, `selected_error(owner, error)` (the owner's one selected cause wins on the wire), `async settle(owner, error, phase)`, `child(record, awaitable)` (a task kept in `record.children`), `unregister(registry, record)` (tolerates a record already removed), `hold(record)` (sets `held_by_handler`), `release_owner(registry, record, admitted)` (a handler's end: at once, or from the done callback of a running `record.continuation` or unsettled send), `torn_down(plugin)`, `send_bounds_passed(plugin)`, `async join(tasks)` and `async settle_shielded(record, owner, error, phase)` (cleanup joined through cancellation), `async bounded_send(plugin, send, *, record=None, rpc_deadline=None, terminal=False)` (one send under ING-LIFE-061's bound; `TimeoutError` past it), `locale_error(value, *, field, locales, default_locale)` and `header_problem(header, declared_rate, declared_channels)` (a resolved RIFF header against the declared constraints and the accept-matrix) |
| `operational` | `OPERATIONAL_PATHS` (`/plugins/riva/v1/health/ready`, `/plugins/riva/v1/health/live`, `/plugins/riva/v1/version`, `/plugins/riva/v1/metadata`); `operational_endpoints(plugin) -> list[tuple[str, handler]]` (the path and handler of each of the four `GET` routes, which read the plugin at request time); `readiness_body(ready) -> tuple[int, dict]`, `liveness_body()`, `version_body()`, `metadata_body(served_model, provenance)` |
| `routes` | `PLUGIN_BASE = "/plugins/riva"`, `PLUGIN_PREFIX = "/plugins/riva/v1"`, `AUTH_SCOPE_PREFIX = "/plugins/riva/"`, `BOOTSTRAP_PATH`, `REALTIME_PATH`, `FILE_ROUTE_PATH`, `CORE_FILE_ROUTE_PATH` (core's `/v1/audio/transcriptions`); `FrontendRoute`, `http_route(method, path, endpoint)`, `websocket_route(path, app)`; `core_serves_files(app)`; `collisions(app, routes) -> list[str]` (checked before a route is added), `add_routes(app, routes)`, `shadowed(app, routes) -> list[str]` (checked in `init_state` against core's final table); every frontend route is under `PLUGIN_PREFIX` |
| `auth` | `PathScopedAuthMiddleware(app, *, auth)`: the pure-ASGI middleware `attach_router` adds, which applies core's `--api-key` check to every HTTP and WebSocket request whose path, after `root_path`, starts with `AUTH_SCOPE_PREFIX` and touches no other path; `AuthState`; `in_scope(scope) -> bool`; `configured_keys`, `decide_tokens` |

### Routing constraints the tests observe

- Every lease call a surface makes goes through `StreamOwner`; the owner serializes them, selects
  the one terminal cause and runs the cleanup table. A test observes the bound of a lease call, the
  suspension of idle timing and the call order on `StreamOwner.calls` and the fake lease's record,
  never on a surface's private state.
- Every lease call and every wait for client input is bounded through the plugin's `clock`
  (`Clock.monotonic`, `Clock.sleep`, `Clock.wait_for`, `Clock.timer`), the one place a test replaces time.
- Every admitted owner is created through `OwnerRegistry.register` after `IngressCap.try_admit`
  returned an `Admission`, and a refusal registers nothing. The fault collector and the listener
  monitor are the only supervision tasks outside the registry; the other tasks outside it are a
  refusal's send, awaited inline by its own handler, which registers it (`adopt_send`) only if it
  abandons it, and the lifecycle helper tasks (`_run_to_end`'s work, a `factory.drain(D)` call
  running past `D`) that the enclosing lifecycle invocation retains and awaits through
  cancellation before it exits (ING-EP-106). No wait starts an unregistered task for a deadline: it is a
  `Clock.timer` future, or a task started under the owner's record (the file gate's wait, the
  multipart read).
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
  `entry_points` callable from the plugin, so no test patches `importlib.metadata`; a factory it
  constructs and does not return goes to its required `discard` callback, and the plugin settles
  it (ING-EP-069).
- The path-scoped authentication middleware decides scope only through `auth.in_scope`; a request
  outside `AUTH_SCOPE_PREFIX` passes to the wrapped app unchanged.

## Shapes

Where the designs name a behaviour without fixing the Python shape, this section fixes it. Each
entry names its ground: a design section, a spec, a core file or the public client
(`riva/client/realtime.py` of `nvidia-riva/python-clients`, installed as `nvidia-riva-client`).

### Servicer

- `StreamingRecognize(request_iterator, context)` is a coroutine, the reader-writer stream-stream
  form of `grpc.aio`: it writes each `StreamingRecognizeResponse` with `await context.write(...)`
  and ends a failed stream with `await context.abort(status, details)`, which raises.
  `Recognize(request, context)` and `GetRivaSpeechRecognitionConfig(request, context)` are
  coroutines returning their response message. Ground: the design's "Every error is a gRPC status
  via `context.abort`" (errors leaf, Projection rules); ING-LIFE-061 (a write the servicer awaits
  is one it can bound, which a yielded response is not); one handler form so one test driver fits
  every leaf.
- Every write and every `context.abort` (the pre-owner refusal of ING-ERR-015 included) runs
  through `surfaces.bounded_send`: `response_send_timeout` from the send's start, clipped to the
  RPC's absolute deadline and, while a `drain` call runs its graceful work, to `D` (the plugin's
  `drain_deadline`); core's `state.draining` alone clips nothing. Once the send bounds count as
  passed (below) the bound has passed. Ground: ING-LIFE-061, ING-ADM-016, ING-ERR-064,
  ING-ERR-056, ING-EP-110.
- Drain announcement: `drain` sets `drain_deadline` and then resolves every future handed out by
  `RivaFrontendPlugin.drain_announced()`; each later change of the send bounds resolves the futures
  handed out since (the settlement beginning or the graceful steps skipped, `stop`, and the end of
  the `drain` call, when the bounds stop counting as passed). A waiting send includes one in its
  wait, with or without an owner, so a send already waiting when the drain begins re-clips to `D`
  at once, not at the settle point, and one waiting when the settlement begins fails at once; the
  trigger itself is delivered to the `StreamOwner`, which attaches itself as the record's shutdown
  observer. Ground: ING-LIFE-061; session design, Bounds ("Sends").
- Send rule. A send whose bound has already passed when it is due is not started (the coroutine
  is closed) and takes the transport-failure path, unless it is `terminal=True`: a status
  (`context.abort`) here, a terminal `error` event and its close on the Realtime adapter. A
  terminal send starts, runs to its first suspension and is cancelled if not complete, adding no
  wait, so a status the transport takes at once (the ING-ERR-062 abort row at `D`,
  `request_timeout` after the RPC deadline) is still made. A terminal attempt covers a whole
  response: the Realtime terminal `error` (or `failed`) event and its close, and a refusal's
  whole sequence, run in sequence as one attempt under one bound, with no wait between the
  messages. Writes (hypotheses, the final) are ordinary sends. Every bound counts as passed once
  `stop` has marked the plugin torn down and, for the rest of a `drain` call, once its settlement
  began, `D` has passed or it skipped its graceful steps (ING-EP-110): an ordinary send does not
  start and a terminal one gets only that attempt. When the `drain` call returns the bounds stop
  counting as passed (never on a torn-down plugin), so a send started afterwards gets the
  ordinary `response_send_timeout` again. A send not complete
  when its bound passes has failed, even if the transport completes it or suppresses the
  cancellation before the waiting coroutine resumes (only a terminal send's one attempt succeeds
  by being done at its first suspension). The judgement runs in the send's own task, as its last
  step, when the send completes: whether it completed before its bound decides, not when the
  waiting coroutine resumes, so a send that completes in time is delivered even when the drain
  notice wakes the caller in the same loop pass. The caller judges inline only a send cancelled
  before it ran, which never reached that step. A failed send is cancelled and given one loop turn; the handler never
  waits for it to settle. Its task is a child of the stream's record (`record=`), and before that
  turn it is put in `record.unsettled_sends`, which holds the record until it settles or the
  settlement awaits it, so a cancellation of the handler during the turn cannot orphan it; one that settles
  in the turn leaves the set again. The same applies when the handler is cancelled while a send
  waits, before its bound is seen to pass: the cancellation then propagates and the owner takes
  the shutdown abort row (its status gets the one attempt). Once the failure is established (the
  bound seen to pass, or a terminal attempt incomplete at its first suspension), it is final: for
  an owner's send, a cancellation of the handler during the abandonment's turn, or one that finds
  the terminal attempt incomplete, is absorbed and withdrawn (`Task.uncancel`), `bounded_send`
  still raises `TimeoutError`, and the handler takes the transport-failure path (next bullet), so
  no further send follows, the close included. Owner cleanup then starts at once under its one
  `session_cleanup_timeout` budget. A refusal handler cancelled during its terminal attempt
  instead retains the cancellation, abandons the send, and re-raises the cancellation once the
  send is registered or, with the registry closed, has ended (`surfaces._last_attempt`). A
  refusal has no record: its handler
  awaits the send inline and owns it; on abandonment it registers the send with
  `registry.adopt_send(task)` while the registry is open, and once the settlement closed the
  registry it awaits the send itself, through any number of cancellations. A waiting send's
  deadline is a timer of the clock (`Clock.timer`), not a task, so a refusal starts nothing
  outside the registry but its own send (ING-EP-106); `stop`, which makes every bound count as
  passed, wakes the wait through the drain announcement. Ground: ING-LIFE-061; session design,
  Bounds ("Sends"); ING-EP-093, ING-EP-095.
- A send past its bound, or a failed write, takes the transport-failure path: the owner is failed
  with phase `transport` (cleanup as for a disconnect, ING-LIFE-038), nothing more is written,
  and the handler ends with no status (it raises `CancelledError`; a refusal with no owner just
  ends so). The HTTP/2 stream then stays open until the client cancels, its deadline passes,
  transport keepalive finds the peer dead or the server stops; the owner, its slot and its lease
  are already released. Ground: ING-LIFE-061, ING-LIFE-038; session design, Bounds ("Sends").
- Record lifetime. Both streaming surfaces register the record and mark it `held_by_handler`
  (`surfaces.hold`), so the end of the owner's cleanup does not remove it: the record stays
  registered until the handler, its `record.continuation` and its `unsettled_sends` have all
  ended, and the drain and `stop` reach a final, error or close still being sent. A handler's
  cleanup runs as a child of its record and is joined (`surfaces.settle_shielded`) through any
  further cancellation, which is absorbed unless the plugin is torn down; the handler then
  releases its record and admission with `surfaces.release_owner`, which leaves both to a running
  continuation (Stream owner, "Cancellation settlement is bounded") or unsettled send. The same
  holds on the Realtime adapter. Ground: ING-LIFE-063, ING-EP-093; session design,
  Bounds ("Sends").
- `FrontendError.grpc_details()` renders `<code>: <detail>` and appends ` (fields: <a>, <b>)` only
  when the error names fields; a `busy` error's detail begins with `authority: <authority>`, so
  the gRPC details and the HTTP `message` both begin `busy: authority: load_shed` (or `provider`).
  `FrontendError.fields` is a tuple of names in order. Ground: ING-ERR-064, ING-ERR-051.
- `ErrorCatalog.grpc_status(code)` returns a `grpc.StatusCode`; `FrontendError.http_body(status)`
  returns core's error shape as a plain `dict` (`{"error": {"message", "type", "param", "code"}}`,
  built without importing core), which the route sends with that status. Ground: errors design, Projection rules (HTTP).
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
- The Realtime route serves exactly the query `intent=transcription` and refuses any other query
  before it accepts the connection (a close with 1008, one terminal attempt), creating no owner or
  lease and taking no ingress-cap count. Ground: ING-NIMWS-026, ING-NIMWS-027; the Realtime route
  section of the streaming design.
- Every adapter send and close (the refusal of ING-NIMWS-028 included) runs through
  `surfaces.bounded_send` with the bound, drain announcement, send rule and record lifetime of the
  Servicer section (no RPC deadline): clipped to `D` while a `drain` call runs its graceful work,
  re-clipped when the drain begins, and, once the send bounds count as passed, offered without
  waiting only for the terminal `error` (or `failed`) event and its close, sent in sequence as one
  attempt, so the 1012 abort row reaches a transport that takes it at once and a stalled one is
  abandoned at its bound. Other sends past it (`completed`, acknowledgements) are not started; a
  refusal's `websocket.accept`, `error` and close run as one terminal attempt. Ground:
  ING-LIFE-061, ING-EP-110, ING-ERR-051, ING-ERR-062.
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
- `IngressCap(*, max_sessions, max_file_owners, serving=None)` is constructed closed: `is_open` is
  false and `try_admit` returns `None` until `open()` is called, and, while `serving()` is false
  (core has set `state.draining`), whatever its state. `start` calls `open()` as its last step;
  `drain`, `stop` and a recorded fault call `close()` (a recorded fault closes the file gate too);
  `plugin.admission_open` is `ingress.is_open`. Before `start`, every
  streaming handler refuses with `service_unavailable` (`UNAVAILABLE`; `error` and close 1013).
  Ground: vehicle design, Listener ("opens admission as its last step, so no owner is admitted
  before `start` completes"); ING-EP-080; ING-ADM-011.
- The gRPC health service carries two names: `""` (the server) and `RIVA_SERVICE_NAME`; `start` sets
  both to `SERVING` when admission opens and `drain` sets both to `NOT_SERVING`. Ground: ING-EP-005,
  ING-EP-080.
- `OwnerRegistry()` takes no arguments. `register(kind, task)` returns an `OwnerRecord` with
  `continuation=None` and `children=set()`; `children` is a mutable `set[asyncio.Task]` the owner
  adds to directly; `attach_continuation(record, task)` sets `continuation` (a file request's
  cleanup continuation, or a stream owner's that outlived its answer). `held_by_handler` (set by
  the streaming surfaces) makes the handler's `surfaces.release_owner`, not the end of the
  `StreamOwner`'s cleanup, remove the record; `unsettled_sends` are the stream's sends cancelled at
  their bound and not yet settled, which hold the record (Servicer, "Record lifetime").
  `adopt_send(task)` registers the send of a refusal that created no owner, abandoned while the
  registry is open, until it ends (it takes no ingress-cap count); `close_sends()` ends that, and
  `sends_closed` reports it. `cancel_owners()` cancels each owner task once, through
  `record.cancel_by_plugin()`, and skips one already being cancelled; it cancels no continuation,
  child or send. `outstanding()` is everything the settlement awaits: owner tasks, continuations,
  children (a stream's unsettled sends among them), unsettled sends and the sends of no owner.
  Ground: ING-EP-093, ING-EP-094, ING-EP-095, ING-FILE-079.
- `StartupError(failures)` is the exception `init_state` and `start` raise; `ConfigError` is a
  subclass of it raised by `parse_config`, so a configuration failure propagates from `init_state`
  as a `StartupError` whose `failures` list every option. `parse_config` derives
  `file_execution_timeout` (`max(300, 2 * unary_max_decoded_duration_seconds)`, from configuration
  alone) and leaves `max_concurrent_files` and `max_file_owners` at `None` when unset, because they
  depend on the engine's capacity; `init_state` stores the derived-complete `FrontendConfig` on
  `plugin.config`. Ground: vehicle design, configuration table; ING-EP-018,
  ING-EP-048, ING-ADM-014.
- `Recognize` admits (`OwnerKind.recognize`) and registers its owner at servicer entry, after the
  capability check and before any decode; a refusal registers nothing. It passes the `Admission`
  and the `OwnerRecord` to `files.recognize_file`, which releases both exactly once, from the
  cleanup continuation when that outlasts the call; until the hand-over the servicer releases them.
  Ground: ING-ADM-018 ("for `Recognize`, at servicer entry before any decode"), ING-ADM-014,
  ING-FILE-079.
- Registry contracts. `OwnerRegistry` carries what an owner needs from the plugin without holding
  it: `torn_down`, `drain_deadline` (`D`, only while a `drain` call runs its graceful work),
  `send_bounds_passed` and `shutdown_trigger` while draining, `on_fault` and `escalate(error)` (an
  `engine_dead` reason escalates to a requested shutdown). No owner calls a lease method for the
  plugin's stopping: the lease's cleanup is `abort`/`release` as its state allows, and a session
  left to the factory is the factory settlement's (ING-EP-096). `attach_owner(record, owner)` sets the record's shutdown
  observer (a `StreamOwner` attaches itself) and delivers a trigger already raised;
  `deliver_shutdown(trigger)` raises the trigger for every stream owner and cancels every file
  owner by `cancel_by_plugin()` (its cancellation path, ING-EP-093); `cancel_owners()` cancels
  owner tasks only; `pending()`,
  `settled(kinds=None)` and `changed()` let `drain` wait for owners on the clock. Ground: ING-EP-037,
  ING-EP-093, ING-EP-106.
- Handlers read the plugin's attributes at request time, never at construction, because
  `attach_router` creates them before `init_state`: the file route reads `files_enabled`,
  `ingress`, `registry`, `file_gate`, `config`, `clock`, `catalog`, `drain_deadline`,
  `send_bounds_passed` and `torn_down`; `Recognize` also reads `listener_stopping`, `transcription` (core's `openai_serving_transcription`),
  `engine_client` and `state` (core's app state), which `init_state` keeps; the Realtime adapter
  reads `ingress`, `registry`, `factory`, `served_model`, `config`, `clock`, `catalog` and
  `drain_deadline`. Ground: vehicle design, the lifecycle table (`attach_router` runs in
  `build_app`, before `init_state`).
- `--shutdown-timeout` is read from `engine_client.vllm_config.shutdown_timeout`, the value core's
  launcher reads (`vllm/entrypoints/launchers/launcher.py:331`) before it passes the deadline
  `time.monotonic() + shutdown_timeout` to `drain_endpoint_plugins` (`launcher.py:332-334`), which
  hands each plugin's `drain` the seconds left to that deadline
  (`vllm/plugins/endpoint_plugins/interface.py:279-293`). Ground: ING-EP-027.
- `init_state` reads the served model's generation settings (ING-FILE-054) from
  `state.openai_serving_transcription.default_sampling_params`, the mapping core's serving builds
  from `model_config.get_diff_sampling_param()` (`vllm/entrypoints/speech_to_text/base/serving.py:114`).
- `stop` marks the plugin torn down, closes admission, makes every send bound count as passed and
  announces that, then runs the settlement with no time limit and no graceful phase: every await
  in it is cooperative, with no wait that a deadline cuts short, and it ends by cancelling and
  awaiting the listener monitor. Ground: ING-EP-079, ING-EP-093.
- The graceful steps of `drain` end in two waits, each bounded by the time left to `D` and neither
  cancelling anything: the fault collector is signalled with `request_stop()` and awaited, and
  `factory.drain(D)` runs as a task of its own under the factory lock. When `D` passes first the
  wait ends and the settlement begins; the settlement still awaits the collector and that
  `factory.drain(D)` call, uncancelled, before the factory settlement (which takes precedence over
  `D`). A rolled-back `start` is the one place that cancels the collector. Ground: ING-EP-066,
  ING-EP-093, ING-EP-100.
- A cancellation of `init_state` while discovery's discarded factories are being settled lets that
  settlement finish, settles the selected factory too, and then propagates: the plugin holds no
  factory. Ground: ING-EP-069, ING-EP-097.

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
  immutable; a provisional normal or shutdown finalization yields), cancels and awaits the bounded
  settlement of the current `open`, `feed`, `update_locale`, `flush` or `finish` call (never a
  running `release`), then runs the shielded cleanup; it returns after cleanup, or at the
  cleanup bound with a cancelled call unsettled (next bullet). The owner method
  whose call it interrupted (`open`, `feed`, `update_locale`, `finalize`) raises `FrontendError`
  with the selected cause's code in its caller, so the surface projects that one cause. Ground:
  session design, One cause and Cleanup order; ING-LIFE-035, ING-LIFE-056.
- Cancellation settlement is bounded. The cleanup timer starts when the cleanup does, so the
  settlement of the cancelled call, `abort` and `release` share one `session_cleanup_timeout`
  budget (clipped to `D` while draining); an expired budget still lets a cancelled call run one
  loop turn to settle. A cancelled call (the interrupted operation's, or an `abort` or `release`
  cancelled at the bound) still unsettled at the bound does not hold the answer: the owner
  selects `internal` (phase `cleanup`) unless a cause stands, becomes `terminal`, and its method
  raises (or `fail` returns) the selected cause. The cleanup task becomes the record's
  continuation (`registry.attach_continuation(record, task)`). It waits, unbounded, for the call
  to settle, starts no other lease call before then, runs the remaining cleanup with the spent
  budget (each call started, then cancelled after its first suspension) and unregisters the record
  when it ends, unless the record is `held_by_handler`, whose handler removes it (Servicer, "Record
  lifetime"); the plugin's settlement (`drain` or `stop`) awaits it and cancels nothing in it. A surface whose handler
  ends while `record.continuation` is running releases its `Admission`, and unregisters the
  record, from that task's done callback, not at once. Ground: ING-LIFE-035, ING-LIFE-060,
  ING-LIFE-063 (the record persists until its last cleanup step), ING-EP-106.
- A cause selected after `finish` has settled is recorded (`owner.cause`) and suppresses the final:
  `finalize` raises its `FrontendError` instead of returning the transcript. It does not cancel a
  running `release`, which keeps its own bound, and cleanup goes on to `release` only; `fail`
  returns once that `release` has ended. Ground: ING-LIFE-028, ING-LIFE-025.
- "`release` has begun" (ING-LIFE-060) means the owner has created the `release` call as a task
  and let it run to its first suspension before it cancels it. A cleanup budget already spent when
  `abort` is cancelled therefore still starts `release` (it is not handed to `clock.wait_for` with
  the expired deadline, which would cancel it before it starts) and cancels it after that first
  suspension, so the cancellation hands the session to the factory's cleanup (the model plugin's
  SESS-REL-003) rather than discarding a call that never ran. Ground: ING-LIFE-060; the clock seam
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
- `FileGate.acquire` on a closed gate raises `FrontendError("service_unavailable")`. The gate is
  closed by `close()` (the plugin's `drain`, a recorded fault) and reads closed whenever its
  `serving` predicate is false, so an upload admitted before core set `state.draining` that
  reaches the gate after it is refused as a new waiter and gets no permit, with no `drain` call
  needed. A waiter already queued at the gate when it closes is not refused: it keeps its place
  until it is granted a permit or cancelled. Ground:
  ING-FILE-053, ING-EP-082 with ING-ADM-011 (a draining plugin answers `service_unavailable`).
- `catalog.http_status(code)` returns one status per code; for `request_timeout` it is 408 (the
  body-read timeout). The core-execution timeout is the one place that passes 504 to
  `FrontendError.http_body(504)`. Ground: ING-ERR-055.
- A text part that core's `TranscriptionRequest` rejects answers core's validation error: HTTP 400
  in core's error shape, as core's `RequestValidationError` handler answers
  (`vllm/entrypoints/serve/exception_handling/handlers/validation.py`). Ground: ING-FILE-042.
- After response headers, a failure that can be serialized ends the event stream as core's own
  generator ends a failed stream: core's streaming error event (`data: ` and core's error shape, as
  `create_streaming_error_response` writes it), then `data: [DONE]`, and the body is completed
  (`vllm/entrypoints/speech_to_text/base/serving.py:853-858`). Ground: ING-FILE-035.
- `seed` and `watermarking` default to absent, so an explicit value of either is refused whatever
  it is (`seed=0`, `watermarking=false`). Ground: ING-FILE-020, files design, Guard.
- At each operation boundary (the permit, the return of core's call, each pull) `FileRequestOwner`
  arbitrates before it consumes the operation's outcome: the client's leaving first, then the
  earliest expired deadline among the RPC's and the operation's own bound, the own bound winning a
  tie. A core result, failure or event that becomes visible together with an expired deadline
  therefore ends as `request_timeout`. Ground: ING-FILE-068, ING-FILE-065, ING-ERR-056.
- A file route's response is sent per ING-FILE-091: the start and the body of a response sent
  before headers are one terminal attempt once the send bounds count as passed. Otherwise the
  messages leave back to back from one task and each message's bound runs from the completion of
  the one before; a message that completes at or after its own bound has failed the send, no
  further message starts, and a late completion advances nothing. A failed send ends delivery.
  Ground: ING-FILE-091, ING-EP-110.
- Every request that reaches `FileRequestOwner.run`, including one ended before the permit, gets
  a cleanup continuation registered with its record. It settles the route's disconnect watcher
  (a task in `record.children`) together with the call's children, and holds the permit, the
  file-owner slot, the load count and the record until they settle, and the plugin's settlement awaits it. Ground:
  ING-FILE-079, ING-FILE-078.

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
| Clock | `RivaFrontendPlugin(clock=...)`: a `Clock` with `monotonic()`, `async sleep(seconds)`, `async wait_for(awaitable, deadline)` and `timer(deadline)` (a future the clock resolves at the deadline, no task behind it). The plugin passes it to owners, the file gate and the fault collector, which wait on `timer` between collections (as a refusal send does for its deadline) instead of starting a task; absolute deadlines (`factory.drain`, `ShutdownTrigger`) are `clock.monotonic()` values. `tests/support/clock.py` provides `FakeClock` (virtual time; `advance(seconds)` runs the loop until it is idle, moves time, wakes due sleepers and timers and runs the loop to idle again), a `RealClock` like the production one, and the test's own loop waits `until(condition)` (bounded real time) and `settle(rounds)` (the loop run until idle, at least `rounds` turns). The two clocks' `wait_for` agree: at or after its deadline a done awaitable returns its result and anything else is cancelled before it runs further, so a coroutine handed over with an expired deadline never starts; code that must start a call before cancelling it (ING-LIFE-060) does so itself. A test that needs wall time (a real gRPC listener) uses `RealClock` with short configured bounds. |
| gRPC | Two modes. Every leaf that drives a stream calls the servicer directly through `tests/support/riva.py`: `StreamingCall(servicer, clock, ...)` (the one driver of `StreamingRecognize`: `configure`, `audio`, `half_close`, `wait`, `finish`, `status` (the gRPC status), `code` (the catalog code), `finals` and `interims` (transcripts), `final_responses`), `recognize(servicer, request, context)` (a `RecognizeOutcome`: `response`, `status`, `code`, `details`, `transcript`) and the message builders (`recognition_config`, `config_request`, `audio_request`, `recognize_request`), over `tests/support/grpc_harness.py`'s `FakeServicerContext` (observable `abort`, `cancel()` that fires done callbacks, and `time_remaining()` counting down to the absolute `deadline` fixed at construction on the call's clock, as a real context counts down in wall time) and `RequestStream` (a test-driven request iterator). Vehicle leaf tests of `start`, `ready`, `drain` and `stop` run a real `grpc.aio` server on a free port (`free_port()`, since the design refuses port 0) and talk to it through `aio_channel`. |
| HTTP and WebSocket | `tests/support/app.py` builds a FastAPI app with core's real routers for `/health`, `POST /v1/audio/transcriptions` and `/v1/audio/translations` (present only with the `transcription` task), a recording stand-in for core's `/v1/realtime` WebSocket route, then attaches the plugin through core's real `attach_endpoint_plugins` (discovery replaced by the given plugin list), core's exception handlers and core's middleware (`init_entrypoints_middleware`, so `--api-key` and the middleware order are core's). State is filled as core's `init_app_state` fills it, and `init_endpoint_plugins_state`, `start_endpoint_plugins`, `endpoint_plugins_ready`, `drain_endpoint_plugins` and `stop_endpoint_plugins` are core's real functions; `serve_plugin(plugin, engine, ...)` runs build, state init and `start` in one call and returns a `Served` handle, and `drain_plugin(plugin, state, timeout)` drives one plugin's `drain` under `FakeClock` (Shapes, above). HTTP is exercised with `httpx.ASGITransport`, with the raw ASGI driver `asgi_http` of `tests/support/asgi.py` for chunked bodies and disconnects, or with `tests/support/http.py`'s `HttpExchange` (test-controlled chunk timing, disconnect, blocked and failing sends) and its multipart builders (`multipart_body`, `file_request`, `assert_core_error`). WebSockets use `tests/support/asgi.py`'s in-loop ASGI client `AsgiWebSocket` (`consumed` counts the messages the app has read, `push_json` delivers one without yielding; `BlockingSendWebSocket` stalls the app's sends and keeps the stalled messages in `parked`), and every leaf drives the Realtime dialect through `tests/support/realtime.py`'s `RealtimeClient` (`connect`, `next`, `until_type`, `open_session`, `update`, `append`, `done_event`, `assert_closes`, `drain_events`) and its builders and readers (`default_session`, `update_event`, `append_event`, `error_code`, `error_fields`, `error_authority`). Starlette's `TestClient` runs the app on another loop and is not used. |
| Engine | `tests/support/fake_engine.py`'s `FakeEngineClient` carries `vllm_config` (`additional_config`, `cache_config.num_gpu_blocks`, `scheduler_config.max_num_seqs` and `max_num_active_seqs`, `parallel_config.data_parallel_size`, `shutdown_timeout`), `model_config`, `errored`, `dead_error`, `abort` and `check_health`, with the attribute names of core's `EngineClient` and config classes. |
| Audio | `tests/support/audio.py` generates PCM16, μ-law, A-law and RIFF/WAV bytes in code. A test observes decoding and resampling by wrapping `frontend.StreamingAudioFrontend.decode`, `drain` and `reset` on the class (`monkeypatch.setattr`): the audio frontend is the only decoder and resampler of streaming audio (Routing constraints), so the wrappers see every decode and resample a surface makes, and a refusal that must precede decoding (ING-NIMWS-015, ING-NIMWS-018) shows no `decode` or `drain` call once the session is configured (constructing the frontend and `reset` initialise buffers and do not count). |

## Tests

| Leaf | Directory | Covers |
| --- | --- | --- |
| vehicle | `tests/vehicle/` | declaration, `init_state` order and checks, listener, `ready`, `drain`, `stop`, faults, configuration |
| session | `tests/session/` | discovery, seam conformance (against `FakeSessionFactory`), the stream owner, bounds, cleanup |
| streaming | `tests/streaming/` | the Riva servicer's wire contract, the Realtime route and adapter |
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
