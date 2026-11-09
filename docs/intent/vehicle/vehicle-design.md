---
parent: high-level-design
prefix: ING
facets: [EP, VEH, ADM]
---

# Vehicle

## Context and Design Philosophy

The vehicle is the part of the frontend that vLLM core sees: one `EndpointPlugin`
(`RivaFrontendPlugin`) and the lifecycle methods core calls on it. It owns the package and its
declaration, the order of startup checks, the registry of owner tasks, the ingress cap, the
relation between the frontend's time bounds and the engine's stream deadlines, shutdown, fault
escalation, and the frontend's configuration. Everything a surface does with a stream, a file or a
wire format belongs to another leaf; this leaf decides when those parts exist and when they stop.

Principles:

- **Fail at startup, name the option.** A server that cannot serve correctly does not start; the
  error names the option or check that failed, and a configuration error lists every failing
  option, not only the first.
- **Every task has an owner until it ends.** The frontend starts no task outside its registry.
  Engine-facing tasks belong to the model plugin's factory.
- **Bounded everywhere.** Every read, wait and lease call has a finite bound, and during shutdown
  every bound is the time left to one deadline.
- **Recover, do not latch.** A fault withdraws readiness and asks core for an orderly restart; no
  state outlives its cause.
- **Core's lifecycle, not a parallel one.** The frontend implements the lifecycle methods core
  defines (`attach_router`, `init_state`, `start`, `ready`, `drain`, `stop`) and adds none.

## Package and declaration

The frontend is distributed as the package `vllm-riva-frontend` (import package
`vllm_riva_frontend`). Its one entry point is `riva_frontend` in the `vllm.endpoint_plugins` group,
resolving to a zero-argument factory that returns `RivaFrontendPlugin`. No Riva- or NIM-shaped
code, dependency or route is merged into vLLM.

The class declares:

| Attribute | Value | Reason |
| --- | --- | --- |
| `name` | `riva_frontend` | log name; `VLLM_PLUGINS` allowlists the entry-point name |
| `required_tasks` | `("realtime",)` | the streaming surfaces need the model's `realtime` task; the CPU-only render server has no engine and does not load the plugin |
| `owns_non_http_listener` | `True` | the gRPC listener makes core require one API server process |

Normal runtime dependencies are the Riva stubs (`nvidia-riva-client`, pinned), `grpcio` with its
health and reflection packages, `numpy`, `scipy` (the resampler) and `python-multipart` (the
file-route parser). The package has no resolver-owned dependency on vLLM, which the host
environment supplies, and none on the model plugin. It imports vLLM only inside the lifecycle
methods and the file route, where the contract it uses is core's. It adds no vLLM version bound:
the capability checks in `init_state` decide whether a given build is usable (Decisions).

**Interpreter range.** The plugin is imported into vLLM's own interpreter, so it declares the
interpreter range vLLM declares (`requires-python` of the pinned core: `>=3.11,<3.15`) and stays
installable and importable across all of it. A narrower declaration is a defect even when every
current host runs one version. Every module compiles and imports on the lowest supported version,
using no syntax or name introduced later (a version-gated import or a declared compatibility
package is permitted; an unguarded one is not). That floor is proven by a gate that compiles and
imports every module on the lowest supported version, reports every offending module rather than
stopping at the first, and fails closed; a gate whose exit status a pipeline can mask, or that
reports one failure per module, does not count. The linter target version and the type checker's
language level name the lowest supported version, so a later-version-only construct fails review.
Versions in the range that no gate runs are recorded as declared-but-unverified, not reported as
supported. The record is the README's "Supported Python versions" section: a table with columns
`Python` and `Status` and one row per minor version of the declared range, in order, whose status
is `supported` when a CI workflow's `python-version` matrix executes that version and
`declared-but-unverified` otherwise.

## One process, one engine

When `riva_frontend` is loaded, one `vllm serve` API-server process serves, over exactly one engine
client and one session factory:

- core's `/v1/realtime` route, unchanged for every scope the dispatcher does not claim;
- the frontend's WebSocket dispatcher, file route, Realtime bootstrap and operational routes, on the
  HTTP listener;
- the Riva servicer, on one separate gRPC listener.

The frontend creates no remote provider, provider-mode selector, internal WebSocket loopback,
second render or generation loop, dedicated batcher or second engine. Cross-session batching is
entirely the engine scheduler's.

The gRPC server is `grpc.aio.server()` with the servicer registered through the generated
`add_RivaSpeechRecognitionServicer_to_server`; no OS thread is committed per concurrent streaming
session for that session's duration. Every streaming adapter passes its validated cadence and
locale only to `SessionFactory.open` and carries no per-session override through a process-shared
object. Engine rendering and the streaming integration stay inside the model plugin's session; the
frontend renders no prompt and passes no transport-owned configuration to the engine. (The exact
locale the file path sends in `vllm_xargs` is request data the model defines, not transport-owned
configuration.)

## Lifecycle methods

Core calls the methods in this order; the frontend's work in each is fixed here.

| Method | Called | Frontend work |
| --- | --- | --- |
| `attach_router(app)` | `build_app`, after core's routers | add the operational and bootstrap routes; the file route only when core's `POST /v1/audio/transcriptions` is present; the WebSocket dispatcher as middleware; fail on a path collision |
| `init_state(engine_client, state, args)` | `init_app_state`, engine running | the startup checks below, then build the owner registry, ingress cap, file gate, servicer and error catalog |
| `start(state)` | lifespan startup | bind and start the gRPC server, start the fault collector and listener monitor, then open admission |
| `ready(state)` | `/health` and the operational readiness route | attribute reads only |
| `drain(state, timeout)` | graceful shutdown, concurrently with other plugins | the bounded sequence below |
| `stop(state)` | after HTTP stops | terminal and prompt |

### `attach_router`

The frontend adds the Realtime bootstrap (`POST /v1/realtime/transcription_sessions`) and the four
operational `GET` routes as ordinary routes. It adds the file route when core's
`POST /v1/audio/transcriptions` is registered, and replaces nothing otherwise (files leaf). It does
not touch `POST /v1/audio/translations`.

The dispatcher is added with `app.add_middleware`. `build_app` runs plugin hooks before core's
middleware, and Starlette puts later middleware outermost, so core's authentication middleware wraps
the dispatcher and `--api-key` applies to every claimed scope. (Core's authentication covers paths
under `/v1`, which includes the plugin's operational, bootstrap and file routes, and the claimed
WebSocket path; it does not cover the gRPC listener.) The dispatcher's pass-through and
authentication behaviour is part of the frontend's CPU test set.

Before adding a route, the frontend checks `app.routes` for an existing operation with the same
path and method among the paths it adds as new (operations leaf) and raises, which stops startup,
rather than rely on registration order.

### `init_state`: startup checks

`init_state` validates in this order and raises one error naming every failed option within a step:

1. **Configuration** from `engine_client.vllm_config.additional_config["riva_frontend"]` (the
   configuration section below).
2. **Served identity**: `state.openai_serving_models.model_name()` is called once; its non-empty
   string result is the plugin lifetime's immutable served-model identity. The serving-model
   registry is the only identity authority: the frontend does not read the engine's `ModelConfig`,
   `state.args` or a serve-argument namespace, and does not parse CLI aliases. A missing registry,
   a missing or non-callable accessor, a raised lookup, or a non-string or empty result fails
   startup before the listener starts or any lease opens. Because the registry returns the first
   base-model name, the Riva gRPC and Realtime WebSocket surfaces expose only that primary
   identity: the canonical checkpoint identifier (when different) and every secondary alias are not
   additional selectors there. The file route's `model` field is validated by core against every
   served name, as core's own route does.
3. **Engine stream deadlines**: the three `stream_admission` deadlines (below) are read, and each
   frontend bound is checked against its engine deadline.
4. **Capacity**: data parallel size must be 1 (the model requires parallel sizes of 1, and slot
   count and active limit are read per engine), otherwise startup fails. The state-slot count and
   the active-request limit are then read (below).
5. **Complete-file service**: if `state.openai_serving_transcription` is present, the file gate and
   file-owner sub-cap are sized against the capacity of step 4, the multipart body bound is checked
   against core's upload limit, and `unary_max_decoded_duration_seconds` is checked against core's
   decode-duration limit (files leaf). If it is absent, complete files are not served: `Recognize`
   answers `UNIMPLEMENTED` and no route is replaced. If `min(slots, active limit)` is below 2 and
   `max_concurrent_files` is not configured, complete files are disabled with a startup warning:
   one files-enabled flag becomes false, `Recognize` answers `UNIMPLEMENTED`, no `offline` entry is
   listed, and the frontend's route (already registered by `attach_router`) refuses with
   `unsupported_capability`, HTTP 501. An explicitly configured `max_concurrent_files` that fails
   the sizing check fails startup.
6. **Build** (everything except the factory): registry, ingress cap, file sub-cap, gate, error
   catalog. Then check the route table: the `app` kept from `attach_router` is searched, and
   `init_state` fails startup unless core's final table (after every plugin attached) still routes
   each path the frontend added or replaced to the frontend's own handler.
7. **Session factory, last**: discovery, the single-accept rule and the `560ms` cadence check
   (session leaf), then the servicer, which needs the factory. Any failure after the factory has
   been constructed calls `factory.teardown_all()` before the error propagates, so a failed
   `init_state` leaves no engine-facing task. No listener is bound and no other task is started in
   `init_state`.

If `--shutdown-timeout` is 0, `init_state` logs a warning that no `drain` will run and that
shutdown is `stop` alone.

#### Engine deadlines and the frontend's bounds

The engine ends an admitted stream that is never fed, parked idle or draining past its deadlines
(`never_fed_timeout`, `parked_idle_timeout`, `draining_timeout`). The frontend wants its own
timeouts to be below the engine's, so that the engine's abort (reported by the model plugin as the
lease failure `expired`) is a backstop and not the normal idle path. The frontend cannot guarantee
that its bound fires first: sub-unit traffic (including accepted audio shorter than one unit before
the engine's never-fed deadline) and client clocks suspended during lease calls can let the engine
end a stream first, and the engine's `expired` then projects as an idle timeout.

`init_state` reads the deadlines from `vllm_config.additional_config["stream_admission"]`, the
mapping core's state scheduler reads. A missing section or key means the documented defaults
(`never_fed_timeout` 15 s, `parked_idle_timeout` 60 s, `draining_timeout` 40 s); a present value
that is not a finite positive number fails startup, because it means the frontend cannot know the
deadline the engine enforces. The engine-client protocol does not declare these settings, so this
mapping is the contract; the frontend's copy of the defaults is pinned by a test against the core
build.

Each frontend bound is checked against the matching engine deadline and must be strictly below it:

| Frontend bound | Starts at | Matching engine deadline | Default | Engine default |
| --- | --- | --- | --- | --- |
| `first_audio_timeout` | successful `open` | `never_fed_timeout` | 10 s | 15 s |
| `session_idle_timeout` | successful `open`, reset by accepted audio | `parked_idle_timeout` | 45 s | 60 s |
| `session_finalization_timeout` | accepted finalization | `draining_timeout` | 30 s | 40 s |
| `lease_call_timeout` | a `feed` or `update_locale` call starts | `parked_idle_timeout` | 45 s | 60 s |

The configuration, cleanup and duration bounds have no engine counterpart. The defaults are chosen
so that the default deployment passes these checks.

The engine is the authority for sub-unit traffic. The frontend's idle bound resets on any accepted
audio, but the engine's parked-idle clock resets only when a complete unit is sent. A client that
delivers less than one unit of audio (560 ms at the streaming cadence) per engine idle window can
therefore be ended by the engine (`expired`, projected as an idle timeout) even though the
frontend's bound would not have fired. The startup check compares the bounds as above and does not
close this gap; the seam is unchanged.

#### Capacity

The state-slot count is the engine's page count minus the reserved null page:
`vllm_config.cache_config.num_gpu_blocks - 1`, which core updates in the API server process after
the engine reports its cache configuration. The active-request limit is
`vllm_config.scheduler_config.max_num_active_seqs`, or `max_num_seqs` when that is unset. The file
gate size must be below both (files leaf). If either value is not yet available (the page count is
`None` or 0), startup fails: a server whose capacity is unknown cannot size a gate.

## Owners, ingress cap and bounded ingress

**Owner registry.** The plugin holds one registry of transport owner tasks: one per gRPC stream
(`StreamingRecognize`), one per Riva Realtime WebSocket session, one per `Recognize` call and one
per file-route request. A handler registers its owner task before it reads its first message or
request body and keeps it registered until its last cleanup step: for a stream, `SessionLease.release`
(or the factory's handover of a cancelled `open`); for a file request, the release of its file-gate
permit after its children have settled. The registry holds one record per owner. A file
request's cleanup runs as a cleanup continuation registered with its record: when the route or RPC
answers before settlement, the continuation keeps the record, with its permit, file-owner slot and
load count, until it finishes or `stop` cancels it. Refusals register nothing. The frontend starts no task outside the registry, apart from the
two long-lived supervision tasks (fault collector, listener monitor) that `start` creates and `stop`
cancels. The model plugin's factory owns every engine-facing task of its sessions.

**Ingress cap.** One finite positive cap (`load_shed_max_sessions`) counts active owners across
`StreamingRecognize`, `Recognize`, the Realtime WebSocket and the file route together. It excludes
core's `/v1/realtime`, `GetRivaSpeechRecognitionConfig`, the bootstrap POST and the operational
routes. Before a handler reads its first message or body (for `Recognize`, whose request is already in
memory under the framework cap, at servicer entry before any decode), parses multipart or calls
`open`, it atomically checks that the plugin is serving (`state.draining` unset, admission open) and that the
count is below the cap, and registers its owner when both hold. A draining plugin answers
`service_unavailable`; a full cap answers retryable `busy` with `authority: load_shed`. Below the
cap the counter claims no residency, reserves no engine capacity and queues nothing, and sessions
already open continue unchanged. This cap bounds frontend memory and work; the file gate (files
leaf) separately bounds files in inference, and the engine's admission (below) bounds streams.

**File sub-cap.** File owners (gate waiters, body readers and executing files, for `Recognize` and
the route) also count against their own sub-cap `max_file_owners`, which sits inside the ingress cap
and must be below it. A file request arriving at a full sub-cap answers `busy`
(`authority: load_shed`) while streams keep the remaining headroom (`load_shed_max_sessions -
max_file_owners` at least), so a file backlog never refuses a stream with `busy`.

**Memory bound.** Frontend audio memory is bounded per owner and in total.

- A file owner holds at most 128 MiB (files leaf, Memory per owner: the parsed part, the upload
  object, core's joined bytes for the route; the encoded request, decoded samples and WAV for
  `Recognize`) at the default configuration; the bound is derived from configuration (files leaf).
  The default file total is `max_file_owners` times that: 3 to 6 owners, because the derived gate
  is between 1 and 4, reaching 4 only when the smaller of slots and active limit is at least 16;
  at most 768 MiB.
- A stream owner holds one received message plus a decode slice. Decoding is incremental: the audio
  frontend decodes a message in slices whose decoded output is at most `pre_submit_max_samples`
  samples, so the decoded form never exceeds the pre-submit bound (about 1 MB with resampler
  temporaries). The received message is bounded by `grpc_receive_max_bytes` (32 MiB) or, for the
  Realtime WebSocket, the message text and its base64-decoded bytes (16 MiB plus 12 MiB at
  `ws_receive_max_bytes`). A stream owner is therefore at most about 32 MiB.
- The stream total is bounded by the ingress cap's stream capacity times that bound. Stream
  headroom is not a maximum: with no file owners the cap admits all 64 streams, so the stream-only
  worst case (every stream sending a maximal message at once) is about 2 GiB at the defaults;
  typical messages are kilobytes, and operators who need a smaller worst case lower the receive
  caps. The file figure excludes core's own decode memory and assumes the default caps; larger
upload or duration limits raise it as the files leaf's formula gives.

The default `max_file_owners` is `max_concurrent_files + 2`, and the default `max_concurrent_files`
is capped at 4. While complete files are disabled or core's transcription service is absent, both
are 0: no file gate is built, and `Recognize` and the file route refuse without creating an owner.

**Engine refusals.** When `open` raises a `SessionRefused` with reason `capacity` or `active_limit`,
the surface consumes no audio and answers `busy` carrying `authority: provider`. Riva invents no
success acknowledgement and the Realtime configuration acknowledgement is not relabelled as
admission. The other seam reasons are mapped by the errors leaf.

**Bounded ingress.** Every read, wait and send of an admitted owner is finite: the first message
(`preconfiguration_timeout`), a file request's body read (`http_body_read_timeout`), a file-gate
wait (`file_gate_wait_timeout`) and each send (`response_send_timeout`, on every surface). A client
disconnect releases what the owner held: its multipart memory and, for a stream, its lease through
the session leaf's cleanup, and, once its children have settled (files leaf), its registry entry,
its ingress-cap count, its permit and its file-owner slot.

## Listener (`start`)

`start` builds one `grpc.aio.server()` with the Riva servicer, the standard gRPC health service and
server reflection registered, and sets health to `NOT_SERVING`. It binds the configured address (a
port of 0 is refused, because the bound port must be known) and starts the server; after the bind
the listener is locally reachable before `start` returns. Then `start` creates the fault collector and the listener monitor, and opens admission as its last
step; health becomes `SERVING` when admission opens, so no owner is admitted before `start`
completes.

If `start` fails after the bind, it cancels the fault collector and the listener monitor if it
created them (so the rollback's listener exit is not reported as a fault), closes the listener
(`stop(0)`), calls `factory.teardown_all()` and re-raises. Startup rollback
after a later plugin fails is core's `stop` call on this plugin, so `stop` is correct for a plugin
that started.

`ready(state)` reads attributes only and is true while the listener is serving, the plugin is not
draining or torn down, and no cleanup fault is recorded. Core's `/health` and the operational
readiness route both read it.

## Shutdown

### `drain(state, timeout)`

`drain` computes the core deadline, `now + timeout`, and reserves time for core to shut the engine
down: the plugin finishes by `D = deadline - engine_reserve`, where `engine_reserve = max(5 s, 10%
of timeout)`. Every wait inside `drain`, including each lease call's bound and the factory wait, is
the time remaining to `D`. The vehicle owns only the shutdown trigger and `D`; the session leaf's
owner runs the legal, serialized finalization sequence for each stream.

1. Set health to `NOT_SERVING` and close admission: new owners are refused (`state.draining` is
   already set by core).
2. Let owners finish normally until the settle point, `D - min(5 s, (D - now) / 4)`. Clients that
   finish their audio get their finals. This step lasts until the settle point even when no owner
   remains, so health checkers and load balancers observe `NOT_SERVING` and stop routing to the
   server before its listener closes.
3. At the settle point, raise the shutdown trigger with `D` for every remaining stream owner.
   An owner whose lease is open and that has selected neither a client finalization nor an
   abnormal cause selects provisional shutdown mode when it observes the trigger; an owner already
   finalizing for its client or already cleaning up keeps that path, as the session leaf's
   shutdown finalization states. An owner in shutdown mode wakes if it is blocked
   in a transport receive by cancelling that receive (never a lease call), settles any outstanding
   call under that call's own deadline with the wait clipped to `D`, feeds the audio frontend's drained tail, calls
   `flush`, `finish` and `release`, delivers the final only after `release`, and ends the stream:
   gRPC `OK` after the final, WebSocket close 1001. An owner whose sequence fails or does not finish
   by `D` aborts as the session leaf states (an open or broken lease takes `abort` then `release`;
   a lease already flushed takes `release` only) and ends the stream with gRPC `UNAVAILABLE`
   (`service_unavailable`) or WebSocket close 1012. File requests still running at the settle point
   are cancelled (errors leaf: `service_unavailable`, 503 before headers, stream termination after)
   and take their cancellation path; the wait for their settlement is bounded by
   `min(session_cleanup_timeout, time to D)`, and children still unsettled stay under their owner,
   with its permit and accounting, until settlement or `stop`.
4. Stop the gRPC server with the time remaining to `D` as the grace period.
5. Call `factory.drain(D)`, wait on its cleanup tasks until `D`, and log the faults it returns.

When `timeout` is not larger than `engine_reserve`, `drain` skips steps 2 and 3, explicitly
cancels every owner (each takes its abnormal path with the same flushed-lease rule), then performs
steps 4 and 5 with whatever time remains.

When `drain` is cancelled (core's deadline, or an early exit such as engine death), it cancels the
owners without awaiting their cleanup and re-raises; `stop` reaps what remains. `drain` honours
cancellation at every await.

### `stop(state)`

`stop` is terminal and prompt, whether the engine is live, stopping or dead, and it also runs in
startup rollback before any engine shutdown:

1. Mark the plugin torn down (owners then take their teardown path: they skip `abort` and
   `release` and call the lease's synchronous `teardown`).
2. Call `factory.teardown_all()`, which is terminal: after it, `open` and lease calls raise and
   owner cancellation cannot create factory work.
3. Cancel the owners, every file cleanup continuation with the children it holds (a core call or
   disconnect watcher that outlived its cleanup bound), the fault collector and the listener monitor.
4. Stop the gRPC server without grace.
5. Wait once for the cancelled tasks, with a short fixed bound (1 s), only so that cancellation is
   delivered; return whether or not they finished.

`stop` awaits no engine work and makes no direct engine call. Cancelling a factory-owned
`generate()` task can make the engine client await its own abort; `stop` does not wait for it
beyond the bound.

## Faults and escalation

Two supervision tasks run while the plugin serves.

- **Fault collector.** It calls `factory.drain(now)` (an immediate deadline) every second and records
  any cleanup faults the factory returns. The call is serialized with the shutdown `drain`'s own
  call so that two callers never take the same faults. The factory records orphaned-session cleanup
  faults (a retirement wait that raised `EngineDeadError` or an unexpected exception) and returns
  them only through `drain`, so without the collector a fault during normal service would stay
  invisible until shutdown.
- **Listener monitor.** It waits for the gRPC server's termination. An exit that is not caused by
  `drain`, `stop` or `start`'s own rollback is a fault.

Escalation has exactly three triggers: any call that fails with reason `engine_dead`, a cleanup
fault returned by the collector, and an unexpected listener exit. On any trigger the plugin records
the fault, which makes `ready` false, and calls `state.request_shutdown()` from the event loop,
which starts core's graceful shutdown; the orchestrator restarts the process. A cleanup bound that
expires on one session is not a trigger: the owner stops waiting, the session is handed to the
factory's own cleanup, and any resulting fault reaches the collector like any other orphaned-session
fault (session leaf).

Engine death on the file path is covered by core's own engine watchdog and error handling, not by
the frontend's escalation: a file request ends in core's serving, which core already reports as an
error and which starts core's own shutdown. The frontend's triggers concern the stream path and its
own listener.

## Configuration

Options come from `--additional-config` under the key `riva_frontend`, a JSON object. Omission of
the key or of an option means its default. Validation is strict and runs in `init_state`:

- an unknown key fails, listing every unknown key;
- an explicit `null` fails, naming the option; no option accepts `null` to mean "no limit" or "use
  the default" (`grpc_keepalive_seconds` and `max_session_duration` mean "off" by omission);
- every number is finite and positive, integers where the table says so;
- every failure names the plugin key `riva_frontend` and the offending option, and all failures are
  reported together;
- the relations in the table hold.

The set is deliberately the set of options with an operator-visible effect; protocol constants
(timeouts of the model plugin, the resampler's window, the settle reserve) are not options.

| Option | Default | Meaning and relation |
| --- | --- | --- |
| `grpc_bind` | `0.0.0.0:50051` | `host:port`; port 1 to 65535 |
| `grpc_receive_max_bytes` | 33,619,968 | gRPC message cap; equals `unary_max_encoded_audio_bytes + grpc_config_envelope_max_bytes` |
| `grpc_config_envelope_max_bytes` | 65,536 | allowance for the non-audio part of a request |
| `grpc_keepalive_seconds` | off | keepalive ping interval, applied as the server option `grpc.keepalive_time_ms` (the value times 1000); omit to disable |
| `unary_max_encoded_audio_bytes` | 33,554,432 | encoded audio cap for `Recognize` |
| `unary_max_decoded_duration_seconds` | 600 | decoded duration cap for `Recognize`; at most core's decode-duration limit |
| `max_riff_header_bytes` | 1,048,576 | RIFF header scan bound; at most `unary_max_encoded_audio_bytes` |
| `load_shed_max_sessions` | 64 | ingress cap; above `max_file_owners` |
| `max_file_owners` | derived | file-owner sub-cap; default `max_concurrent_files + 2`, 0 while files are disabled or not served; at least `max_concurrent_files`, below `load_shed_max_sessions` |
| `max_concurrent_files` | derived | file-gate size; default `min(4, max(1, floor(min(slots, active limit) / 4)))`, 0 while files are disabled or not served; while files are enabled, below both slots and active limit; when unset and `min(slots, active limit) < 2`, complete files are disabled with a warning; when set and failing the check, startup fails |
| `file_gate_wait_timeout` | 30 s | bound on a gate wait |
| `pre_submit_max_samples` | 65,536 | samples handed to `feed` and not yet accepted, per stream |
| `preconfiguration_timeout` | 30 s | first message plus `open` |
| `first_audio_timeout` | 10 s | below `never_fed_timeout` |
| `session_idle_timeout` | 45 s | below `parked_idle_timeout` |
| `lease_call_timeout` | 45 s | bound on one `feed` or `update_locale` call; below `parked_idle_timeout` |
| `session_finalization_timeout` | 30 s | below `draining_timeout` |
| `session_cleanup_timeout` | 30 s | abnormal-path `abort` and `release` bound outside drain |
| `max_session_duration` | off | cap on a streaming session from `open`; no limit unless set |
| `ws_receive_max_bytes` | 16,777,216 | one received WebSocket message; at least `ws_event_envelope_max_bytes` |
| `ws_event_envelope_max_bytes` | 8,388,608 | one JSON event before decode |
| `http_multipart_envelope_max_bytes` | 33,816,576 | whole file-route body; at least core's upload limit plus 262,144 bytes of framing |
| `http_content_type_max_bytes` | 4,096 | `Content-Type` value |
| `http_multipart_boundary_max_bytes` | 200 | boundary token |
| `http_multipart_max_parts` | 32 | parts per request |
| `http_multipart_max_header_bytes` | 8,192 | headers of one part |
| `http_text_field_max_bytes` | 4,096 | one text part |
| `http_body_read_timeout` | 120 s | total time to read the request body |
| `file_execution_timeout` | derived: `max(300 s, 2 * unary_max_decoded_duration_seconds)` | bound on core's execution of one file, `Recognize` and route |
| `response_send_timeout` | 60 s | bound on each send and close, on the route and both streaming surfaces (clipped to the RPC deadline and the drain deadline) |
| `session_factory` | unset | entry-point name that selects one factory when several accept |
| `resampler_identifier` | `scipy-poly-v1` | must equal the built-in identifier; a fixed fact |
| `deployment_image`, `pin`, `precision_policy` | `unspecified`, `unspecified`, `fp32-bringup-v1` | deployment-owned identifiers shown by the metadata route |

`plugin_shutdown_grace` is not an option: the shutdown budget is core's `--shutdown-timeout`.

Provenance values (`deployment_image`, `pin`, `precision_policy`, `resampler_identifier`) are
refused at startup if they carry leading or trailing whitespace or contain `://`, and
`deployment_image`, `precision_policy` and `resampler_identifier` are also refused when they are a
bare hexadecimal token of 8 to 64 characters. `pin` may be a git commit SHA, which is bare
hexadecimal by nature. What the rule excludes is a model-registry reference or a model-profile hash
presented as the deployment's identity; a commit identifier of the code being served is the
deployment's own fact. The two
deployment-owned values default to the literal `unspecified`, so a zero-configuration server never
advertises another deployment's image or pin.

## Deployment

The repository's deployment guide, `docs/deployment.md`, sets:

- `VLLM_PLUGINS=riva_frontend,nemotron_asr_streaming`: core loads no endpoint plugin when the
  variable is unset, and allowlists by entry-point name in every plugin group, so the model plugin
  must be named too;
- `--additional-config` carrying `riva_frontend` options and, when the defaults are changed,
  `stream_admission`;
- `--shutdown-timeout` above the engine's drain time (the engine's `draining_timeout` default is
  40 s; 60 s is the documented value);
- the model plugin's own serve flags (HLD Constraints). The reference deployment serves
  `supported_num_lookahead_tokens` `[6, 13]` with `num_lookahead_tokens` 13: 6 is the 560 ms arm
  that the Riva streaming surfaces open at, and 13 fixes the 1120 ms cadence at which complete
  files and core's own `/v1/realtime` run;
- `--max-model-len` large enough for the longest file `unary_max_decoded_duration_seconds` admits at
  the cadence `num_lookahead_tokens` fixes. The model's worst-case token budget is
  `1 + (C + 1) * Q + 1` with `r` that lookahead, `Q = (r + 1) * 10 + 1` and `C` the number of whole
  cadence chunks of `(r + 1) * 1280` samples. At the reference `r = 13` (`Q = 141`, 17,920 samples
  per chunk) a 600 s file needs at least 75,578; the model plugin's README serves 76,800. A
  deployment that runs files at 560 ms (`r = 6`, `Q = 71`, 8,960 samples per chunk) needs 76,114.
  The frontend does not compute the model's budget (it needs model-specific knowledge outside the
  structural seam); the formula and values here are for operators.

One API-server process is required and core enforces it from `owns_non_http_listener`. Data
parallel size above 1 is refused at startup, because the model requires parallel sizes of 1 and
slot count and active limit are per engine.

## Decisions & Alternatives

| Decision | Chosen | Alternatives Considered | Rationale |
| --- | --- | --- | --- |
| Vehicle | RFC 2 endpoint plugin, one class | A fork-only application plugin; patches to core | The lifecycle (`start`, `drain`, `stop`, `request_shutdown`) is what a long-lived listener needs and has an upstream path; patching core would merge Riva-shaped code into vLLM |
| vLLM version gate in packaging | None; `init_state` checks capabilities (session factory constructor raising on a missing engine call, the `realtime` task) | A lower bound on `vllm`; a version check at startup | No release carries RFC 1 and RFC 2 yet, so a bound would name a version that does not exist; a capability failure names the missing call, which is what the operator needs. A version gate is deferred until releases carry both |
| Engine deadlines | Read the `stream_admission` mapping from `additional_config` and apply the documented defaults when absent; fail on an unreadable value | Import core's `StreamAdmissionConfig`; read nothing and use fixed bounds | The mapping is the documented public setting; importing a scheduler-internal class couples the package to core internals; fixed bounds cannot be checked against what the engine enforces. A test pins the copied defaults against core |
| Frontend bounds below engine deadlines | Strictly below each, checked at startup, with defaults that pass | Equal bounds; engine deadlines as the only idle mechanism | A frontend timeout can name its cause and release its own resources first; the engine abort is then a backstop whose lease failure `expired` is projected as an idle timeout |
| Slot count and active limit | `num_gpu_blocks - 1` and `max_num_active_seqs or max_num_seqs`, read from `vllm_config` in `init_state` | A new `EngineClient` method; probing the scheduler | These values are already in the configuration the plugin is handed after the engine reports its cache; a new protocol method would be a core change outside RFC 1 and RFC 2 |
| Derived `max_concurrent_files` default | A quarter of the smaller of slots and active limit, between 1 and 4; files disabled when capacity is below 2 and the cap is not configured | A fixed default | A fixed number would fail small deployments and over-admit files on large ones; the HLD's startup check is kept as the guard |
| Cleanup bound expiry | Hand the session to the factory's cleanup; not an escalation trigger | Escalate to `request_shutdown()` | The factory already owns a session whose owner is gone and reports a fault only when the engine is dead or the cleanup itself fails; restarting the process because one release was slow turns a slow engine call into an outage |
| Escalation triggers | `engine_dead`, a collected cleanup fault, an unexpected listener exit | A failure latch; every error | A latch outlives its cause. Orderly restart is core's tool for a plugin that cannot continue |
| Fault collection | A task polling `factory.drain(now)` every second | A fault callback registered with the factory; collecting only at shutdown | The seam has no callback and adding one changes the model plugin's protocol; collecting only at shutdown leaves a dead engine unnoticed during service. The 1 s period is a constant, not an option |
| Dispatcher placement | Middleware from `attach_router`, inside core's middleware | A plugin WebSocket route; a namespaced path | Core attaches its `/v1/realtime` route first so a plugin route there is unreachable; a namespaced path breaks unchanged clients. As inner middleware, `--api-key` applies |
| Route table after attach | `init_state` verifies core's final table still routes the frontend's paths to its handlers | Rely on plugin order | Core attaches in entry-point discovery order and `VLLM_PLUGINS` only filters membership, so a later plugin can override the frontend and no list order prevents it |
| Route collisions | Raise in `attach_router` for a path the plugin adds as new | Last registration wins (core's behaviour) | Core overrides silently; for paths the frontend adds as new, two plugins serving one path is a deployment error that should stop the server |
| Configuration channel | `--additional-config` key `riva_frontend`, strict validator, errors listed together | A separate flag; a file path option | One flag surface already exists and core's own settings use it; strictness names the failing option. Core hashes `additional_config` into the compile cache key, which costs nothing under eager execution |
| `plugin_shutdown_grace` | Removed | Keep as the drain budget | Core passes `drain` the remaining `--shutdown-timeout`; two budgets would disagree |
| File backlog and streams | File owners count against a sub-cap `max_file_owners` inside the ingress cap | One shared cap | A backlog of waiting files would fill a shared cap and refuse streams with `busy`; the sub-cap leaves streams the remaining headroom |
| Sub-unit traffic | The engine is the authority; document the limit | A frontend check on units delivered; a seam change | The engine's clock resets on units sent, which the frontend cannot observe without a seam change; the engine's expiry is already a safe backstop |
| Startup order | Factory built last; any later failure runs `teardown_all`; admission opens last in `start`; `start` failure tears the factory down | Build it first | A factory owns engine-facing tasks, so a failed startup must not leave any; admitting before `start` completes would admit owners into a failing plugin |
| Live streams at drain | The frontend finalizes them at the settle point (tail, `flush`, final, then gRPC `OK` or WebSocket 1001); failures abort with `UNAVAILABLE` or 1012 | Cancel everything at the settle point; wait for clients | Clients that sent audio get a complete transcript; an aborted stream is told to retry elsewhere |
| Drain deadline | `deadline - max(5 s, 10% of timeout)` | The full core deadline | Core needs time to shut the engine down after plugins drain |
| `max_session_duration` | Off by default | 600 s | A default cap truncates legitimate long streams; operators set it when they want one |
| Memory bound | `max_file_owners` times the body bound plus per-stream bounds, default file sub-cap modest | Unbounded waiters | The bound is stated so operators can size memory; the defaults keep it near half a gigabyte |
| Data parallel | Refused when above 1 | Support per-rank ports | The model requires parallel sizes of 1; slots and active limit are per engine |
| Tiny deployments | Files disabled with a warning when no cap is configured and capacity is below 2; the route stays registered and refuses with 501; an explicit failing cap fails startup | Always fail; remove the route | A one-slot server is valid for streams only; an operator who configured files expects them |
| File-path engine death | Core's watchdog | Frontend escalation | The file request lives in core's serving, which already reports it |
| `pin` provenance | May be a commit SHA | Refuse all bare hex | The rule exists to keep registry identities out; a commit SHA is the deployment's own identifier |
| Interpreter range | The range vLLM declares, with a fail-closed gate | A range chosen by the frontend | The plugin runs in vLLM's interpreter; a narrower range excludes supported hosts |
| Served identity | `model_name()` once, primary identity only on Riva and Realtime | Accept every alias | Riva clients read one name from the listing; the registry is the sole authority for it |

## Open Questions & Future Decisions

### Deferred
1. A vLLM version gate in `init_state`, once releases carry RFC 1 and RFC 2.
2. If core adds callbacks or a client protocol for stream-admission settings, slot count and
   active limit, `init_state` reads them there instead of from configuration.

## References

- RFC 2 endpoint plugins: `docs/design/endpoint_plugins.md` and
  `vllm/plugins/endpoint_plugins/interface.py` on `pdtgct/vllm` branch `endpoint-plugin-lifecycle`.
- RFC 1 persistent state and stream admission: `docs/design/persistent_state.md`,
  `vllm/v1/core/stream_admission.py`, `vllm/v1/core/state_slot_manager.py` on `pdtgct/vllm` branch
  `persistent-state-primitive-v2`.
- Core middleware order: `vllm/entrypoints/launchers/app.py`,
  `vllm/entrypoints/serve/middleware/register.py`, `.../authenticate.py`.
- Model plugin: `pdtgct/vllm-nemotron-asr-streaming-plugin` (`docs/intent/session/`,
  `docs/intent/registration/`).
- High-level design: `docs/high-level-design.md`.
