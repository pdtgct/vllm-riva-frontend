---
parent: high-level-design
prefix: ING
facets: [EP, VEH, ADM]
---

# Vehicle

## Context and Design Philosophy

The vehicle is the part of the frontend that vLLM core sees: one `EndpointPlugin`
(`RivaFrontendPlugin`) and the lifecycle methods core calls on it. It owns the package and its
declaration, its HTTP namespace and the authentication applied to it, the order of startup checks,
the registry of owner tasks, the ingress cap, the relation between the frontend's time bounds and
the engine's stream deadlines, shutdown and its settlement, fault escalation, and the frontend's
configuration. Everything a surface does with a stream, a file or a
wire format belongs to another leaf; this leaf decides when those parts exist and when they stop.

Principles:

- **Fail at startup, name the option.** A server that cannot serve correctly does not start; the
  error names the option or check that failed, and a configuration error lists every failing
  option, not only the first.
- **Every task has an owner until it ends.** The frontend starts no task outside its registry,
  apart from the fault collector, the listener monitor and refusal sends. A refusal send runs as a
  task that its own request handler awaits inline under `response_send_timeout`, so the handler
  owns it; it enters the registry only when the handler abandons it (Owner registry, ING-EP-095).
  Engine-facing tasks belong to the model plugin's factory. `drain`, `stop` and a failed startup
  await every task they end before they return or raise.
- **Bounded everywhere.** Every read, wait and lease call has a finite bound, and during shutdown
  every bound on graceful work is the time left to one deadline. The settlement that follows it is
  cooperative and has no fixed time limit.
- **Its own namespace.** Every HTTP and WebSocket route of the frontend is under
  `/plugins/riva/v1/`; core's routes are neither replaced, shadowed nor wrapped.
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

The gRPC listener needs exactly one API server process, since every process would bind its port;
`init_state` enforces that (startup checks below).

Normal runtime dependencies are the Riva stubs (`nvidia-riva-client`, pinned), `grpcio` with its
health and reflection packages, `numpy`, `scipy` (the resampler) and `python-multipart` (the
file-route parser). The package has no resolver-owned dependency on vLLM, which the host environment
supplies, and none on the model plugin. It imports vLLM only inside the lifecycle methods, the
authentication middleware and the complete-file path (`Recognize` and the frontend's file route),
where the contract it uses is core's. It adds no vLLM version bound: the capability checks in
`init_state` decide whether a given build is usable (Decisions).

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

- core's `/v1/realtime` and, for a transcription model, `POST /v1/audio/transcriptions`, unchanged
  and outside the frontend's ingress cap, file gate and guard;
- the frontend's routes under `/plugins/riva/v1/` (the Realtime WebSocket route, the Realtime
  bootstrap, the file route and the operational routes) and its path-scoped authentication
  middleware, on the HTTP listener;
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
| `attach_router(app)` | `build_app`, after core's routers and before core's middleware | add the namespaced routes (the WebSocket route, the bootstrap, the operational routes, and the file route only when core's `POST /v1/audio/transcriptions` is present) and the path-scoped authentication middleware; fail on a path collision |
| `init_state(engine_client, state, args)` | `init_app_state`, engine running, after core's middleware is registered | the startup checks below, including the API-server-process and authentication checks, then build the owner registry, ingress cap, file gate, servicer and error catalog |
| `start(state)` | lifespan startup | bind and start the gRPC server, start the fault collector and listener monitor, then open admission; roll back on failure or cancellation |
| `ready(state)` | core's `/ready` and the operational readiness route | attribute reads only, including `state.draining` |
| `drain(state, timeout)` | graceful shutdown, after core sets `state.draining`, concurrently with other plugins | the graceful sequence below, then settlement |
| `stop(state)` | after HTTP stops, for a plugin whose `start` completed | terminal; awaits every task it cancels |

Core sets `state.draining` before it calls any `drain` hook and calls none when no shutdown budget
remains. Admission and `ready` therefore read `state.draining` as well as the plugin's own state,
so new owners are refused from the moment core sets it, with or without a `drain` call.

### `attach_router`

Every route the frontend adds is under `/plugins/riva/v1/`, the plugin-owned prefix that vLLM's
security guide recommends for endpoint plugins:

| Route | Leaf |
| --- | --- |
| WebSocket `/plugins/riva/v1/realtime` | streaming |
| `POST /plugins/riva/v1/realtime/transcription_sessions` (Realtime bootstrap) | streaming |
| `GET /plugins/riva/v1/health/ready`, `health/live`, `version`, `metadata` | operations |
| `POST /plugins/riva/v1/audio/transcriptions`, only when core's `POST /v1/audio/transcriptions` is registered | files |

All are new routes. The frontend replaces, shadows and wraps none of core's routes and does not
touch core's `/v1/realtime`, `POST /v1/audio/transcriptions` or `POST /v1/audio/translations`.

**Route checks.** The vehicle owns three checks that keep the frontend's routes and core's apart;
the operations leaf owns the route set they apply to.

- *Collision, in `attach_router`.* Before adding a route, the frontend checks `app.routes` for an
  existing route that would serve the same request: an HTTP route with the same path and method, a
  WebSocket route with the same path, or a mount whose path is a prefix of the frontend's path. It
  raises on a match, which stops startup, rather than rely on registration order. Core replaces a
  same-path, same-method operation silently, and an earlier WebSocket route or mount wins over a
  later route, so registration order would otherwise decide which handler serves.
- *Override, in `init_state`.* The check in `attach_router` cannot see routes added after it, and
  core attaches plugins in entry-point discovery order (`VLLM_PLUGINS` filters membership only), so
  step 8 of the startup checks verifies the final table.
- *Core's table unchanged, in CPU tests.* With the plugin loaded, core's route table is core's table
  without the plugin plus exactly the frontend's routes: every core route keeps its path, methods,
  endpoint and relative order, and every added route is under `/plugins/riva/v1/`. The same tests
  verify that the frontend's route set collides with no route of core's table.

**Path-scoped authentication.** Core's authentication middleware checks `--api-key` only on its own
prefixes (`/v1`, `/v2`, `/inference`, `/cohere`), so it does not cover the namespace. `attach_router`
therefore adds one pure-ASGI middleware with `app.add_middleware`. `build_app` runs plugin hooks
before it registers core's middleware, and Starlette puts later middleware outermost, so this
middleware runs inside core's middleware (core's WebSocket metrics, authentication and CORS
middleware wrap it). For each scope it:

- passes a scope through unchanged when its type is neither `http` nor `websocket`, when its method
  is `OPTIONS` (core's exemption), or when its path, after removing `root_path` as core does, does
  not start with `/plugins/riva/` (so `/plugins/riva`, `/plugins/rivax` and every core path are
  untouched; the frontend has no route outside `/plugins/riva/v1/`);
- passes it through when authentication is off (the authentication check in `init_state`, below);
- otherwise checks the request's headers with core's own token check: the `verify_token` method of
  an instance of core's `AuthenticationMiddleware` that `init_state` builds with the tokens it
  resolves (below). An HTTP request that fails the check gets the response core's middleware sends,
  401 with `{"error": "Unauthorized"}`. A request that passes goes on to routing, so an unknown path
  under the namespace is authenticated before routing answers it (the errors leaf lists the
  responses).

**WebSocket refusal.** A WebSocket handshake that fails the check is refused before accept; that is
the normative behaviour. When the scope's `extensions` include the ASGI WebSocket denial-response
extension (`websocket.http.response`), the refusal carries core's 401 status and body, which is
how core's own middleware answers an unauthenticated handshake on `/v1/realtime`. When the
extension is absent, the middleware sends `websocket.close` before accept, which the ASGI server
answers with HTTP 403 and no body. Either way the connection is never accepted, so the refusal
fails closed whatever the server's capabilities.

The middleware is part of the HTTP surface only; the gRPC listener is outside `--api-key`. Until
`init_state` has resolved the tokens the middleware fails closed under the namespace. An HTTP
request gets 503 with the body `{"error": "Service Unavailable"}`. A WebSocket handshake is refused
before accept, by the rule of the WebSocket refusal above: a 503 response with the same body through
the denial-response extension when the scope offers it, otherwise `websocket.close` before accept
(HTTP 403). Core serves no request before `init_state` completes, so this state is not reachable in
a completed startup. The class, its `tokens` argument and `verify_token` are core internals outside
RFC 2's supported surface; tests pin them against core (the token check, the `tokens` keyword of the
middleware's registration, and the 401 body).

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
   answers `UNIMPLEMENTED` and the frontend has no file route (core attached none, so
   `attach_router` added none). If `min(slots, active limit)` is below 2 and
   `max_concurrent_files` is not configured, complete files are disabled with a startup warning:
   one files-enabled flag becomes false, `Recognize` answers `UNIMPLEMENTED`, no `offline` entry is
   listed, and the frontend's route (already registered by `attach_router`) refuses with
   `unsupported_capability`, HTTP 501. An explicitly configured `max_concurrent_files` that fails
   the sizing check fails startup.
6. **API server processes**: `args.api_server_count` must count one process. Core leaves it `None`
   when it runs the single API server in this process, and `None` (or an absent attribute) counts as
   one; 1 is one; any larger value fails startup, naming the option, because every process would
   bind the gRPC port.
7. **Authentication**: two signals decide whether the namespace is authenticated and with which
   tokens. *Key configured* applies core's own selection rule (`register.py`): the candidate list
   is `args.api_key` when that list is non-empty, and otherwise the single value of
   `VLLM_API_KEY`; empty strings are then discarded, and a key is configured when the result is
   non-empty. Selection precedes filtering, so `--api-key ""` with a non-empty `VLLM_API_KEY`
   configures no key, as in core. *Registration* is found in the registered middleware list of the
   `app` kept from `attach_router` (core registers its middleware after `attach_router` and before
   `init_state`): the entries whose class is core's `AuthenticationMiddleware`. A registration is
   readable when it has a `tokens` keyword whose value is a non-empty list of non-empty strings.

   | Key configured | Core's `AuthenticationMiddleware` registrations | Result |
   | --- | --- | --- |
   | no | none | authentication off: the middleware passes every request through, as core does without a key |
   | yes | exactly one, readable | authentication on, verifying with that registration's tokens |
   | yes | none | startup fails |
   | no | one or more | startup fails |
   | either | more than one, or one that is not readable | startup fails |

   Each failure names the signals it compared. The resolved tokens are never logged.
8. **Build** (everything except the factory): registry, ingress cap, file sub-cap, gate, error
   catalog, and the verifier the authentication middleware calls. Then the override check (Route
   checks): the `app` kept from `attach_router` is searched, and `init_state` fails startup unless,
   in core's final table (after every plugin attached), the first route that matches each path the
   frontend added (and its method, for an HTTP route) is the frontend's own.
9. **Session factory, last**: discovery, the single-accept rule and the `560ms` cadence check
   (session leaf), then the servicer, which needs the factory. The vehicle owns every factory a
   constructor returned: a factory that discovery discards (not the named one, or one of several
   accepting) is settled when it is discarded, the selected one in `drain`, `stop` or a
   rolled-back `start`, and any failure after a factory has been constructed settles every factory
   constructed so far (Factory settlement) before the error propagates, so a failed `init_state`
   leaves no engine-facing task.

`init_state` binds no listener and starts no task, and the factory constructor starts no task or
session (the model plugin's constructor only records configuration and creates empty tables). A
factory that exists when `init_state` returns is therefore inert until the first `open`. Core
provides no hook between a completed `init_state` and `start` (it calls `stop` only for a plugin
whose `start` completed), and this invariant is what makes a startup failure in that interval,
for instance another plugin's `init_state` raising, leave nothing of this plugin's running.

If `--shutdown-timeout` is 0, `init_state` logs a warning that no `drain` will run and that
shutdown is `stop` alone. With that budget core stops the engine before it calls `stop`
(`stop(state)`), so `stop`'s settlement ends only when the lease calls and the retirement wait
end on a stopped engine; if one of them does not, only a repeated signal or the orchestrator's
kill grace (Deployment) ends the process.

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
build. That test needs RFC 1's stream-admission settings and runs only in the RFC 1 + RFC 2 build;
the frontend's other CPU tests need nothing from RFC 1 and run against an RFC 2 build without it.

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
load count, until it finishes. `drain`'s settlement and `stop` cancel the owner and await the
continuation through that cancellation: the continuation still completes its abort retries, the
final abort, its children's settlement and the release of its accounting (files leaf), so
cancellation never cuts it short.

Admission refusals and query refusals register no owner. Each refusal send (an admission refusal's
response or status, or a WebSocket query refusal's close before the handshake is accepted) is
awaited inline by its own request handler under `response_send_timeout` (session leaf, Sends). An
engine-capacity refusal is not among them: it is sent by the owner of the stream that `open`
refused, under that owner's record. If a refusal send is still unsettled one turn of the event
loop after its bound passed and the registry is still open, the handler cancels it and the send
moves to the registry as a send without an owner, from that moment of abandonment: it takes no
ingress-cap count, and the settlement and `stop` await it like an owner's send. Core keeps serving
HTTP while plugins drain (it stops the HTTP server only after `drain` and the engine shutdown), so
refusals can arrive during settlement; settlement therefore closes the registry to sends without
an owner before its final wait on unsettled sends (Settlement, step 3). A refusal send that passes
its bound after that is cancelled and awaited by its own handler, so no send becomes plugin work
that the final wait does not see. Two properties make this hold. The close of the registry, a
handler's check of whether the registry is open and the registration of its abandoned send run on
the event loop with no suspension between the check and the registration, so a send is either
registered before the close or handled by its own handler after it. A handler awaiting its
cancelled send after the close keeps awaiting it if the handler is itself cancelled, and re-raises
that cancellation only once the send has ended, so the send is never abandoned.

The frontend starts no task outside the registry, apart from three exceptions. Two are long-lived
supervision tasks that `start` creates: the fault collector, which settlement ends before the
factory settlement in `drain` and in `stop`, and the listener monitor, which `drain` leaves running
and `stop` ends and awaits. The third is a refusal send, which its own request handler awaits
inline and owns until abandonment moves it into the registry (above). The model plugin's factory
owns every engine-facing task of its sessions.

**Ingress cap.** One finite positive cap (`load_shed_max_sessions`) counts active owners across
`StreamingRecognize`, `Recognize`, the Realtime WebSocket route and the frontend's file route
together. It excludes core's `/v1/realtime` and `POST /v1/audio/transcriptions`,
`GetRivaSpeechRecognitionConfig`, the bootstrap POST, the operational routes, a WebSocket connection
refused for its query (streaming leaf), and every request the authentication middleware refuses.
Before a handler reads its first message or body (for `Recognize`, whose request is already in
memory under the framework cap, at servicer entry before any decode), parses multipart or calls
`open`, it atomically checks that the plugin is serving (`state.draining` unset, admission open, not
torn down) and that the count is below the cap, and registers its owner when both hold. A draining
plugin answers `service_unavailable`; a full cap answers retryable `busy` with
`authority: load_shed`. Below the cap the counter claims no residency, reserves no engine capacity
and queues nothing, and sessions already open continue unchanged. This cap bounds frontend memory and work; the
file gate (files leaf) separately bounds files in inference, and the engine's admission (below)
bounds streams.

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
the listener is locally reachable before `start` returns. Then `start` creates the fault collector
and the listener monitor, and opens admission as its last step; health becomes `SERVING` when
admission opens, so no owner is admitted before `start` completes. If `state.draining` is already
set at that step, `start` leaves health `NOT_SERVING`; admission refuses in any case, because it
reads the flag.

**Startup rollback.** Core calls `stop` only for a plugin whose `start` completed. If `start` fails
or is cancelled at any point, it rolls back what exists: it cancels the fault collector and the
listener monitor if it created them (so the rollback's listener exit is not reported as a fault),
closes the listener without grace if it was bound (`stop(0)`), awaits the tasks it cancelled, runs
the factory settlement (Settlement) and re-raises the failure or the cancellation. Admission opens
last, so no owner has run, no session exists and the factory holds no fault during a rollback;
cancelling the fault collector there loses nothing. A cancellation that arrives during the rollback
does not interrupt it; it is re-raised afterwards. Startup rollback after a later plugin fails is
core's `stop` call on this plugin, so `stop` is correct for a plugin that started.

`ready(state)` reads attributes only and is true while the listener is serving, the plugin is not
draining (neither `state.draining` nor the plugin's own drain flag is set) or torn down, and no
cleanup fault is recorded. Core's `/ready` aggregates it with engine health and `state.draining`
and is the process's readiness for orchestrators; core's `/health` reports engine health only and
does not read it. The operational readiness route reports it alone (operations leaf).

## Shutdown

### `drain(state, timeout)`

`drain` computes the core deadline, `now + timeout`, and a target for graceful work,
`D = deadline - engine_reserve`, where `engine_reserve = max(5 s, 10% of timeout)`. The target is
meant to leave core time to shut the engine down; core reserves no engine allowance and passes the
engine only what remains, and settlement (below) takes precedence over the target. Every wait of
graceful work inside `drain`, including each lease call's bound and the factory wait, is the time
remaining to `D`. The vehicle owns only the shutdown trigger, `D` and the settlement; the session
leaf's owner runs the legal, serialized finalization and cleanup sequence for each stream.

1. Set health to `NOT_SERVING` and close admission: new owners are refused (`state.draining` is
   already set by core).
2. Let owners finish normally until the settle point, `D - min(5 s, (D - now) / 4)`. Clients that
   finish their audio get their finals. This step lasts until the settle point even when no owner
   remains, so health checkers and load balancers observe `NOT_SERVING` and stop routing to the
   server before its listener closes.
3. At the settle point, raise the shutdown trigger with `D` for every remaining stream owner. An
   owner whose lease is open and that has selected neither a client finalization nor an abnormal
   cause selects provisional shutdown mode when it observes the trigger; an owner already finalizing
   for its client or already cleaning up keeps that path, as the session leaf's shutdown
   finalization states. An owner in shutdown mode wakes if it is blocked in a transport receive by
   cancelling that receive (never a lease call), settles any outstanding call under that call's own
   deadline with the wait clipped to `D`, feeds the audio frontend's drained tail, calls `flush`,
   `finish` and `release`, delivers the final only after `release`, and ends the stream: gRPC `OK`
   after the final, WebSocket close 1001. An owner whose sequence fails or does not finish by `D`
   aborts as the session leaf states (an open or broken lease takes `abort` then `release`; a lease
   already flushed takes `release` only) and ends the stream with gRPC `UNAVAILABLE`
   (`service_unavailable`) or WebSocket close 1012. File requests still running at the settle point
   are cancelled (errors leaf: `service_unavailable`, 503 before headers, stream termination after)
   and take their cancellation path; the wait for their settlement is bounded by
   `min(session_cleanup_timeout, time to D)`, and children still unsettled stay under their cleanup
   continuation, with its permit and accounting, which the settlement awaits.
4. Stop the gRPC server with the time remaining to `D` as the grace period.
5. End the fault collector: signal it to stop and await its end (a collection call in progress
   finishes, and its faults are recorded). Then call `factory.drain(D)`, wait on its cleanup tasks
   until `D`, and log the faults it returns.

When `timeout` is not larger than `engine_reserve`, `drain` skips steps 2 to 5 and goes straight to
the settlement, which aborts every owner (each takes its abnormal path with the same flushed-lease
rule).

### Settlement

RFC 2 requires a plugin to settle or abort, and await, all work it owns before `drain` returns,
raises or is cancelled. `drain` therefore settles on every exit: after step 5, when `D` or core's
deadline passes, when a step raises, and when `drain` is cancelled (core cancels it at its
deadline). Settlement runs these steps in order:

1. **Owner settlement.** It aborts every owner still registered by cancelling its task. A stream
   owner takes the session leaf's abnormal path with the shutdown abort row, unless it already
   selected a cause: an open or failed lease takes `abort` then `release`, a flushed lease `release`
   only. These cleanup calls run under the frontend's own cleanup bound, `session_cleanup_timeout`,
   not under the time to `D`, which has usually passed. When that bound expires, the owner takes the
   same hand-over as before the deadline (session leaf, Cleanup order): it cancels an outstanding
   `release`, or cancels an outstanding `abort`, awaits that call's settlement, then begins
   `release`. The owner runs `release` as a task, lets it reach its first suspension, never cancels
   it before its first step, and awaits the cancelled call before the owner or its continuation
   counts as ended; the factory does not suspend in `release` before its hand-over coverage is in
   place (session leaf, Factory obligation 6). A cancelled `release` leaves the session with one
   tracked cleanup task of the factory (`SESS-REL-003`, `SESS-OWN-003`), registered when the
   cancellation is delivered inside the call; awaiting the cancelled call is what places that task
   in the factory before the factory settlement below. An owner whose `open` is pending cancels it,
   which hands a started session to one factory cleanup task (`SESS-OPEN-006`). If the cancelled
   `open`'s generation task never started, the factory drops the session and starts no cleanup task:
   that task never runs its body, makes no engine call and ends at the next step of the event loop,
   so it is untracked and harmless. A file owner takes its cancellation path, and its cleanup
   continuation runs the abort retries and the final abort (files leaf).
2. It stops the gRPC server without grace if it is still serving.
3. It closes the registry to sends without an owner (Owner registry), then awaits every owner
   task, every cleanup continuation (stream and file) with the children it holds, and every
   unsettled send in the registry, with or without an owner, until each has ended. A refusal send
   that passes its bound after the registry closed is cancelled and awaited by its own handler,
   so the registry gains no entry after this wait begins.
4. It ends the fault collector as graceful step 5 does, if it is still running.
5. It runs the factory settlement (below).

The order is fixed: owners settle before the factory closes, because a closed factory refuses
every lease call other than `teardown` (`SESS-OWN-006`), so an owner still cleaning up would lose
its `abort` and `release`; and the factory closes last, after every task it owns has ended, so the
settlement ends every engine-facing task without cancelling any of them while it awaits the engine
client's abort.

**Cancellation ownership.** A session's generation task is cancelled by the party that ends the
session: `abort` (`SESS-ABORT-001`) or a cancelled pending `open` (`SESS-OPEN-006`). The factory's
`open`, `abort`, `release` and its own cleanup and holding tasks deliver no cancellation to a
generation task whose cancellation handler has begun (session leaf, Factory obligation 5);
`teardown_all` and the lease's `teardown()` cancel unconditionally by contract and are excepted. The
system-level property, which the settlement order provides, is that a generation task receives no
cancellation after its cancellation handler has begun, that is, none while it awaits the engine
client's abort of its request: a cancelled `generate()` task awaits that abort, and a cancellation
delivered then would interrupt it and could leave the request un-aborted in the engine. A cancelled
`open` requests the task's cancellation twice, once when its wait is cancelled and again when it
hands the session over, with no suspension between the two requests; the task's handler has not
begun when the second request is made, so both requests are delivered as one cancellation and the
property holds. The vehicle cancels no factory task in any settlement (`drain`, `stop` or a
rollback), except in the last-resort case of factory settlement step 3. The factory's own tasks
never cancel a generation task: a cleanup task awaits the session's retirement and then the end of
its generation task, and the task that holds a session after a failed retirement wait awaits the end
of the generation task.

**Factory settlement.** `drain`'s settlement, `stop`, a rolled-back `start` and a failed
`init_state` end the factory the same way, after the owners, continuations and unsettled sends
have ended and the fault collector has ended:

1. Call `factory.drain(math.inf)`. It awaits every factory cleanup task without cancelling it
   (`SESS-OWN-004`; the infinite deadline is Factory obligation 1 of the session leaf): the cleanup
   tasks of sessions handed over by a cancelled `release` or a cancelled `open`, and the tasks that
   hold a session until its generation task ends. Each of these ends only after its session's
   generation task has ended (Factory obligation 2), so every generation task the owners handed
   over finishes its cancellation, including the engine client's abort. The owners have ended, and
   each awaited its cancelled `release` before it counted as ended, so every hand-over has
   registered its cleanup task and no lease call can register a further one during the wait.
2. Call `factory.teardown_all()`. It marks the factory closed, cancels every task the factory owns
   and returns them (`SESS-OWN-005`): the generation task of every session still in the factory's
   table and every factory cleanup task. By step 1 every returned task has already ended, so its
   cancellation has no effect. Once closed, the factory refuses `open` and every lease call other
   than `teardown` (`SESS-OWN-006`) and starts no task (Factory obligation 7).
3. Await every returned task until each has ended. The wait does not stop at a task that ends with
   an exception, so no remaining task is skipped. A returned task that was not done when
   `teardown_all` returned violates the invariant of step 1 (for example, a session whose owner
   failed before handing it over): `teardown_all` cancelled it as a last resort, the wait awaits
   it, and the settlement records a last-resort fault keyed by that task object, the task the
   vehicle saw cancelled.
4. Call `factory.drain(now)` once more. It collects the faults of tasks that ended during steps 2
   and 3.
5. Report faults: the faults `factory.drain` returned in steps 1 and 4, as returned, and the
   last-resort faults of step 3. The factory is the sole reporter of its tasks' faults and returns
   each exactly once (Factory obligations 3 and 4): a cleanup or holding task leaves the factory's
   table when it ends and drops its session, so only the factory still knows of its faults and of
   its generation task's. The vehicle therefore reads no exception from the tasks `teardown_all`
   returns, and the only faults it originates are last-resort cancellations. A later factory
   settlement on the same factory reports nothing twice by construction: `drain` has cleared what
   it returned, and a task that has already ended is never found still running again. In `drain`
   and `stop` the faults are logged and not escalated, because shutdown is already under way; in a
   failed startup they are logged and the startup error propagates.

These steps rely on the factory obligations the session leaf lists (Factory obligations 1 to 7),
which the frontend imposes on every factory and checks in its seam conformance suite. The model
plugin's session specification already states that `drain` awaits cleanup tasks and returns the
recorded cleanup faults once (`SESS-OWN-004`), that `teardown_all` closes the factory and cancels
and returns every task it owns (`SESS-OWN-005`), that a closed factory refuses `open` and lease
calls (`SESS-OWN-006`), that a cancelled `release` hands the session to the factory's cleanup
(`SESS-REL-003`), and that a `release` after a cancelled `abort` keeps a tracked task until the
generation task ends (`SESS-ABORT-002`); the suite tests these as well, including that
`teardown_all` returns the generation task of a session whose `abort` was cancelled. Obligations 3
and 4 need a change to the model plugin's code; the rest are true of its code and owe statements
in its specification (Open Questions, Dependencies).

This order settles the case where an owner's `abort` outlives its cleanup bound. The cancelled
`abort` leaves the lease `aborted` with its generation task cancelled and possibly still running
(the engine client awaiting its own abort). The owner's `release` hand-over gives the session to a
factory cleanup task, which waits for the retirement and then for that generation task; step 1
awaits the cleanup task, and the generation task completes its abort before `teardown_all` runs.

The awaits have no fixed time limit; they are cooperative, as core's own shutdown is, and core's
exit on a repeated signal bounds a hang. A further cancellation of `drain` during settlement does
not interrupt it: settlement continues, and `drain` re-raises the cancellation after it completes,
however many cancellations arrive. When a step raised and `drain` was also cancelled, the
cancellation is re-raised; otherwise the step's exception is, after settlement. When settlement
ends, no frontend task and no factory task remains that can submit engine work while core shuts the
engine down.

**Zero budget.** If core's remaining budget is zero, core sets `state.draining` and calls no
`drain`; zero is core's default `--shutdown-timeout`. Admission and `ready`, and therefore core's
`/ready`, observe the flag at once (Lifecycle methods). gRPC health stays `SERVING` until `stop`
stops the listener, because only `drain` sets `NOT_SERVING`; a gRPC health checker sees the
listener close rather than a `NOT_SERVING` status, and every new owner is refused meanwhile. `stop`
settles everything that remains.

### `stop(state)`

`stop` is terminal, and core calls it only for a plugin whose `start` completed (including when a
later plugin's startup fails). Core does not guarantee that the engine has stopped when `stop`
runs: its contract says the engine may already be stopped, and core shuts the engine down in its
shutdown task, which runs only after a shutdown request. When the server exits without a
completed shutdown task, core cancels that task before stopping plugins, so `stop` can find live
owners on a live engine. `stop` therefore settles in `drain`'s order:

1. Mark the plugin torn down: admission refuses and `ready` is false.
2. Run the settlement (Settlement, steps 1 to 5) without a graceful phase: abort every owner by
   cancelling its task, stop the gRPC server without grace, await every owner, cleanup
   continuation and unsettled send (every send bound counts as passed once the plugin is torn
   down), end the fault collector, and run the factory settlement, which reports only what the
   factory's `drain` returns and any task it finds still running, so nothing a `drain` settlement
   reported is reported again.
3. Cancel and await the listener monitor.

A cancellation of `stop` does not interrupt these steps; it is re-raised afterwards. After a
completed `drain` settlement, `stop` finds no owner and no running factory task, and its factory
settlement ends at once.

With a zero shutdown budget (core's default `--shutdown-timeout`), core calls no `drain` and stops
the engine before it calls `stop`, so `stop` settles on a stopped engine the owners `drain` never
saw. Their `abort` and `release` end under `session_cleanup_timeout` like any other cleanup call,
but the factory settlement has no bound: it ends only when every handed-over retirement wait and
every generation task's engine abort end on the stopped engine. The design therefore requires that
the model plugin's `abort` and `release`, the retirement wait and the abort of a cancelled
`generate()` task end on an engine that core has stopped, and a required test establishes it against
the real factory on a stopped engine. That test runs in the model plugin repository's conformance
suite on the RFC 1 + RFC 2 build with a real engine client (its GPU lane); the frontend's RFC 2-only
CPU lane covers the same settlement order with the fake factory. A call that never ends there holds
`stop` until a repeated signal or the orchestrator's kill grace ends the process (Deployment). The
same holds on a live engine for a `generate()` task or engine-client call that never ends after its
cancellation, or a retirement wait that never returns.

`stop` starts no engine work of its own: the lease calls of its owners are the cleanup the
session leaf prescribes, and a cancelled `generate()` task completes the engine client's abort of
its request before the factory closes. `stop` awaits every task with no fixed time limit, as RFC 2
asks of `stop`.

## Faults and escalation

Two supervision tasks run while the plugin serves.

- **Fault collector.** It calls `factory.drain(now)` (an immediate deadline) every second and
  records the retained faults the factory returns. The call is serialized with the shutdown
  `drain`'s own call so that two callers never take the same faults. The factory retains exactly
  the faults of work no caller can still see (session leaf, Factory obligation 3), each an
  exception other than `CancelledError`: a cleanup task's retirement wait that reports a dead
  engine or is unexpected; a generation task's exception that no `open` or lease call surfaced to
  its caller, because the task ended after its session was handed over to a cleanup or holding
  task or because its cancellation handler raised while an uncancelled `abort` awaited it; and an
  exception of a cleanup or holding task itself. A refusal or failure surfaced to its caller is not
  retained and therefore causes no collector-driven escalation: a capacity refusal reaches the
  owner with `session_error` `"refused"`, and a generation failure as a lease `reason` or with
  `session_error` `"failed"`; `engine_dead` remains a direct escalation trigger. A cleanup or
  holding task that `teardown_all` cancels as a last resort is reported only by the vehicle, as a
  last-resort cancellation fault (Settlement, step 3), never through `drain`. The factory returns
  retained faults only through `drain`, so without the collector a fault during normal service
  would stay invisible until shutdown. `drain` and `stop` end it by a signal (graceful step 5 or
  the settlement) and await its end; it is never cancelled there, so a call in progress is not
  lost. A rolled-back `start` cancels and awaits it, which loses nothing because no fault can exist
  before admission opens (Startup rollback).
- **Listener monitor.** It waits for the gRPC server's termination. An exit that is not caused by
  `drain` (including its settlement), `stop` or `start`'s own rollback is a fault. It runs until
  `stop` or a rolled-back `start` cancels and awaits it.

Escalation has exactly three triggers: any call that fails with reason `engine_dead`, a retained
fault returned by the collector during service (escalated as a cleanup fault), and an unexpected
listener exit. Ordinary refusals and request failures are never triggers: they are surfaced to the
owner that made the call and are not retained, and a test checks that an ordinary capacity refusal
and an ordinary generation failure produce no collected fault and no escalation. On any trigger the
plugin records the fault, which makes `ready` false, and calls `state.request_shutdown()` from the
event loop, which starts core's graceful shutdown; the orchestrator restarts the process. A trigger
observed after shutdown has begun records the fault in the same way; core ignores the repeated
shutdown request. A cleanup bound that expires on one session is not a trigger: the owner stops
waiting, the session is handed to the factory's own cleanup, and any resulting retained fault
reaches the collector like any other (session leaf).

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
| `session_cleanup_timeout` | 30 s | abnormal-path `abort` and `release` bound outside drain and in the settlement after the drain deadline |
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
- `--max-model-len` large enough for the longest file either frontend file surface admits, at the
  cadence `num_lookahead_tokens` fixes. The frontend's file route admits a file up to core's
  decode-duration limit (`VLLM_MAX_AUDIO_DECODE_DURATION_S`, 600 s by default), and `Recognize` up
  to `unary_max_decoded_duration_seconds`, which `init_state` keeps at or below that limit; so
  core's limit is the governing duration, and the guide states it whenever it changes it. The
  model applies no input-length limit of its own: a file that reaches `--max-model-len` ends with
  a `length` finish, and on a core without the pending change that turns such an ending into a
  failure, core returns the partial transcript as success (HTTP 200 on the route, a normal
  `Recognize` result). The worst-case token count of a file is `1 + (C + 1) * Q + 1` with `r` that
  lookahead, `Q = (r + 1) * 10 + 1` and `C` the number of whole cadence chunks of `(r + 1) * 1280`
  samples. At the reference `r = 13` (`Q = 141`, 17,920 samples per chunk) a 600 s file needs at
  least 75,578; the model plugin's README serves 76,800. A deployment that runs files at 560 ms
  (`r = 6`, `Q = 71`, 8,960 samples per chunk) needs 76,114. The frontend does not compute this
  count (it needs model-specific knowledge outside the structural seam); the formula and values
  here are for operators. Core's own `POST /v1/audio/transcriptions` admits files up to the same
  decode-duration limit, so the same value covers it.

One API-server process is required: `init_state` refuses to start when `--api-server-count` is
above 1 (startup checks). Data parallel size above 1 is refused at startup, because the model
requires parallel sizes of 1 and slot count and active limit are per engine.

The guide also sets:

- **Probes.** Orchestrators probe core's `/ready` for readiness. It needs no token, aggregates
  engine health, every plugin's `ready` and `state.draining`, and answers 503 from the moment core
  begins draining (operations leaf).
- **Realtime client.** The stock Riva Realtime client is run with `--server host:port/plugins/riva`,
  because it appends `/v1/...` to that argument (operations leaf).
- **Core's file route.** Core's `POST /v1/audio/transcriptions` is outside the frontend's ingress
  cap, file gate and guard, so it is not exposed to untrusted clients; they reach complete files
  through the frontend's route or `Recognize` (files leaf).
- **API key and the stock Realtime client.** Under `--api-key` the namespace requires a bearer
  token, and the stock Realtime client sends no `Authorization` header; a deployment that serves it
  puts an intermediary in front that adds the token to the bootstrap request and the WebSocket
  handshake.
- **Kill grace.** The orchestrator's grace period before it kills the process (for Kubernetes,
  `terminationGracePeriodSeconds`) is above `--shutdown-timeout`, with room for `stop` and the
  engine's shutdown, so the drain and its settlement are not cut short by a kill. At a zero
  shutdown budget the kill grace is also the only bound, besides a repeated signal, on a `stop`
  whose settlement waits on an engine call that does not end on the stopped engine (`stop(state)`).
- **Python API server only.** The serve mode that runs core's Rust frontend
  (`vllm/entrypoints/cli/serve.py`) starts a Rust frontend process instead of Python API-server
  processes, so no endpoint plugin loads and `init_state`'s checks never run. The frontend does not
  support that mode.

## Decisions & Alternatives

| Decision | Chosen | Alternatives Considered | Rationale |
| --- | --- | --- | --- |
| Vehicle | RFC 2 endpoint plugin, one class | A fork-only application plugin; patches to core | The lifecycle (`start`, `drain`, `stop`, `request_shutdown`) is what a long-lived listener needs and has an upstream path; patching core would merge Riva-shaped code into vLLM |
| vLLM version gate in packaging | None; `init_state` checks capabilities (session factory constructor raising on a missing engine call, the `realtime` task) | A lower bound on `vllm`; a version check at startup | No release carries RFC 1 and RFC 2 yet, so a bound would name a version that does not exist; a capability failure names the missing call, which is what the operator needs. A version gate is deferred until releases carry both |
| Engine deadlines | Read the `stream_admission` mapping from `additional_config` and apply the documented defaults when absent; fail on an unreadable value | Import core's `StreamAdmissionConfig`; read nothing and use fixed bounds | The mapping is the documented public setting; importing a scheduler-internal class couples the package to core internals; fixed bounds cannot be checked against what the engine enforces. A test pins the copied defaults against core in the RFC 1 + RFC 2 build |
| CPU test build | RFC 2 without RFC 1, except the deadline-default pin, which runs in the RFC 1 + RFC 2 build | Every CPU test on the RFC 1 + RFC 2 build | The frontend needs nothing from RFC 1 beyond what the factory provides, so an RFC 2-only build tests the plugin against the lifecycle it targets; only the pin reads RFC 1's settings |
| Frontend bounds below engine deadlines | Strictly below each, checked at startup, with defaults that pass | Equal bounds; engine deadlines as the only idle mechanism | A frontend timeout can name its cause and release its own resources first; the engine abort is then a backstop whose lease failure `expired` is projected as an idle timeout |
| Slot count and active limit | `num_gpu_blocks - 1` and `max_num_active_seqs or max_num_seqs`, read from `vllm_config` in `init_state` | A new `EngineClient` method; probing the scheduler | These values are already in the configuration the plugin is handed after the engine reports its cache; a new protocol method would be a core change outside RFC 1 and RFC 2 |
| Derived `max_concurrent_files` default | A quarter of the smaller of slots and active limit, between 1 and 4; files disabled when capacity is below 2 and the cap is not configured | A fixed default | A fixed number would fail small deployments and over-admit files on large ones; the HLD's startup check is kept as the guard |
| Cleanup bound expiry | Hand the session to the factory's cleanup; not an escalation trigger | Escalate to `request_shutdown()` | The factory already owns a session whose owner is gone and retains a fault only for work no caller can still see (session leaf, Factory obligation 3); restarting the process because one release was slow turns a slow engine call into an outage |
| Escalation triggers | `engine_dead`, a retained fault collected during service, an unexpected listener exit | A failure latch; every error; every generation-task exception | A latch outlives its cause. Orderly restart is core's tool for a plugin that cannot continue. Capacity refusals and request failures are routine and already surfaced to their owner; escalating them would turn load into restarts |
| Fault collection | A task polling `factory.drain(now)` every second | A fault callback registered with the factory; collecting only at shutdown | The seam has no callback and adding one changes the model plugin's protocol; collecting only at shutdown leaves a dead engine unnoticed during service. The 1 s period is a constant, not an option |
| HTTP and WebSocket paths | Every route under `/plugins/riva/v1/`, as vLLM's security guide recommends for endpoint plugins; core's routes untouched | A dispatcher middleware claiming `/v1/realtime?intent=transcription`; an override of core's `POST /v1/audio/transcriptions` | The frontend would change core's own surfaces, and core's routes would behave differently depending on which plugins are loaded. The stock Realtime client reaches the namespace through its `--server` argument (operations leaf) |
| `--api-key` on the namespace | A path-scoped pure-ASGI middleware from `attach_router` that calls core's `verify_token` with core's configured tokens | Rely on core's middleware; a copy of the token check; a FastAPI dependency on each route; a change to core's guarded prefixes | Core's check covers only `/v1`, `/v2`, `/inference` and `/cohere`, so the namespace would be unauthenticated; a copy is a second implementation of a security check; a dependency covers only routes that declare it, not unknown paths under the prefix, and needs separate WebSocket handling; a prefix change is a core change outside RFC 1 and RFC 2 |
| `--api-key` scope | Every path under `/plugins/riva/` after `root_path`, as the HLD sets; `/plugins/riva` itself untouched | Also guard `/plugins/riva` without the trailing slash | The frontend has no route outside `/plugins/riva/v1/`, so that path reaches no frontend handler; guarding exactly the HLD's prefix keeps one scope across the documents |
| Token source | Read from core's installed `AuthenticationMiddleware` in `init_state`, used only when a key is configured by core's rule and exactly one readable registration exists; every disagreement fails startup | Recompute from `--api-key` and `VLLM_API_KEY`; read in `attach_router`; use installed tokens whenever readable | Core registers its middleware after the plugin hooks, so `attach_router` cannot see it; the installed tokens are what core enforces on its own prefixes, and recomputing them alone is a second copy of core's precedence rule. The configured-key signal is a cross-check in both directions: a key without readable middleware would leave the namespace open, and middleware without a key means a registration core did not make, whose tokens the frontend cannot trust |
| WebSocket authentication refusal | Refused before accept; core's 401 where the ASGI denial-response extension exists, otherwise a close before accept (HTTP 403) | 401 unconditionally; accept, then close with a code | A denial body needs the extension, which not every ASGI server offers; a close before accept fails closed on every server; accepting would expose the WebSocket to an unauthenticated client |
| One API server process | `init_state` refuses a count above 1, `None` counting as one | A declaration on the plugin class that core enforces; per-process gRPC ports | Core's endpoint-plugin protocol has no such declaration; every process would bind the same gRPC port; per-process ports would need a client-visible port scheme |
| Route table after attach | `init_state` verifies core's final table still routes the frontend's paths to its handlers | Rely on plugin order | Core attaches in entry-point discovery order and `VLLM_PLUGINS` only filters membership, so a later plugin can override the frontend and no list order prevents it |
| Route collisions | Raise in `attach_router` for any route of the frontend (all are new) that an existing HTTP route, WebSocket route or mount would serve | Last registration wins (core's behaviour); check HTTP routes only | Core overrides silently; two plugins serving one path is a deployment error that should stop the server; a WebSocket route or mount earlier in the table wins over a later route, so checking HTTP routes alone misses it |
| Core's route table | CPU tests verify that core's table with the plugin is core's table without it plus exactly the frontend's namespaced routes | Check only that the frontend's routes are present | The namespace exists so that core's surfaces do not depend on which plugins load; only a comparison against core's own table shows that no core route changed |
| Configuration channel | `--additional-config` key `riva_frontend`, strict validator, errors listed together | A separate flag; a file path option | One flag surface already exists and core's own settings use it; strictness names the failing option. Core hashes `additional_config` into the compile cache key, which costs nothing under eager execution |
| Plugin shutdown budget | No separate plugin shutdown budget; `drain` uses the time core passes it | A `plugin_shutdown_grace` option as the drain budget | Core passes `drain` the remaining `--shutdown-timeout`; two budgets would disagree |
| File backlog and streams | File owners count against a sub-cap `max_file_owners` inside the ingress cap | One shared cap | A backlog of waiting files would fill a shared cap and refuse streams with `busy`; the sub-cap leaves streams the remaining headroom |
| Sub-unit traffic | The engine is the authority; document the limit | A frontend check on units delivered; a seam change | The engine's clock resets on units sent, which the frontend cannot observe without a seam change; the engine's expiry is already a safe backstop |
| Startup order | Factory built last; any later failure runs the factory settlement; admission opens last in `start`; a failed or cancelled `start` tears the factory down and awaits what it cancels before re-raising | Build it first; re-raise without awaiting | A factory owns engine-facing tasks, so a failed startup must not leave any, and core calls `stop` only for a plugin whose `start` completed; admitting before `start` completes would admit owners into a failing plugin |
| Live streams at drain | The frontend finalizes them at the settle point (tail, `flush`, final, then gRPC `OK` or WebSocket 1001); failures abort with `UNAVAILABLE` or 1012 | Cancel everything at the settle point; wait for clients | Clients that sent audio get a complete transcript; an aborted stream is told to retry elsewhere |
| Drain deadline | A graceful-work target `D = deadline - max(5 s, 10% of timeout)`; settlement may run past it | The full core deadline | The target aims to leave core time to shut the engine down after plugins drain; core reserves no engine allowance, so the target is not a guarantee, and settlement takes precedence |
| Shutdown settlement | Graceful work within `D`, then abort and await every owner, continuation, unsettled send and the fault collector, then the factory settlement, before `drain` returns, raises or re-raises a cancellation | Return at the deadline with work still cancelling; leave the remainder to `stop` | RFC 2 requires owned work to be settled so that it cannot submit engine requests after the hook returns; core shuts the engine down once `drain` returns |
| Factory settlement | Owners, continuations and unsettled sends settle and the fault collector ends; then `factory.drain(math.inf)`, then `teardown_all()`, await every task it returns, then `factory.drain(now)` once more, and report the faults both `drain` calls return plus last-resort cancellations | `teardown_all()` first and await what it returns; `factory.drain` with a large finite deadline, then `teardown_all()`; the lease's `teardown()` on bound expiry | `teardown_all` cancels unconditionally, so calling it while a generation task is still completing the engine client's abort (after `abort` or a cancelled `open`) interrupts that abort. `factory.drain(math.inf)` lets every handed-over generation task finish its cancellation, so the later `teardown_all` cancels only ended tasks. A finite bound reintroduces a deadline after which `teardown_all` would cancel a task still awaiting the abort. `teardown()` also cancels again and starts no factory task that the settlement could await |
| Cancellation of a generation task | Owned by the party that ends the session: `abort` or a cancelled `open`; no cancellation reaches a generation task after its cancellation handler has begun; the vehicle cancels a factory task only as a last resort, when `teardown_all` returns one still running, and records it as a fault | `teardown_all` alone, as the single cancellation; re-cancel still-running tasks periodically; a change to `teardown_all` so that it skips tasks already cancelled; a literal limit of one `cancel()` request per task | The session leaf's cleanup cancels the generation task through `abort` before the factory closes, so `teardown_all` can never be the only cancellation; a cancellation delivered while the task awaits the engine client's abort interrupts that abort. A cancelled `open` requests cancellation twice with no suspension between the requests, which is harmless, so a request count is the wrong property to test. Skipping cancelled tasks in `teardown_all` would alter its terminal contract (it cancels every task it owns, `SESS-OWN-005`) for a case the settlement order already makes safe; the model-plugin code change the frontend does require is confined to fault retention (Open Questions, Dependencies). Core's repeated-signal exit bounds a task that never ends |
| Factory seam obligation | The frontend imposes the numbered obligations of the session leaf (Factory obligations 1 to 7) on every factory, checks them in its seam conformance suite, and is owed them as statements in the model plugin's session specification; obligations 3 and 4 (retained faults, sole reporter) also need a change to the model plugin's code. The suite also tests what `SESS-OWN-004`, `SESS-OWN-005`, `SESS-OWN-006`, `SESS-REL-003` and `SESS-ABORT-002` state | Settle through a new factory method that also awaits; collect faults from the tasks `teardown_all` returns; leave the coverage untested | After `drain(math.inf)` an ended cleanup task has left the factory's table and dropped its session, so `teardown_all` no longer returns it or its generation task, and the done-callback that retrieves a generation task's exception consumes it without recording it: an exception raised while the task awaits the engine abort would never be reported. Only the factory sees every task it owns, so it must keep their faults. The model plugin's factory already accepts an infinite deadline (its wait timeout is the non-negative time to the deadline), its cleanup tasks already await the generation task, and its `release` first suspends inside the covered retirement wait, but its specification states none of these; a gap in another factory would break `drain`'s guarantee silently, so the suite that every factory runs checks every obligation |
| Cleanup after the drain deadline | `session_cleanup_timeout`, then the same cancelled-`release` hand-over as before the deadline; the factory settlement ends whatever the hand-over leaves | The time remaining to `D`; unbounded `abort` and `release`; the lease's `teardown()` on expiry | After `D` the remaining time is zero, which would skip `abort` and `release`; an unbounded call can hang on a stuck engine; one expiry rule in and out of shutdown keeps the owner's sequence identical; the hand-over's cleanup task, which the factory settlement awaits, covers the generation task, which `teardown()` would cancel again while it awaits the engine client's abort, and leave unawaited |
| Rollback ownership of factories | The vehicle settles every constructed factory: a discarded candidate when discarded, the selected one on any later failure or at shutdown; the constructor and `init_state` start no task | Settle only the selected factory; allow constructors that start tasks | A discarded candidate is still a factory whose tasks nobody else would end; core has no hook between `init_state` and `start`, so only an inert factory makes a failure in that interval safe |
| `stop` order | `drain`'s settlement order: owners take the shutdown abort row, then the factory settlement | Close the factory first and have owners call the lease's `teardown()` | Core does not guarantee a stopped engine when `stop` runs, so a further cancellation of an aborted generation task could still interrupt the engine client's abort; one settlement order serves `drain`, `stop` and rollback |
| `stop` wait | Await every owner, continuation, send and factory task, with no fixed time limit | A fixed 1 s wait that returns whether or not tasks finished | RFC 2 makes `stop` cooperative and requires it to settle background work; a task left running could still call the engine client while core deletes app state. Core's repeated-signal exit bounds a hang |
| Refusal sends during settlement | Each admission or query refusal send is awaited inline by its own handler under `response_send_timeout`; a send abandoned after its bound enters the registry from the moment of abandonment while the registry is open; settlement closes the registry before its final wait, and a refusal send that passes its bound after that is cancelled and awaited by its own handler | Register every refusal send in the registry when it starts; snapshot the abandoned sends when settlement begins; refuse all requests without sending once settlement begins | Core keeps HTTP serving while plugins drain, so refusals arrive during settlement. A snapshot misses a send abandoned while the final wait runs, which then outlives `drain` as detached plugin work. Registering every send at creation adds a registry entry to the common path for a case the close already settles. Refusing without a send would drop the `service_unavailable` answer that tells a client to retry elsewhere |
| Fault reporting across settlements | The factory is the sole reporter of its tasks' faults, each exactly once through `drain`; the vehicle reports `drain`'s faults as returned, calls `factory.drain(now)` again after awaiting what `teardown_all` returns, and originates only last-resort cancellations, keyed by the task it saw cancelled | Merge `drain`'s faults with the exceptions of the tasks `teardown_all` returns and deduplicate per task; report everything `teardown_all` returns each time | `drain` returns exceptions without the task that raised them, and exception identity is not task identity (two tasks can raise one object; a retained retirement failure is not the cleanup task's own result), so a per-task deduplication cannot be implemented through the seam. `teardown_all` returns ended tasks still in the factory's table again, so reading their exceptions makes `stop` after a `drain` settlement log the same faults twice. One reporter makes repetition impossible by construction |
| Ending the fault collector | Signal it and await its end; its call in progress finishes | Cancel it | The factory clears faults as `drain` returns them, so cancelling a call in progress could lose the faults it was taking |
| Draining signal | Admission and `ready` read `state.draining` besides the plugin's own state | Close admission only in `drain` | Core sets the flag before any `drain` call and calls none when no budget remains; reading it refuses new owners from that moment |
| `max_session_duration` | Off by default | 600 s | A default cap truncates legitimate long streams; operators set it when they want one |
| Memory bound | `max_file_owners` times the body bound plus per-stream bounds, default file sub-cap modest | Unbounded waiters | The bound is stated so operators can size memory; the defaults keep it near half a gigabyte |
| Data parallel | Refused when above 1 | Support per-rank ports | The model requires parallel sizes of 1; slots and active limit are per engine |
| Tiny deployments | Files disabled with a warning when no cap is configured and capacity is below 2; the route stays registered and refuses with 501; an explicit failing cap fails startup | Always fail; remove the route | A one-slot server is valid for streams only; an operator who configured files expects them |
| File-path engine death | Core's watchdog | Frontend escalation | The file request lives in core's serving, which already reports it |
| `pin` provenance | May be a commit SHA | Refuse all bare hex | The rule exists to keep registry identities out; a commit SHA is the deployment's own identifier |
| Interpreter range | The range vLLM declares, with a fail-closed gate | A range chosen by the frontend | The plugin runs in vLLM's interpreter; a narrower range excludes supported hosts |
| Served identity | `model_name()` once, primary identity only on Riva and Realtime | Accept every alias | Riva clients read one name from the listing; the registry is the sole authority for it |

## Open Questions & Future Decisions

### Dependencies

**Fault retention in the model plugin.** Factory obligations 3 and 4 (session leaf) need a change to
the model plugin's code. Today its done-callback retrieves a generation task's exception without
recording it, its recorded faults are the cleanup tasks' retirement-wait failures alone, and its
cleanup task drops the session when it ends. The other obligations are already true of its code and
owe only statements in its session specification. Until the change lands, the frontend guarantees
that every settlement still ends and awaits every task, in the order above, and reports every
retirement-wait failure; a generation task's exception that no `open` or lease call surfaced (for
example, from its cancellation handler while it awaits the engine abort) may go unreported.

### Deferred
1. A vLLM version gate in `init_state`, once releases carry RFC 1 and RFC 2.
2. If core adds callbacks or a client protocol for stream-admission settings, slot count and
   active limit, `init_state` reads them there instead of from configuration.
3. **Startup failure between `init_state` and `start`.** Core has no hook for this interval. The
   frontend relies on the invariant that `init_state` and the factory constructor start no task,
   listener or session (startup checks); a factory constructor that acquires resources would need
   a core cleanup hook or a broader lifecycle decision. A test verifies the invariant against the
   model plugin's factory constructor.
4. **gRPC health at zero budget.** With `--shutdown-timeout` 0 (core's default) no `drain` runs,
   and gRPC health stays `SERVING` until `stop` while core's `/ready` and admission refuse
   (Settlement, Zero budget). A later change could have the fault collector's tick set
   `NOT_SERVING` once it observes `state.draining`.
5. **Rust frontend mode.** Unsupported (Deployment); supporting it needs a separate design and
   evidence, since no Python endpoint plugin loads there.

## References

- RFC 2 endpoint plugins: `docs/design/endpoint_plugins.md` and
  `vllm/plugins/endpoint_plugins/interface.py` on `pdtgct/vllm` branches `endpoint-plugin-lifecycle`
  (vllm#65546: start and stop) and `endpoint-plugin-readiness-drain-review-20261111` (readiness,
  `state.draining` and drain settlement); core's shutdown task, which drains plugins and then
  stops the engine, and its plugin `stop` call at lifespan exit:
  `vllm/entrypoints/launchers/launcher.py` and `vllm/entrypoints/launchers/utils/server_utils.py`;
  the engine client's abort of a cancelled `generate()`: `vllm/v1/engine/async_llm.py`; core's
  `/ready` and `/health`:
  `vllm/entrypoints/serve/instrumentator/health.py`; the single-process API server count:
  `vllm/entrypoints/cli/serve.py`.
- vLLM's endpoint-plugin security guidance: `docs/usage/security.md` (namespaced plugin routes; the
  prefixes `--api-key` covers).
- RFC 1 persistent state and stream admission: `docs/design/persistent_state.md`,
  `vllm/v1/core/stream_admission.py`, `vllm/v1/core/state_slot_manager.py` on `pdtgct/vllm` branch
  `persistent-state-primitive-v2`.
- Core middleware order and authentication: `vllm/entrypoints/launchers/app.py`,
  `vllm/entrypoints/serve/middleware/register.py`, `.../authenticate.py` (`AuthenticationMiddleware`,
  `verify_token`).
- Model plugin: `pdtgct/vllm-nemotron-asr-streaming-plugin` (`docs/intent/session/`,
  `docs/intent/registration/`).
- High-level design: `docs/high-level-design.md`.
