---
parent: high-level-design
prefix: ING
facets: [SHIM, CLI]
---

# Operations

## Context and Design Philosophy

Operators and Riva or Speech NIM tooling expect a small set of operational HTTP endpoints next to
the inference surfaces: readiness, liveness, version and metadata. This leaf defines those routes,
how they coexist with the routes vLLM core already owns, and the discipline for evaluating the
frontend with canonical clients. The routes contain no inference behaviour.

Principles:

- **Core's operations stay core's.** vLLM's `/health`, `/v1/models`, `/version` and metrics
  endpoints describe the serving implementation, not the ASR wire contract; the plugin neither
  replaces nor shadows them.
- **Same truth as core's readiness.** The NIM readiness route answers from the same predicate the
  plugin gives core's `/health`, so the two cannot disagree about the plugin.
- **Report only this deployment's own facts.** Metadata carries identifiers the deployment authors
  for itself and presents no registry identity.
- **No bespoke clients.** Evaluation uses the canonical public tools.

## Routes

The plugin serves four `GET` routes on the HTTP listener, with NIM-shaped response and error
bodies:

| Route | Meaning | Body |
| --- | --- | --- |
| `GET /v1/health/ready` | readiness | 200 `{"object": "health.response", "message": "ready", "status": "ready"}`; 503 `{"object": "health.response", "message": "not ready", "status": "not_ready"}` |
| `GET /v1/health/live` | liveness | 200 `{"object": "health.response", "message": "live", "status": "live"}` |
| `GET /v1/version` | release | 200 `{"release": <frontend distribution version>, "api": "v1"}` |
| `GET /v1/metadata` | deployment identity | 200, see below |

**Readiness** is a JSON object carrying `status`, and it equals the plugin's `ready(state)`
predicate (vehicle leaf): true only while the gRPC listener is serving, the plugin is not draining
or torn down, and no cleanup fault is recorded. It reads attributes only. Core's `/health` combines
this with the engine's own health; the NIM route reports the plugin's part.

**Liveness** reports process survival rather than model readiness: it answers 200 whenever the
HTTP server serves it. There is no `not_live` response: `stop` runs after HTTP stops, so the route
cannot be reached in a torn-down plugin.

**Core's WebSocket gauges.** Core's WebSocket metrics middleware wraps the dispatcher, so core's
realtime connection gauges also count claimed Riva Realtime sessions.

**Metadata** reports this deployment's own identity and nothing else:

- `version`: the same release as `/v1/version`;
- `modelInfo`: one entry whose `shortName` is the resolved served-model identity;
- `provenance`: the deployment's image identifier, release pin identifier, precision identifier and
  the resampler identifier (`resampler`), built from exactly those four typed fields (an allowlist,
  not a filtered copy of a mapping), each already validated at startup (vehicle leaf);
- the remaining fields of the NIM shape (`repository_override`, `assetInfo`, `licenseInfo`) empty.

The `pin` may be a git commit SHA; the provenance rule (vehicle leaf) excludes only registry
references and model-profile hashes presented as the deployment's identity.

The route reports only identifiers of this deployment and does not present a Speech NIM model
registry identity, registry reference or profile identifier. Surfaces that publish provenance (the
Realtime session) take it from the same object.

**Authentication.** Under `--api-key`, core's authentication covers paths under `/v1`, so these four
routes require the bearer token while core's `/health` does not. Deployments that probe with the
NIM routes supply the token; probes that cannot should use `/health`.

## Coexistence with core's routes

Core's existing `/health`, `/v1/models`, `/version` and metrics endpoints remain authoritative and
are not replaced, shadowed or reinterpreted. The plugin adds no `GET /v1/health` and no
`GET /v1/metrics`. The plugin-owned compatibility paths are exactly `/v1/health/ready`,
`/v1/health/live`, `/v1/version` and `/v1/metadata`. Core has no `/live`, and there is no host
liveness route to preserve.

Core's route override replaces a same-path, same-method operation silently, in registration order.
For these four paths, which the plugin adds as new routes, `attach_router` therefore checks the
route table (core's routes and routes of plugins loaded earlier) and raises if any already serves
the same method and path, so startup fails rather than depending on registration order. The route
set is verified against core's table by a CPU test. The check cannot see routes added after it, and `VLLM_PLUGINS` only filters which plugins load: core attaches them in entry-point discovery order, so no setting makes this plugin attach last. Instead `init_state` verifies that core's final route table still routes each of the frontend's paths to the frontend's own handler and fails startup otherwise (vehicle leaf). The Realtime bootstrap
(`POST /v1/realtime/transcription_sessions`) is checked the same way; the file route is a
deliberate override and is governed by the files leaf.

## Evaluation discipline

The Riva gRPC and Riva Realtime WebSocket surfaces are driven in evaluation by the canonical
`nvidia-riva/python-clients` tools. The file route is driven by an OpenAI-style multipart
transcription request, because those tools provide no HTTP transcription client; that route is
core's contract, so a request in core's documented shape is the right probe. The frontend
implements no bespoke Riva-family client. `GetRivaSpeechRecognitionConfig` is exercised by the
stock model-listing script.

## Decisions & Alternatives

| Decision | Chosen | Alternatives Considered | Rationale |
| --- | --- | --- | --- |
| Where the NIM routes live | `operations` leaf; `CLI` discipline with them | A separate verification leaf | `CLI` is four sentences about how to drive the surfaces; it has no component of its own and sits with the surfaces' operational tooling |
| Readiness | `ready(state)` | Combine with engine health; probe the listener | One predicate that `/health` also reads; engine health is core's `/health` and `engine_dead` already withdraws readiness |
| Liveness | Always 200 while HTTP serves; no `not_live` | Mirror readiness; a 503 after teardown | Liveness restarts a process; a draining or faulted plugin should not be killed by a probe before it has drained, and the route is unreachable after teardown because `stop` runs after HTTP stops |
| Route set | Four paths, no `/v1/health` or `/v1/metrics` | Add NIM's other paths | Those would collide with or reinterpret core's operations |
| Collisions | Fail startup in `attach_router` | Last registration wins | Registration order is an accident; a duplicate operational route is a deployment error |
| Metadata identity | Allowlist of this deployment's own identifiers | Mirror a NIM identity | A deployment must not present another system's registry identity as its own |
| Later overrides | `init_state` verifies the final route table; startup fails if a path no longer reaches the frontend | Ask operators to list the plugin last | `VLLM_PLUGINS` does not set attachment order; verifying the final table detects any override |
| Authentication | Core's `/v1` rule applies | Exempt the NIM routes | The plugin does not relax core's authentication; `/health` remains the unauthenticated probe |
| File-route evaluation client | OpenAI-style multipart request | A NIM HTTP client | It is core's contract; the canonical tools have no HTTP transcription client |

## Open Questions & Future Decisions

### Deferred
1. Whether Kubernetes charts should probe `/health` (core) or `/v1/health/ready` with a token; a
   deployment decision, not a frontend one.

## References

- Core routes: `vllm/entrypoints/` health, version and models routers; middleware
  `vllm/entrypoints/serve/middleware/authenticate.py` on `pdtgct/vllm`.
- RFC 2: `docs/design/endpoint_plugins.md` (`ready`, route override behaviour).
- Public NIM health and metadata documentation.
- `nvidia-riva/python-clients` (stock evaluation tools).
- Vehicle leaf (readiness predicate, configuration), streaming leaf (bootstrap), files leaf.
