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

- **Core's operations stay core's.** vLLM's `/health`, `/ready`, `/v1/models`, `/version` and
  metrics endpoints describe the serving implementation, not the ASR wire contract; the plugin
  neither replaces nor shadows them.
- **Same truth as core's readiness.** The NIM readiness route answers from the same predicate the
  plugin gives core's `/ready`, so the two cannot disagree about the plugin.
- **Report only this deployment's own facts.** Metadata carries identifiers the deployment authors
  for itself and presents no registry identity.
- **No bespoke clients.** Evaluation uses the canonical public tools.

## Routes

The plugin serves four `GET` routes on the HTTP listener, under the frontend's namespace
(vehicle leaf), with NIM-shaped response and error bodies:

| Route | Meaning | Body |
| --- | --- | --- |
| `GET /plugins/riva/v1/health/ready` | readiness | 200 `{"object": "health.response", "message": "ready", "status": "ready"}`; 503 `{"object": "health.response", "message": "not ready", "status": "not_ready"}` |
| `GET /plugins/riva/v1/health/live` | liveness | 200 `{"object": "health.response", "message": "live", "status": "live"}` |
| `GET /plugins/riva/v1/version` | release | 200 `{"release": <frontend distribution version>, "api": "v1"}` |
| `GET /plugins/riva/v1/metadata` | deployment identity | 200, see below |

The NIM paths keep their `v1/...` suffix under the plugin's prefix, so NIM tooling that takes a
base URL reaches them at `host:port/plugins/riva`.

**Readiness** is a JSON object carrying `status`, and it equals the plugin's `ready(state)`
predicate (vehicle leaf): true only while the gRPC listener is serving, neither core
(`state.draining`) nor the plugin is draining, the plugin is not torn down, and no cleanup fault is
recorded. It reads attributes only. The NIM route reports the plugin's part. Core's `/ready` is
the process's aggregate readiness: engine health, every loaded plugin's `ready` and
`state.draining`. Core's `/health` reports engine health only and does not read plugin readiness.

**Liveness** reports process survival rather than model readiness: it answers 200 whenever the
HTTP server serves it and the request passes authentication (below). There is no `not_live`
response: `stop` runs after HTTP stops, so the route cannot be reached in a torn-down plugin.

**Core's WebSocket gauges.** Core's WebSocket metrics middleware wraps every WebSocket route,
including the frontend's, and counts a connection from its accept. Core's realtime connection
gauges therefore count Riva Realtime sessions on `/plugins/riva/v1/realtime` and the admission
refusals the route accepts before closing (streaming leaf), and do not count a connection refused
for its query, which is never accepted.

**Metadata** reports this deployment's own identity and nothing else:

- `version`: the same release as `/plugins/riva/v1/version`;
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

**Authentication.** Under `--api-key` (or `VLLM_API_KEY`), the frontend's path-scoped middleware
(vehicle leaf) applies core's tokens, through core's token check, to every HTTP and WebSocket
request under `/plugins/riva/`: these four routes, the Realtime bootstrap, the Realtime WebSocket
route and the file route all require the bearer token. An HTTP request without it gets core's 401;
a WebSocket handshake without it is refused before accept (vehicle leaf, WebSocket refusal).
`OPTIONS` is exempt, as in core. Paths outside the namespace are untouched, so core's `/health` and
`/ready` need no token, while core's `/v1/...` routes keep core's own check. The gRPC listener is
outside `--api-key`.

**Probes.** Orchestrators probe core's `/ready` for readiness: it needs no token, aggregates engine
health, every plugin's readiness and core's draining flag, and turns 503 from the moment core
begins draining. A liveness probe that cannot supply a token uses core's `/health`. Deployments
whose tooling probes the NIM routes supply the token.

## Coexistence with core's routes

Core's existing `/health`, `/ready`, `/v1/models`, `/version`, `/v1/realtime`,
`/v1/audio/transcriptions` and metrics endpoints remain authoritative and are not replaced,
shadowed or reinterpreted. The plugin adds nothing outside `/plugins/riva/`. The plugin-owned
compatibility paths are exactly `/plugins/riva/v1/health/ready`, `/plugins/riva/v1/health/live`,
`/plugins/riva/v1/version` and `/plugins/riva/v1/metadata`. Core has no `/live`, and there is no
host liveness route to preserve.

The frontend's route set is the four operational routes, the Realtime bootstrap
(`POST /plugins/riva/v1/realtime/transcription_sessions`), the Realtime WebSocket route
(`/plugins/riva/v1/realtime`) and, when added, the file route
(`POST /plugins/riva/v1/audio/transcriptions`). Every route in it is new. The checks that keep this
set and core's routes apart (the collision check in `attach_router`, the final-table check in
`init_state`, and the CPU tests that core's own table is unchanged) are the vehicle leaf's (Route
checks).

## Evaluation discipline

The Riva gRPC and Riva Realtime WebSocket surfaces are driven in evaluation by the canonical
`nvidia-riva/python-clients` tools. The gRPC tools take the server's gRPC `host:port`. The Realtime
client builds its bootstrap and WebSocket URLs by appending `/v1/...` to its `--server` argument,
so it is run with `--server host:port/plugins/riva` and its default endpoint and query. The file
route, `POST /plugins/riva/v1/audio/transcriptions`, is driven by an OpenAI-style multipart
transcription request, because those tools provide no HTTP transcription client; the route keeps
core's request and response contract, so a request in core's documented shape is the right probe.
The frontend implements no bespoke Riva-family client. `GetRivaSpeechRecognitionConfig` is
exercised by the stock model-listing script.

The stock Realtime client sends no `Authorization` header on either request, so it is evaluated
against a server with API-key authentication disabled: neither `--api-key` nor `VLLM_API_KEY` set.
A deployment that runs it under `--api-key` puts an intermediary in front that adds the bearer
token. Bearer-token behaviour of the namespace is verified separately, by CPU tests that send
requests with and without the token.

## Decisions & Alternatives

| Decision | Chosen | Alternatives Considered | Rationale |
| --- | --- | --- | --- |
| Where the NIM routes live | `operations` leaf; `CLI` discipline with them | A separate verification leaf | `CLI` is four sentences about how to drive the surfaces; it has no component of its own and sits with the surfaces' operational tooling |
| Readiness | `ready(state)` | Combine with engine health; probe the listener | One predicate that core's `/ready` also reads; engine health is core's, aggregated by `/ready`, and `engine_dead` already withdraws readiness |
| Orchestrator probe | Core's `/ready` | Core's `/health`; the NIM readiness route | `/ready` aggregates engine health, plugin readiness and draining and needs no token; `/health` is engine-only and stays 200 while the plugin drains; the NIM route needs a token under `--api-key` and reports the plugin alone |
| Liveness | Always 200 while HTTP serves, to a request that passes authentication; no `not_live` | Mirror readiness; a 503 after teardown | Liveness restarts a process; a draining or faulted plugin should not be killed by a probe before it has drained, and the route is unreachable after teardown because `stop` runs after HTTP stops |
| Route set | Four paths under `/plugins/riva/v1/` | Add NIM's other paths; serve them at `/v1/...` | NIM's other paths would collide with or reinterpret core's operations; paths under `/v1` are core's namespace |
| Metadata identity | Allowlist of this deployment's own identifiers | Mirror a NIM identity | A deployment must not present another system's registry identity as its own |
| Authentication | Core's tokens and token check on every request under `/plugins/riva/`, through the path-scoped middleware | Exempt the NIM routes; leave the namespace unauthenticated | The plugin applies core's `--api-key` to its own surfaces as core does to `/v1`; core's `/ready` and `/health` remain the unauthenticated probes |
| Stock Realtime client under `--api-key` | Evaluated with authentication disabled; an intermediary adds the token in authenticated deployments | Accept unauthenticated Realtime traffic under `--api-key` | The client sends no `Authorization` header; exempting it would leave the namespace open |
| File-route evaluation client | OpenAI-style multipart request | A NIM HTTP client | It is core's contract; the canonical tools have no HTTP transcription client |

## Open Questions & Future Decisions

### Deferred
1. Whether Kubernetes charts also probe `/plugins/riva/v1/health/ready` with a token, beside core's
   `/ready`; a deployment decision, not a frontend one.

## References

- Core routes: `vllm/entrypoints/` health (`serve/instrumentator/health.py`: `/health`, `/ready`),
  version and models routers; middleware `vllm/entrypoints/serve/middleware/authenticate.py` and
  `register.py`; WebSocket gauges `vllm/entrypoints/speech_to_text/realtime/metrics.py`; on
  `pdtgct/vllm` branches `endpoint-plugin-lifecycle` and
  `endpoint-plugin-readiness-drain-review-20261111`.
- RFC 2: `docs/design/endpoint_plugins.md` (`ready`, route override behaviour);
  `docs/usage/security.md` (namespaced plugin routes, `--api-key` prefixes).
- Public NIM health and metadata documentation.
- `nvidia-riva/python-clients` (stock evaluation tools; `riva/client/realtime.py` builds its URLs
  from `--server`).
- Vehicle leaf (readiness predicate, configuration), streaming leaf (bootstrap), files leaf.
