# Operations specs

Design: [operations-design.md](operations-design.md). Prefix `ING`, facets SHIM, CLI.

## Operational routes (SHIM)

- [x] **ING-SHIM-007**: The plugin shall serve `GET /v1/health/ready`, `GET /v1/health/live`, `GET /v1/version` and `GET /v1/metadata` on the HTTP listener with NIM-shaped response and error bodies, and the readiness body shall be a JSON object whose `status` is `ready` when the plugin's `ready(state)` predicate (ING-EP-009) is true and `not_ready` when it is false (ING-SHIM-009).
- [x] **ING-SHIM-009**: The readiness route shall answer 200 with `{"object": "health.response", "message": "ready", "status": "ready"}` when ready and 503 with `message` `not ready` and `status` `not_ready` otherwise.
- [x] **ING-SHIM-010**: The liveness route shall answer 200 with `{"object": "health.response", "message": "live", "status": "live"}` whenever the HTTP server serves it and shall define no `not_live` response.
- [x] **ING-SHIM-011**: The version route shall answer 200 with `release` equal to the frontend distribution version and `api` equal to `v1`.
- [x] **ING-SHIM-017**: The metadata `version` field shall equal the `release` that `GET /v1/version` reports.
- [x] **ING-SHIM-002**: `GET /v1/metadata` shall report only identifiers of this deployment (its image identifier, release pin identifier, resolved served-model identity and precision identifier) and shall never reproduce a Speech NIM model-registry identity, a registry-scheme provenance reference or a NIM profile hash.
- [x] **ING-SHIM-012**: The metadata `provenance` object shall be built from exactly the four typed fields image, pin, precision policy and resampler, `modelInfo` shall hold one entry whose `shortName` is the served-model identity, and `repository_override`, `assetInfo` and `licenseInfo` shall be empty.
- [x] **ING-SHIM-013**: Under `--api-key`, a request without the bearer token shall receive 401 from the four operational routes and the Realtime bootstrap POST and shall be refused on the claimed `/v1/realtime` WebSocket, while core's `/health` shall be exempt from the token check and return core's own health result (200 on a healthy, ready server).
- [x] **ING-SHIM-014**: A claimed Riva Realtime session shall be counted by core's WebSocket connection gauge while it is open.

## Coexistence (SHIM)

- [x] **ING-SHIM-008**: Core's existing `/health`, `/v1/models`, `/version` and metrics endpoints shall remain authoritative and not be replaced or shadowed by the plugin, the plugin shall add no `GET /v1/health` or `GET /v1/metrics`, and if core or an earlier plugin already serves a path the plugin adds as new, `attach_router` shall fail startup rather than depend on registration order.
- [x] **ING-SHIM-015**: A CPU test shall verify that the plugin's four operational paths and the bootstrap path collide with no route in core's table.

## Evaluation (CLI)

- [x] **ING-CLI-004**: The Riva gRPC and Riva Realtime WebSocket surfaces shall be driven in evaluation by the canonical `nvidia-riva/python-clients` tools, the file route by an OpenAI-style multipart transcription request, and the frontend shall implement no bespoke Riva-family client.
