# Deployment

This guide serves the Riva frontend and the streaming model plugin from one `vllm serve` process.
The frontend is a vLLM endpoint plugin (`riva_frontend`); the model plugin
(`pdtgct/vllm-nemotron-asr-streaming-plugin`, entry point `nemotron_asr_streaming`) publishes the
session factory the frontend's streams run on. Both are installed into the environment of a vLLM
build that carries RFC 2 (endpoint plugins); the frontend needs nothing else of core. The model
plugin's own requirements of vLLM are in its documentation.

## Serve

<!-- @spec ING-EP-019 -->

```bash
export VLLM_PLUGINS=riva_frontend,nemotron_asr_streaming

vllm serve nvidia/nemotron-3.5-asr-streaming-0.6b \
    --dtype float32 \
    --enforce-eager \
    --no-async-scheduling \
    --no-enable-prefix-caching \
    --config-format nemotron_asr_streaming \
    --tokenizer-mode nemotron_asr_streaming \
    --max-model-len 76800 \
    --shutdown-timeout 60 \
    --additional-config '{"riva_frontend": {"grpc_bind": "0.0.0.0:50051", "deployment_image": "registry.example/riva-frontend:0.3.0b1", "pin": "unspecified"}}' \
    --hf-overrides '{"supported_num_lookahead_tokens": [6, 13], "num_lookahead_tokens": 13}'
```

One process serves:

- on the HTTP port, core's own routes, unchanged (`/v1/realtime`, `POST /v1/audio/transcriptions`,
  `/health`, `/ready`), and the frontend's routes, all under `/plugins/riva/v1/`: the Riva Realtime
  WebSocket (`/plugins/riva/v1/realtime?intent=transcription`), its bootstrap
  (`POST /plugins/riva/v1/realtime/transcription_sessions`), complete-file transcription
  (`POST /plugins/riva/v1/audio/transcriptions`) and the operational routes
  (`/plugins/riva/v1/health/ready`, `/plugins/riva/v1/health/live`, `/plugins/riva/v1/version`,
  `/plugins/riva/v1/metadata`);
- Riva gRPC (`StreamingRecognize`, `Recognize`, `GetRivaSpeechRecognitionConfig`, health and
  reflection) on its own listener at `grpc_bind`.

## Settings

**`VLLM_PLUGINS`.** Core loads no endpoint plugin when the variable is unset, and it allowlists by
entry-point name in every plugin group, so both the frontend and the model plugin are named.

**Frontend options.** The frontend reads its options from the `riva_frontend` object of the
additional configuration (the JSON object in the command above). Omitted options take their
defaults; an unknown option, an explicit `null` or a value out of range stops startup with an
error naming every failing option. `deployment_image` and `pin` identify this deployment on
`GET /plugins/riva/v1/metadata`; they default to `unspecified`. `pin` may be a commit SHA.

**Engine stream deadlines.** The engine ends an admitted stream that is never fed, parked idle or
draining past its deadlines, set by the `stream_admission` object of the same additional
configuration (defaults: `never_fed_timeout` 15 s, `parked_idle_timeout` 60 s,
`draining_timeout` 40 s). The frontend's own bounds must stay below them, and startup fails
otherwise; when these defaults change, pass `stream_admission` beside `riva_frontend` in the same
JSON object.

**Shutdown.** The shutdown timeout is the frontend's whole drain budget: open streams are
finalized, files settle and the gRPC listener stops before the engine shuts down, with time kept
for core. It must exceed the engine's `draining_timeout`; 60 s is the documented value. A zero
timeout means no drain runs, and the frontend logs a warning at startup.

**Shutdown signal.** Send `SIGTERM` or `SIGINT` to the `vllm serve` API-server process only; in a
terminal, use `kill -INT <api-server-pid>`. A signal sent to the whole process group, such as
Ctrl-C, also reaches the engine process, which then shuts down without waiting for the frontend's
drain: the drain is cancelled and open streams end with an error and no final result.

Kubernetes and `docker stop` signal only the container's main process, so start `vllm serve` as
that process (exec form); a shell wrapper that runs it without `exec` may not forward the signal.
Both platforms force-kill after a grace period (by default 30 s for Kubernetes'
`terminationGracePeriodSeconds` and 10 s for `docker stop --time`), which is below the shutdown
timeout above. Set the grace period above `--shutdown-timeout`, with room for shutdown overhead
and any pre-stop hook, or the drain is killed.

## Operations

- The kill grace (for Kubernetes, `terminationGracePeriodSeconds`) is above `--shutdown-timeout`,
  with room for the plugin's `stop` and the engine's shutdown, so a kill does not cut the drain and
  its settlement short.
- At a zero shutdown budget (core's default `--shutdown-timeout 0`) core calls no `drain` and stops
  the engine before it calls `stop`, so the kill grace is the only bound, besides a repeated
  signal, on a `stop` that waits on a stopped engine.
- The serve mode that runs core's Rust frontend is unsupported: it starts a Rust frontend process
  instead of Python API-server processes, so no endpoint plugin loads and the frontend's startup
  checks never run.
- Orchestrators probe core's `/ready` for readiness. It needs no token, reports engine health,
  every plugin's readiness and `state.draining`, and answers 503 once core begins draining.
- A liveness probe that cannot supply the bearer token probes core's `/health`, which stays
  unauthenticated. Under `--api-key`, tooling that probes the frontend's operational routes under
  `/plugins/riva/v1/` supplies the token.

**One API server, one engine.** Every API-server process would bind the gRPC port, so one API
server process is required (`--api-server-count` at most 1; headless mode runs none and is
refused). Data parallel size must be 1 and a larger value is refused at startup: the model needs
parallel sizes of 1, and slot count and active limit are read per engine.

## Model plugin settings

The model plugin's own serve flags are the ones above (float32, eager, synchronous scheduling, no
prefix caching, its config format and tokenizer mode, and the lookahead overrides).

- `supported_num_lookahead_tokens` is `[6, 13]`: 6 is the 560 ms arm the Riva streaming surfaces
  open at, and 13 fixes the 1120 ms cadence at which complete files and core's own `/v1/realtime`
  run.
- `num_lookahead_tokens` is 13 and must be in the served set.
- `--max-model-len` must cover the longest file `unary_max_decoded_duration_seconds` admits at the
  cadence `num_lookahead_tokens` fixes. With lookahead `r`, `Q = (r + 1) * 10 + 1` tokens per
  chunk and `C` whole chunks of `(r + 1) * 1280` samples at 16 kHz, a file needs
  `1 + (C + 1) * Q + 1` tokens. At `r = 13` a 600 s file needs 75,578, and the model plugin serves
  76,800. Running files at 560 ms (`r = 6`) needs 76,114 for the same file. The frontend does not
  compute this budget; the model refuses a file that exceeds it.

## Complete files

Complete files need the model's `transcription` task; without it, `Recognize` answers
`UNIMPLEMENTED` and the frontend's file route is absent. The file gate defaults to a quarter of the
smaller of the state-slot count and the active-request limit, between 1 and 4; on a server where
that smaller value is below 2, complete files are disabled with a startup warning unless
`max_concurrent_files` is set. The multipart body bound (`http_multipart_envelope_max_bytes`)
must be at least core's upload limit (`VLLM_MAX_AUDIO_CLIP_FILESIZE_MB`) plus 262,144 bytes, and
`unary_max_decoded_duration_seconds` at most core's `VLLM_MAX_AUDIO_DECODE_DURATION_S`.

## Authentication and exposure

Under `--api-key`, the frontend's path-scoped middleware applies core's token check to every HTTP
request and WebSocket handshake under `/plugins/riva/`, including the operational routes and
paths that match no route; no other path is touched. Core's `/health` and `/ready` stay
unauthenticated.

- The stock Riva Realtime client is run with `--server host:port/plugins/riva`: it appends
  `/v1/...` to that argument, so it reaches the bootstrap and the WebSocket under the namespace.
- The stock Realtime client sends no `Authorization` header. Under `--api-key`, a deployment that
  serves it puts an intermediary in front that adds the bearer token to the bootstrap request and
  the WebSocket handshake.
- To evaluate the stock Realtime client, run it against a server with authentication disabled:
  neither `--api-key` nor `VLLM_API_KEY` set.
- Core's `POST /v1/audio/transcriptions` is outside the frontend's ingress cap, file gate and
  guard, so it is not to be exposed to untrusted clients; they reach complete files through the
  frontend's file route or `Recognize`.
- The gRPC listener is outside `--api-key` and `VLLM_API_KEY`. It is not to be exposed to
  untrusted clients without an authenticating intermediary in front of it, or secured at the
  network layer.
