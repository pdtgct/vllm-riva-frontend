# Deployment

This guide serves the Riva frontend and the streaming model plugin from one `vllm serve` process.
The frontend is a vLLM endpoint plugin (`riva_frontend`); the model plugin
(`pdtgct/vllm-nemotron-asr-streaming-plugin`, entry point `nemotron_asr_streaming`) publishes the
session factory the frontend's streams run on. Both are installed into the environment of a vLLM
build that carries RFC 1 (persistent state and stream admission) and RFC 2 (endpoint plugins).

## Serve

<!-- @spec ING-EP-019, ING-EP-058 -->

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
    --additional-config '{"riva_frontend": {"grpc_bind": "0.0.0.0:50051", "deployment_image": "registry.example/riva-frontend:0.2.0", "pin": "unspecified"}}' \
    --hf-overrides '{"supported_num_lookahead_tokens": [6, 13], "num_lookahead_tokens": 13}'
```

One process serves:

- core's `/v1/realtime` and, with the frontend loaded, the Riva Realtime WebSocket
  (`/v1/realtime?intent=transcription`), its bootstrap (`POST /v1/realtime/transcription_sessions`),
  complete-file transcription (`POST /v1/audio/transcriptions`) and the operational routes
  (`/v1/health/ready`, `/v1/health/live`, `/v1/version`, `/v1/metadata`) on the HTTP port;
- Riva gRPC (`StreamingRecognize`, `Recognize`, `GetRivaSpeechRecognitionConfig`, health and
  reflection) on `grpc_bind`.

## Settings

**`VLLM_PLUGINS`.** Core loads no endpoint plugin when the variable is unset, and it allowlists by
entry-point name in every plugin group, so both the frontend and the model plugin are named.

**Frontend options.** The frontend reads its options from the `riva_frontend` object of the
additional configuration (the JSON object in the command above). Omitted options take their
defaults; an unknown option, an explicit `null` or a value out of range stops startup with an
error naming every failing option. `deployment_image` and `pin` identify this deployment on
`GET /v1/metadata`; they default to `unspecified`. `pin` may be a commit SHA.

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

**One API server, one engine.** The frontend owns a non-HTTP listener, so core requires a single
API-server process. Data parallel size above 1 is refused at startup: the model needs parallel
sizes of 1, and slot count and active limit are read per engine.

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
`UNIMPLEMENTED` and the file route is core's (absent). The file gate defaults to a quarter of the
smaller of the state-slot count and the active-request limit, between 1 and 4; on a server where
that smaller value is below 2, complete files are disabled with a startup warning unless
`max_concurrent_files` is set. The multipart body bound (`http_multipart_envelope_max_bytes`)
must be at least core's upload limit (`VLLM_MAX_AUDIO_CLIP_FILESIZE_MB`) plus 262,144 bytes, and
`unary_max_decoded_duration_seconds` at most core's `VLLM_MAX_AUDIO_DECODE_DURATION_S`.

## Authentication

Under `--api-key`, core's authentication covers the frontend's HTTP routes and the claimed
WebSocket, including the operational routes; core's `/health` stays unauthenticated. The gRPC
listener is not covered: secure it at the network layer.
