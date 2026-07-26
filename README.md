# vLLM Riva frontend

`vllm-riva-frontend` is an explicitly selected downstream application plugin
for the Nemotron ASR capability in vLLM-Omni. It cohosts three compatibility
surfaces with the native `/v1/realtime` API while sharing the same engine,
session factory, and provider pool:

- Riva `RivaSpeechRecognition` gRPC (`StreamingRecognize`, `Recognize`, health,
  and reflection) on a separate deployment-owned bind;
- Riva Realtime WebSocket at
  `/v1/realtime?intent=transcription`, with session bootstrap at
  `POST /v1/realtime/transcription_sessions`; and
- Speech NIM HTTP transcription at `POST /v1/audio/transcriptions`.

This repository is a development preview. The package is not published to
PyPI, and it currently requires the matching Nemotron ASR and generic
application-plugin changes in vLLM-Omni.

The plugin does not mount `/v1/audio/translations`, replace the host health or
metrics endpoints, create a second inference engine, or define transport
security policy. TLS, authentication, encryption, bind exposure, network
policy, and proxy/sidecar termination belong to deployment configuration.

Install alone has no serving effect. Select the plugin explicitly:

```text
vllm-omni serve MODEL --omni \
  --served-model-name PUBLIC_MODEL_NAME \
  --application-plugin riva_frontend \
  --application-plugin-config riva_frontend=/path/riva_frontend.json \
  --ws-max-size WS_RECEIVE_BYTES \
  --h11-max-incomplete-event-size HTTP_HEADER_BYTES
```

`PUBLIC_MODEL_NAME` is the frontend's primary served-model identity. It is
advertised and validated consistently by Riva gRPC, Riva Realtime, Speech NIM
HTTP, and plugin metadata. If `--served-model-name` is omitted, the canonical
`MODEL` value is used. When vLLM is given multiple aliases, this frontend uses
the normalized first alias—the same identity vLLM uses in responses and metrics—
and does not expose the canonical checkpoint or secondary aliases as additional
compatibility selectors.

The two host limits must exactly match the corresponding plugin configuration.
All required byte, sample, session, part, and timeout bounds are validated
before either listener becomes ready. JSON `null`, non-finite values, unknown
fields, unsafe cross-field relationships, multiple API workers, and unsupported
vLLM-Omni major/minor versions fail startup.

At the current RFC-1 pin, the session provider does not publish a typed safe
finalization bound or maximum session duration. Consequently,
`session_finalization_timeout` and `max_session_duration` are required explicit
configuration values. They are validated here and must be qualified for the
selected deployment on hardware; the frontend does not invent provider-derived
defaults for them. The only omission defaults in v1 are the 60-second realtime
idle timeout, disabled gRPC keepalive, and the pinned resampler identifier.

## Deployment module

The repository-owned `deploy/riva_frontend` module renders the qualified
single-worker shape:

```text
make riva-frontend-render
```

Equivalently, an installed package provides:

```text
vllm-riva-frontend-render \
  --values deploy/riva_frontend/home.values.yaml \
  --template deploy/riva_frontend/manifest.template.yaml \
  --output deploy/riva_frontend/rendered.yaml
```

The values file is the sole owner of image, endpoints, GPU declaration, model
selectors, precision/pin provenance, and finite resource bounds. The rendered
pod runs one vLLM-Omni API-server worker and one engine. The HTTP and WebSocket
compatibility routes cohost with native `/v1/realtime` through the host's
inner-ASGI plugin seam, so existing vLLM authentication and request middleware
remain outside and authoritative. gRPC binds separately but uses the same
session factory and provider pool.

## Compatibility scope

The frontend accepts mono PCM16 at 8 or 16 kHz and mono G.711 μ-law/A-law at
8 kHz. Eight-kilohertz input uses the provenance-stamped
`scipy-poly-v1` resampler. General codecs, translation, diarization, word
boosting, word offsets, endpointing, and multi-utterance reuse are outside v1.

Two recognition booleans are intentionally model-intrinsic:
`enable_automatic_punctuation` and `verbatim_transcripts` (spelled
`enable_verbatim_transcripts` in the Realtime JSON dialect). Either value is
accepted, but it does not change the checkpoint's native punctuation or text
normalization. `temperature` on Speech NIM HTTP is likewise accepted as
model-intrinsic because this frontend uses deterministic greedy RNN-T decode.

The WebSocket adapter emits true incremental deltas and exactly one terminal
completion. This intentionally does not reproduce observed cumulative-delta or
interim-completion behavior from some Riva-family deployments.
