# vLLM Riva frontend

`vllm-riva-frontend` is a [vLLM endpoint plugin](https://docs.vllm.ai/en/latest/design/endpoint_plugins/)
(`vllm.endpoint_plugins`) that turns one vLLM core API server into an
NVIDIA Riva/Speech-NIM-compatible speech endpoint, in the same process as
native serving. It does not import vLLM-Omni. The model plugin supplies
the core `AsyncLLM` session through the `vllm_riva.session_factories`
entry-point group; this package only discovers that factory.

It co-hosts compatibility surfaces alongside the native HTTP API, sharing
the engine and that session factory:

- **Riva gRPC** — `nvidia.riva.asr.v1.RivaSpeechRecognition`
  (`StreamingRecognize`, `Recognize`, gRPC health, and reflection) on its
  own bind, compatible with the public
  [`nvidia-riva-client`](https://github.com/nvidia-riva/python-clients)
  Python client and any other Riva ASR gRPC client generated from the
  public Riva protos;
- **Riva Realtime WebSocket** at `/v1/realtime?intent=transcription`, with
  session bootstrap at `POST /v1/realtime/transcription_sessions`, on the
  host's own HTTP port; and
- **Speech NIM HTTP transcription** at `POST /v1/audio/transcriptions` is
  **off unless** `serve_offline` is true. Left off, that route and unary
  gRPC `Recognize` stay with core instead of shadowing `/v1/audio/transcriptions`.

It does not replace core's native `/v1/realtime` route (only
`?intent=transcription` is claimed), does not mount `/v1/audio/translations`,
does not create a second inference engine, and does not define transport
security policy: TLS, authentication, network policy, and proxy/sidecar
termination remain deployment concerns.

The plugin sets `owns_non_http_listener = True` because it binds gRPC
itself. Core then requires `--api-server-count 1`. The default bind is
`0.0.0.0:50051`.

## Requirements

- vLLM core on the 0.33.x tip that includes endpoint-plugin lifecycle
  (`start` / `ready` / `drain` / `stop`, `owns_non_http_listener`). The
  package checks the installed `vllm` version at startup and accepts
  0.33.0rc1 through 0.33.x, plus 0.33 and 0.34 dev/local checkouts of that
  tip. It does not depend on `vllm` or `vllm-omni` at install time.
- Python 3.11 through 3.14, matching vLLM core.
- A model plugin that registers a session factory under
  `vllm_riva.session_factories`. That factory's `open(*, cadence, locale)`
  talks to core `AsyncLLM`. If the factory publishes `served_cadences`,
  `streaming_cadence` must be in that set.

This repository is a development preview and is not yet published to PyPI.
Until it is, install it from git:

```bash
pip install "git+https://github.com/pdtgct/vllm-riva-frontend.git"
```

## Quick start

1. Install this package and the model plugin into the same environment as
   vLLM core.
2. Allowlist **both** entry-point names. Endpoint plugins load only when
   `VLLM_PLUGINS` names them, and a set `VLLM_PLUGINS` also filters general
   plugins, so the model plugin is skipped if you list only `riva_frontend`:

   ```bash
   export VLLM_PLUGINS=nemotron_asr,riva_frontend
   vllm serve MODEL --served-model-name PUBLIC_MODEL_NAME \
     --port 8000 --api-server-count 1
   ```

   `nemotron_asr` is the model plugin's entry-point name (the name it
   registers under `vllm.general_plugins` and `vllm_riva.session_factories`).
   Substitute the name your model package actually registers.
   `riva_frontend` is this package's `vllm.endpoint_plugins` name.

3. Point a canonical Riva client at `localhost:50051` (the default
   `grpc_bind` of `0.0.0.0:50051`). Streaming gRPC and
   `/v1/realtime?intent=transcription` are on without extra config.
   `POST /v1/audio/transcriptions` stays on core unless you set
   `serve_offline` to true.

`PUBLIC_MODEL_NAME` is the frontend's primary served-model identity: it is
advertised and validated consistently across Riva gRPC, Riva Realtime, and
Speech NIM HTTP, and a client must use it — the canonical checkpoint path or
Hugging Face id is rejected on every compatibility surface (see
`docs/surfaces.md`). If `--served-model-name` is omitted, the canonical
`MODEL` value is used instead.

No plugin configuration is required for this quick start: every field
resolves to a qualified zero-config default profile, including
`grpc_bind=0.0.0.0:50051` and `streaming_cadence=560ms`. Core has no
`--application-plugin-config`. Override fields with
`VLLM_RIVA_FRONTEND_CONFIG` (inline JSON or a file path) or with
`--additional-config '{"riva_frontend": ...}'` (object, inline JSON, or a
path). When both are set, `--additional-config` wins. See
`docs/configuration.md` and `docs/quickstart.md`.

## Documentation

- [`docs/quickstart.md`](docs/quickstart.md) — the serve command, a minimal
  streaming gRPC client, and an HTTP transcription example.
- [`docs/configuration.md`](docs/configuration.md) — inline and file-based
  plugin configuration, the full field reference, and validation rules.
- [`docs/surfaces.md`](docs/surfaces.md) — each compatibility surface's
  endpoint, dialect, supported subset, and error behavior.

## Compatibility scope

The frontend accepts mono PCM16 at 8 or 16 kHz and mono G.711 μ-law/A-law at
8 kHz. Eight-kilohertz input is resampled with a pinned, provenance-stamped
resampler. General codecs, translation, diarization, word boosting, word
offsets, endpointing, and multi-utterance session reuse are outside this
version.

Two recognition booleans are intentionally model-intrinsic:
`enable_automatic_punctuation` and `verbatim_transcripts` (spelled
`enable_verbatim_transcripts` in the Realtime WebSocket dialect). Either
value is accepted, but it does not change the served checkpoint's native
punctuation or text normalization. `temperature` on the Speech NIM HTTP
endpoint is likewise accepted as model-intrinsic, because this frontend
uses deterministic greedy decoding.

The WebSocket adapter emits true incremental deltas and exactly one
terminal completion per session.

## Deployment module

The repository-owned `deploy/riva_frontend` module renders a qualified
single-worker deployment shape:

```bash
make riva-frontend-render
```

Equivalently, an installed package provides:

```bash
vllm-riva-frontend-render \
  --values deploy/riva_frontend/home.values.yaml \
  --template deploy/riva_frontend/manifest.template.yaml \
  --output deploy/riva_frontend/rendered.yaml
```

The values file is the sole owner of image, endpoints, GPU declaration,
model selectors, precision/pin provenance, and finite resource bounds. The
rendered pod runs one vLLM API-server worker (`--api-server-count 1`)
and one engine. `VLLM_PLUGINS` names the model plugin and `riva_frontend`.

## License and development

Licensed under the [Apache License 2.0](LICENSE).

Install the locked development environment with `uv sync --group dev`. The
default `pytest` command runs the complete suite and enforces package line
coverage above 90%. During a focused edit, use `pytest --no-cov PATH_OR_NODE_ID`
so an otherwise-passing single test is not reported as a package-wide
coverage failure; run unfiltered `pytest` before committing.
