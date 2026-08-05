# vLLM Riva frontend

`vllm-riva-frontend` is an [application plugin](https://github.com/vllm-project/vllm-omni)
for vLLM-Omni that turns a single vLLM-Omni server into an NVIDIA
Riva/Speech-NIM-compatible speech endpoint, on the same engine and the same
process as native vLLM-Omni serving. It co-hosts three compatibility
surfaces alongside the native `/v1/realtime` API, sharing the same engine,
session factory, and provider pool:

- **Riva gRPC** — `nvidia.riva.asr.v1.RivaSpeechRecognition`
  (`StreamingRecognize`, `Recognize`, gRPC health, and reflection) on its
  own bind, compatible with the public
  [`nvidia-riva-client`](https://github.com/nvidia-riva/python-clients)
  Python client and any other Riva ASR gRPC client generated from the
  public Riva protos;
- **Riva Realtime WebSocket** at `/v1/realtime?intent=transcription`, with
  session bootstrap at `POST /v1/realtime/transcription_sessions`, on the
  host's own HTTP port; and
- **Speech NIM HTTP transcription** at `POST /v1/audio/transcriptions`, also
  on the host's own HTTP port.

It does not replace vLLM-Omni's native `/v1/realtime` route, does not mount
`/v1/audio/translations`, does not create a second inference engine, and
does not define transport security policy: TLS, authentication, network
policy, and proxy/sidecar termination remain deployment concerns, owned by
however you run vLLM-Omni.

## Requirements

- A vLLM-Omni host on the 0.24.x or 0.25.x release line. The plugin checks
  the installed `vllm-omni` version at startup and refuses to start outside
  those two lines (an editable/dev/pre-release checkout is allowed through,
  since its base version is not a reliable line indicator).
- Python 3.10 through 3.13, matching whatever interpreter your vLLM-Omni
  host already runs.
- A model served by vLLM-Omni whose engine exposes the Nemotron ASR
  streaming-session capability this plugin adapts.

This repository is a development preview and is not yet published to PyPI.
Until it is, install it from git:

```bash
pip install "git+https://github.com/pdtgct/vllm-riva-frontend.git"
```

## Quick start

1. Install the package (see Requirements, above) into the same environment
   as your vLLM-Omni host.
2. Serve a supported model on vLLM-Omni and select this plugin explicitly —
   installing the package alone has no serving effect:

   ```bash
   vllm-omni serve MODEL --served-model-name PUBLIC_MODEL_NAME \
     --omni --port 8000 --application-plugin riva_frontend
   ```

3. Point a canonical Riva or Speech NIM client at the running server — for
   example the public `riva.client` Python package (`nvidia-riva-client` on
   PyPI) against `localhost:50051`, or a `curl` request against
   `http://localhost:8000/v1/audio/transcriptions`.

`PUBLIC_MODEL_NAME` is the frontend's primary served-model identity: it is
advertised and validated consistently across Riva gRPC, Riva Realtime, and
Speech NIM HTTP, and a client must use it — the canonical checkpoint path or
Hugging Face id is rejected on every compatibility surface (see
`docs/surfaces.md`). If `--served-model-name` is omitted, the canonical
`MODEL` value is used instead.

No plugin configuration is required for this quick start: every field
resolves to a qualified zero-config default profile. See
`docs/configuration.md` to override any of it, and `docs/quickstart.md` for
a worked end-to-end example including a streaming gRPC client and an HTTP
`curl` example.

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
rendered pod runs one vLLM-Omni API-server worker and one engine.

## License and development

Licensed under the [Apache License 2.0](LICENSE).

Install the locked development environment with `uv sync --group dev`. The
default `pytest` command runs the complete suite and enforces package line
coverage above 90%. During a focused edit, use `pytest --no-cov PATH_OR_NODE_ID`
so an otherwise-passing single test is not reported as a package-wide
coverage failure; run unfiltered `pytest` before committing.
