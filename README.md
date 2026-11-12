# vLLM Riva frontend

`vllm-riva-frontend` is a [vLLM](https://github.com/vllm-project/vllm) endpoint plugin that lets
clients written for NVIDIA Riva and Speech NIM use an ASR model served by `vllm serve`, unchanged.
One `vllm serve` process, with this plugin and a streaming model plugin installed, serves:

- **Riva gRPC**: `nvidia.riva.asr.v1.RivaSpeechRecognition` (`StreamingRecognize`, `Recognize`,
  `GetRivaSpeechRecognitionConfig`), with gRPC health and reflection, on its own listener;
  compatible with the public [`nvidia-riva-client`](https://github.com/nvidia-riva/python-clients)
  and any client generated from the public Riva protos;
- **Riva Realtime WebSocket** at `/plugins/riva/v1/realtime?intent=transcription`, with the session
  bootstrap at `POST /plugins/riva/v1/realtime/transcription_sessions`, on the HTTP port; the stock
  Riva Realtime client reaches it with `--server host:port/plugins/riva`;
- **complete files** at `POST /plugins/riva/v1/audio/transcriptions`: the transcription request and
  response of core's route, behind the frontend's ingress cap, file gate, request guard and bounded
  in-memory multipart parser;
- the NIM **operational routes** `/plugins/riva/v1/health/ready`, `/plugins/riva/v1/health/live`,
  `/plugins/riva/v1/version` and `/plugins/riva/v1/metadata`.

Every HTTP and WebSocket route is under `/plugins/riva/v1/`; core's routes, including `/v1/realtime`
and `POST /v1/audio/transcriptions`, are unchanged.

Streams run on the session factory the model plugin publishes; complete files run on core's
transcription service. The plugin creates no second engine and adds no code to vLLM. It does not
define transport security: under core's `--api-key`, the frontend applies core's token check to
every HTTP and WebSocket request under `/plugins/riva/`, and the gRPC listener is secured at the
network layer.

## Requirements

- A vLLM build carrying RFC 2 (endpoint plugins). The package declares no dependency on vLLM: the
  host environment supplies it, and startup checks the capabilities it needs.
- A model plugin that publishes a session factory in the `vllm_riva_frontend.session_factories`
  entry-point group, such as `pdtgct/vllm-nemotron-asr-streaming-plugin`.
- A Python version vLLM supports (below).

This repository is a development preview and is not published to PyPI. Install it from git:

```bash
pip install "git+https://github.com/pdtgct/vllm-riva-frontend.git"
```

## Supported Python versions

The plugin is imported into vLLM's interpreter, so it declares vLLM's range. A version is
`supported` when CI runs the test suite on it; the others in the range are declared but not yet
verified.

<!-- @spec ING-EP-022 -->

| Python | Status |
| --- | --- |
| 3.11 | supported |
| 3.12 | supported |
| 3.13 | declared-but-unverified |
| 3.14 | declared-but-unverified |

## Serve

Install the frontend and the model plugin into the vLLM environment, allowlist both and pass the
frontend's options through `--additional-config` under the key `riva_frontend`. The deployment
guide, [`docs/deployment.md`](docs/deployment.md), has the full command, the settings it depends
on and their reasons.

Clients select the served model by its primary identity: the first name `--served-model-name`
gives (or the model path when it is unset). Riva gRPC and the Realtime WebSocket accept only that
name; the file route validates `model` as core's route does.

## Compatibility scope

The streaming and `Recognize` surfaces accept mono PCM16 at 8 or 16 kHz and mono G.711 μ-law and
A-law at 8 kHz; 8 kHz input is resampled to 16 kHz with a pinned resampler whose identifier
`/plugins/riva/v1/metadata` reports. Other formats, translation, diarization, word boosting, word offsets,
endpointing and multi-utterance session reuse are refused explicitly, naming the field.

`enable_automatic_punctuation` and `verbatim_transcripts` (`enable_verbatim_transcripts` in the
Realtime dialect) are accepted under either value and do not change the output: punctuation and
normalization follow the checkpoint. Complete files decode greedily, so the file route refuses
beam search, completion limits and non-default sampling fields.

Every failure the frontend raises carries a stable error code: in the gRPC status details, in the
Realtime `error` event, and in core's HTTP error shape on the file route. Some refusals carry no
code and answer as the framework or core does: an HTTP request under `/plugins/riva/` while
authentication is still initializing (503, `{"error": "Service Unavailable"}`) or refused by it
(401, `{"error": "Unauthorized"}`); a WebSocket handshake under `/plugins/riva/` while
authentication is still initializing or refused by it, which is never accepted: it gets that 503
or 401 response through the ASGI WebSocket denial-response extension when the server offers it,
and otherwise a close before accept, which the server answers with HTTP 403 and no body; a
Realtime connection whose query is not exactly `intent=transcription` (refused before accept,
HTTP 403, no body); an unknown path under `/plugins/riva/` (404, or a WebSocket closed before
accept); a gRPC message above the receive cap (gRPC's native `RESOURCE_EXHAUSTED`); the status the
listener stop gives an RPC that has not sent its own; and core's own errors on the file route, in
core's shape unchanged.

## Documentation

- [`docs/deployment.md`](docs/deployment.md): the serve command and its settings.
- [`docs/high-level-design.md`](docs/high-level-design.md) and the leaf designs under
  [`docs/intent/`](docs/intent): the normative design and specifications.

## License and development

Licensed under the [Apache License 2.0](LICENSE).

Install the locked development environment with `uv sync --group dev`. The tests need a vLLM tree
carrying RFC 2 on `PYTHONPATH`. The default `pytest` command runs the complete suite and enforces
package line coverage above 90%; during a focused edit, use `pytest --no-cov PATH_OR_NODE_ID`.
