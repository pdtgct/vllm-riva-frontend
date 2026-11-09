# vLLM Riva frontend

`vllm-riva-frontend` is a [vLLM](https://github.com/vllm-project/vllm) endpoint plugin that lets
clients written for NVIDIA Riva and Speech NIM use an ASR model served by `vllm serve`, unchanged.
One `vllm serve` process, with this plugin and a streaming model plugin installed, serves:

- **Riva gRPC**: `nvidia.riva.asr.v1.RivaSpeechRecognition` (`StreamingRecognize`, `Recognize`,
  `GetRivaSpeechRecognitionConfig`), with gRPC health and reflection, on its own listener;
  compatible with the public [`nvidia-riva-client`](https://github.com/nvidia-riva/python-clients)
  and any client generated from the public Riva protos;
- **Riva Realtime WebSocket** at `/v1/realtime?intent=transcription`, with the session bootstrap at
  `POST /v1/realtime/transcription_sessions`, on the HTTP port; every other `/v1/realtime` scope
  stays core's;
- **complete files** at `POST /v1/audio/transcriptions`: core's transcription route and response,
  behind the frontend's file gate, request guard and bounded in-memory multipart parser;
- the NIM **operational routes** `/v1/health/ready`, `/v1/health/live`, `/v1/version` and
  `/v1/metadata`.

Streams run on the session factory the model plugin publishes; complete files run on core's
transcription service. The plugin creates no second engine and adds no code to vLLM. It does not
define transport security: core's `--api-key` covers its HTTP routes, and the gRPC listener is
secured at the network layer.

## Requirements

- A vLLM build carrying RFC 1 (persistent state and stream admission) and RFC 2 (endpoint plugins);
  the current target is the `rfc12-integration-v2` branch of `pdtgct/vllm`. The package declares
  no dependency on vLLM: the host environment supplies it, and startup checks the capabilities it
  needs.
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
`/v1/metadata` reports. Other formats, translation, diarization, word boosting, word offsets,
endpointing and multi-utterance session reuse are refused explicitly, naming the field.

`enable_automatic_punctuation` and `verbatim_transcripts` (`enable_verbatim_transcripts` in the
Realtime dialect) are accepted under either value and do not change the output: punctuation and
normalization follow the checkpoint. Complete files decode greedily, so the file route refuses
beam search, completion limits and non-default sampling fields.

Every failure carries a stable error code: in the gRPC status details, in the Realtime `error`
event, and in core's HTTP error shape on the file route.

## Documentation

- [`docs/deployment.md`](docs/deployment.md): the serve command and its settings.
- [`docs/high-level-design.md`](docs/high-level-design.md) and the leaf designs under
  [`docs/intent/`](docs/intent): the normative design and specifications.

## License and development

Licensed under the [Apache License 2.0](LICENSE).

Install the locked development environment with `uv sync --group dev`. The tests need the RFC 1 +
RFC 2 vLLM tree on `PYTHONPATH`. The default `pytest` command runs the complete suite and enforces
package line coverage above 90%; during a focused edit, use `pytest --no-cov PATH_OR_NODE_ID`.
