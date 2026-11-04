# Quickstart

This walks through serving a model with the Riva compatibility surfaces
enabled, then exercising streaming gRPC. It assumes vLLM core, the model
plugin, and `vllm-riva-frontend` are installed in the same environment
(see the top-level [README](../README.md)).

## Serve the model

```bash
export VLLM_PLUGINS=nemotron_asr,riva_frontend
vllm serve <model> --served-model-name <name> \
  --port 8000 --api-server-count 1
```

- `<model>` is the checkpoint path or Hugging Face id you would pass to
  `vllm serve`.
- `<name>` is the identity every compatibility surface advertises and
  requires from clients (see `docs/surfaces.md`'s identity rule).
- `VLLM_PLUGINS` must name **both** the model plugin (`nemotron_asr` in
  this example — use the name that package registers) and `riva_frontend`.
  Installing the package alone does not load the endpoint plugin.
- `--api-server-count 1` is required because this plugin owns the gRPC
  listener.

No config file is needed: every field resolves to the qualified default
profile (see `docs/configuration.md`). With that profile, the Riva gRPC
listener binds `0.0.0.0:50051`. The Realtime WebSocket is on the HTTP port
passed to `--port`. Offline `Recognize` and `POST /v1/audio/transcriptions`
stay with core unless `serve_offline` is true.

## Streaming recognition with `riva.client`

The canonical Riva Python client is `riva.client`, distributed on PyPI as
[`nvidia-riva-client`](https://pypi.org/project/nvidia-riva-client/) (also
published as source at
[`nvidia-riva/python-clients`](https://github.com/nvidia-riva/python-clients)).
Install it, then drive `StreamingRecognize` against the gRPC bind above:

```bash
pip install nvidia-riva-client
```

```python
import riva.client

auth = riva.client.Auth(uri="localhost:50051")
asr_service = riva.client.ASRService(auth)

config = riva.client.RecognitionConfig(
    encoding=riva.client.AudioEncoding.LINEAR_PCM,
    sample_rate_hertz=16000,
    language_code="auto",
)
streaming_config = riva.client.StreamingRecognitionConfig(
    config=config, interim_results=True
)


def audio_chunks(path, chunk_size=6400):
    import wave

    with wave.open(path, "rb") as wav_file:
        data = wav_file.readframes(wav_file.getnframes())
    for start in range(0, len(data), chunk_size):
        yield data[start : start + chunk_size]


responses = asr_service.streaming_response_generator(
    audio_chunks=audio_chunks("sample.wav"),
    streaming_config=streaming_config,
)
for response in responses:
    for result in response.results:
        for alternative in result.alternatives:
            print(result.is_final, alternative.transcript)
```

`language_code="auto"` selects the model's default locale. Pass one of the
model's supported locales instead to request it explicitly; an
unrecognized locale is rejected (`unknown_locale`, see
`docs/surfaces.md`).

## Speech NIM HTTP transcription with `curl`

By default this plugin does not own `POST /v1/audio/transcriptions`, so
the request is core's transcription route. Set `serve_offline` to true
when this plugin should answer it instead:

```bash
export VLLM_RIVA_FRONTEND_CONFIG='{"serve_offline": true}'
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@sample.wav" \
  -F "model=<name>"
```

`<name>` must be the exact `--served-model-name` value; omit `model` and
pass `language=<locale>` instead if you prefer to select by locale. When
this plugin serves the route, the response is `{"text": "<transcript>"}`
unless the multipart request also sets `response_format=text`, in which
case the body is the transcript as plain text.

## Riva Realtime WebSocket

The Realtime WebSocket surface serves the Speech NIM realtime dialect on
the same host HTTP port as the transcription endpoint above, at
`/v1/realtime?intent=transcription` (session bootstrap at
`POST /v1/realtime/transcription_sessions`) — see `docs/surfaces.md` for
its event protocol.
