# Quickstart

This walks through serving a model with the Riva/Speech-NIM compatibility
surfaces enabled, then exercising each surface once: a streaming gRPC
client, and an HTTP `curl` transcription request. It assumes `vllm-omni`
and `vllm-riva-frontend` are already installed in the same environment (see
the top-level [README](../README.md) for installation).

## Serve the model

```bash
vllm-omni serve <model> --served-model-name <name> \
  --omni --port 8000 --application-plugin riva_frontend
```

- `<model>` is the checkpoint path or Hugging Face id you would normally
  pass to `vllm-omni serve`.
- `<name>` is the identity every compatibility surface advertises and
  requires from clients (see `docs/surfaces.md`'s identity rule).
- `--application-plugin riva_frontend` selects this plugin. Without it, the
  package being installed has no effect on serving.

No `--application-plugin-config` is needed: every configuration field
resolves to a qualified default profile when omitted (see
`docs/configuration.md`). With that profile, the Riva gRPC listener binds
`0.0.0.0:50051`, and the Riva Realtime WebSocket and Speech NIM HTTP
surfaces are mounted on the same host HTTP port passed to `--port` (`8000`
above).

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

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@sample.wav" \
  -F "model=<name>"
```

`<name>` must be the exact `--served-model-name` value; omit `model` and
pass `language=<locale>` instead if you prefer to select by locale. The
response is `{"text": "<transcript>"}` unless the multipart request also
sets `response_format=text`, in which case the body is the transcript as
plain text.

## Riva Realtime WebSocket

The Realtime WebSocket surface serves the Speech NIM realtime dialect on
the same host HTTP port as the transcription endpoint above, at
`/v1/realtime?intent=transcription` (session bootstrap at
`POST /v1/realtime/transcription_sessions`) — see `docs/surfaces.md` for
its event protocol.
