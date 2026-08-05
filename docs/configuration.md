# Configuration

## Supplying configuration

Configuration is optional. Omitting `--application-plugin-config` entirely
resolves every field below to its default value — the "qualified profile"
this project deploys with. You only need to pass configuration to override
specific fields.

When you do supply configuration, pass it to vLLM-Omni's
`--application-plugin-config` flag under this plugin's entry-point key,
`riva_frontend`, as either inline JSON or a path to a JSON file:

```bash
# Inline JSON
vllm-omni serve <model> --application-plugin riva_frontend \
  --application-plugin-config riva_frontend='{"grpc_bind": ":50052"}'

# Path to a JSON file
vllm-omni serve <model> --application-plugin riva_frontend \
  --application-plugin-config riva_frontend=/path/to/riva_frontend.json
```

A value is treated as inline JSON, rather than a path, the moment its
(whitespace-trimmed) text starts with `{`. That means malformed inline JSON
always fails as malformed JSON — it is never silently reinterpreted as a
(almost certainly nonexistent) file path.

## Validation rules

- **Omission means default.** Any field you don't supply resolves to the
  default in the table below.
- **An explicit value is always validated**, whether it came from you or
  from the default profile — a default is a value, not an exemption. Every
  numeric field must be finite and positive; several fields have additional
  cross-field relationships (for example, `grpc_receive_max_bytes` must
  equal `unary_max_encoded_audio_bytes + grpc_config_envelope_max_bytes`,
  and `plugin_shutdown_grace` must cover
  `session_finalization_timeout + session_cleanup_timeout`).
- **Explicit `null` always fails startup**, naming the field: no field in
  this configuration accepts JSON `null` as a way to request "no limit" or
  "use the default." (`grpc_keepalive_seconds` accepts *omission* to mean
  "keepalive disabled" — it just cannot be set to `null` explicitly.)
- **Unknown fields fail startup**, listing every unrecognized key.
- Every startup validation failure names this plugin's configuration key
  (`riva_frontend`) so a multi-plugin host's log is unambiguous about which
  plugin failed.

## Field reference

All 29 fields below can be set through `--application-plugin-config
riva_frontend=...`. The first 26 govern the plugin's own listeners,
transports, and session lifetime; the last 3 (`deployment_image`, `pin`,
`precision_policy`) are deployment-owned provenance facts (see below).

| Field | Default | Meaning |
| --- | --- | --- |
| `grpc_bind` | `"0.0.0.0:50051"` | `host:port` the Riva gRPC listener binds to. |
| `grpc_receive_max_bytes` | `33619968` | Maximum gRPC message size (bytes) the server accepts; must equal `unary_max_encoded_audio_bytes + grpc_config_envelope_max_bytes`. |
| `grpc_config_envelope_max_bytes` | `65536` | Bytes reserved in the gRPC receive budget for the non-audio (streaming/recognition config) portion of a request. |
| `unary_max_encoded_audio_bytes` | `33554432` | Maximum encoded audio bytes accepted by unary `Recognize` and HTTP transcription requests. |
| `unary_max_decoded_duration_seconds` | `600.0` | Maximum decoded audio duration, in seconds, accepted per unary/HTTP request. |
| `max_riff_header_bytes` | `1048576` | Maximum bytes buffered while sniffing a RIFF/WAV header before its audio format is known. |
| `load_shed_max_sessions` | `64` | Maximum concurrent counted inference sessions (gRPC streaming/unary, Realtime WebSocket, HTTP transcription) admitted before new work is rejected as busy. |
| `pre_submit_max_samples` | `65536` | Maximum normalized audio samples outstanding — submitted to the engine but not yet accepted — per session before a buffer-overflow rejection. |
| `preconfiguration_timeout` | `30.0` | Seconds allowed for a session to complete its configuration/open handshake before a configuration timeout. |
| `session_idle_timeout` | `60.0` | Seconds with no accepted audio before an open session is aborted as idle. |
| `session_finalization_timeout` | `180.0` | Seconds allowed for a session's terminal flush/finish to complete before a finalization timeout. |
| `session_cleanup_timeout` | `30.0` | Seconds allowed for abnormal-path lease abort-and-release cleanup before it is reported as a cleanup fault. |
| `plugin_shutdown_grace` | `240.0` | Seconds the plugin waits for in-flight owners to drain gracefully during shutdown; must cover `session_finalization_timeout + session_cleanup_timeout`. |
| `ws_receive_max_bytes` | `16777216` | Maximum bytes accepted per WebSocket receive; must cover `ws_event_envelope_max_bytes`. Corresponds to vLLM-Omni's own `--ws-max-size` host flag (whose own default, 16 MiB, already matches). |
| `ws_event_envelope_max_bytes` | `8388608` | Maximum size, in bytes, of one Realtime WebSocket JSON event. |
| `http_multipart_envelope_max_bytes` | `33816576` | Maximum total bytes of one multipart HTTP transcription request body; must exceed `unary_max_encoded_audio_bytes` to leave room for multipart framing. |
| `http_content_type_max_bytes` | `4096` | Maximum bytes of the HTTP `Content-Type` header value. |
| `http_request_header_max_bytes` | `65536` | Maximum bytes reserved for HTTP request headers; must be at least `http_content_type_max_bytes`. Deployments that raise or lower this should keep vLLM-Omni's own `--h11-max-incomplete-event-size` host flag consistent with it. |
| `http_multipart_boundary_max_bytes` | `200` | Maximum bytes of the multipart boundary token. |
| `http_multipart_max_parts` | `5` | Maximum number of multipart form parts accepted per request. |
| `http_multipart_max_header_bytes` | `8192` | Maximum bytes of headers within one multipart part. |
| `http_text_field_max_bytes` | `4096` | Maximum bytes of one non-file multipart text field. |
| `http_request_timeout` | `900.0` | Seconds allowed for one HTTP transcription request end to end before a request timeout. |
| `grpc_keepalive_seconds` | `null` (omitted) | gRPC keepalive ping interval, in seconds. Omitted disables keepalive; if set, must be finite and positive. |
| `max_session_duration` | `600.0` | Maximum total seconds a session may stay open, from configuration to finalization, regardless of activity. |
| `resampler_identifier` | `"scipy-poly-v1"` | Identifier of the pinned 8 kHz resampler; must equal the package's built-in identifier — this is a fixed fact, not a real choice. |
| `deployment_image` | `"unspecified"` | Deployment-owned container image identity published in `/v1/metadata` provenance. |
| `pin` | `"unspecified"` | Deployment-owned vLLM/vLLM-Omni version pin published in `/v1/metadata` provenance. |
| `precision_policy` | `"fp32-bringup-v1"` | Deployment-owned precision-policy identifier published in `/v1/metadata` provenance. |

## Provenance defaults are deliberate

`deployment_image` and `pin` default to the literal string `"unspecified"`.
This is intentional, not a placeholder to fix later: provenance is a
per-deployment fact, and a zero-config deployment must never advertise
another deployment's image or version pin through `/v1/metadata`.
Deployments that want `/v1/metadata` to report real image/pin identity set
`deployment_image` and `pin` explicitly, alongside `precision_policy` if it
differs from the package default.

All four provenance-bound values (`deployment_image`, `pin`,
`precision_policy`, and `resampler_identifier`) are rejected at startup if
they carry leading/trailing whitespace, a `scheme://` reference, or look
like a bare hex identifier — shapes associated with NIM registry/profile
identity rather than a deployment-authored fact.
