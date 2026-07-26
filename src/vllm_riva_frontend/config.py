"""Finite frontend configuration and dialect-disposition contract."""

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from vllm_riva_frontend.frontend import RESAMPLER_ID


@dataclass(frozen=True)
class FrontendConfig:
    """Values which must validate before either compatibility listener binds."""

    grpc_bind: str
    grpc_receive_max_bytes: int
    grpc_config_envelope_max_bytes: int
    unary_max_encoded_audio_bytes: int
    unary_max_decoded_duration_seconds: float
    max_riff_header_bytes: int
    load_shed_max_sessions: int
    pre_submit_max_samples: int
    preconfiguration_timeout: float
    session_idle_timeout: float
    session_finalization_timeout: float
    session_cleanup_timeout: float
    plugin_shutdown_grace: float
    ws_receive_max_bytes: int
    ws_event_envelope_max_bytes: int
    http_multipart_envelope_max_bytes: int
    http_content_type_max_bytes: int
    http_request_header_max_bytes: int
    http_multipart_boundary_max_bytes: int
    http_multipart_max_parts: int
    http_multipart_max_header_bytes: int
    http_text_field_max_bytes: int
    http_request_timeout: float
    grpc_keepalive_seconds: float | None
    max_session_duration: float | None
    resampler_identifier: str


@dataclass(frozen=True)
class DeploymentMetadata:
    """Deployment-owned identifiers exposed by compatibility metadata."""

    image: str
    pin: str
    precision_policy: str


def is_positive_finite(value: int | float) -> bool:
    """Pure value predicate used by configuration tests."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value > 0
        and math.isfinite(value)
    )


_INTEGER_FIELDS = frozenset(
    {
        "grpc_receive_max_bytes",
        "grpc_config_envelope_max_bytes",
        "unary_max_encoded_audio_bytes",
        "max_riff_header_bytes",
        "load_shed_max_sessions",
        "pre_submit_max_samples",
        "ws_receive_max_bytes",
        "ws_event_envelope_max_bytes",
        "http_multipart_envelope_max_bytes",
        "http_content_type_max_bytes",
        "http_request_header_max_bytes",
        "http_multipart_boundary_max_bytes",
        "http_multipart_max_parts",
        "http_multipart_max_header_bytes",
        "http_text_field_max_bytes",
    }
)


_DISPOSITIONS = {
    "language_code": "honored",
    "enable_automatic_punctuation": "model_intrinsic",
    "verbatim_transcripts": "model_intrinsic",
    "speaker_diarization": "unsupported_capability",
    "diarization_config": "unsupported_capability",
    "word_boosting": "unsupported_capability",
    "speech_contexts": "unsupported_capability",
    "endpointing_config": "unsupported_capability",
    "runtime_config": "invalid_config_field",
}


# @spec ENV-MOD-003, ING-VEH-012, ING-GRPC-008, ING-HTTP-006, ING-NIMWS-010
def validate_frontend_config(config: FrontendConfig) -> None:
    """Validate finite, unit-bearing values and their required relationships."""
    for name, value in vars(config).items():
        if name in {
            "grpc_bind",
            "grpc_keepalive_seconds",
            "max_session_duration",
            "resampler_identifier",
        }:
            continue
        if not is_positive_finite(value):
            raise ValueError(f"{name} must be finite and positive")
        if name in _INTEGER_FIELDS and type(value) is not int:
            raise ValueError(f"{name} must be an integer")

    if not isinstance(config.grpc_bind, str):
        raise ValueError("grpc_bind must be a host:port value")
    host, separator, port_text = config.grpc_bind.rpartition(":")
    if not host or separator != ":" or not port_text.isdecimal():
        raise ValueError("grpc_bind must be a host:port value")
    port = int(port_text)
    if port < 1 or port > 65535:
        raise ValueError("grpc_bind port must be in 1..65535")
    for name in ("grpc_keepalive_seconds", "max_session_duration"):
        value = getattr(config, name)
        if value is not None and not is_positive_finite(value):
            raise ValueError(f"{name} must be omitted or finite and positive")
    if config.resampler_identifier != RESAMPLER_ID:
        raise ValueError(
            "resampler_identifier must equal the pinned RESAMPLER_ID"
        )

    if config.grpc_receive_max_bytes != (
        config.unary_max_encoded_audio_bytes
        + config.grpc_config_envelope_max_bytes
    ):
        raise ValueError(
            "grpc_receive_max_bytes must equal unary_max_encoded_audio_bytes "
            "+ grpc_config_envelope_max_bytes"
        )
    if config.max_riff_header_bytes > config.unary_max_encoded_audio_bytes:
        raise ValueError(
            "max_riff_header_bytes must not exceed "
            "unary_max_encoded_audio_bytes"
        )
    if config.ws_receive_max_bytes < config.ws_event_envelope_max_bytes:
        raise ValueError(
            "ws_receive_max_bytes must cover ws_event_envelope_max_bytes"
        )
    if (
        config.http_content_type_max_bytes
        > config.http_request_header_max_bytes
    ):
        raise ValueError(
            "http_content_type_max_bytes must not exceed "
            "http_request_header_max_bytes"
        )
    if config.http_multipart_envelope_max_bytes <= (
        config.unary_max_encoded_audio_bytes
    ):
        raise ValueError(
            "http_multipart_envelope_max_bytes requires positive framing "
            "allowance"
        )
    minimum_shutdown_grace = (
        config.session_finalization_timeout + config.session_cleanup_timeout
    )
    if config.plugin_shutdown_grace < minimum_shutdown_grace:
        raise ValueError(
            "plugin_shutdown_grace must cover session_finalization_timeout "
            "+ session_cleanup_timeout"
        )


# @spec ING-GRPC-005, ING-NIMWS-008
def dispositioned_fields(
    values: dict[str, object], *, dialect: str
) -> dict[str, str]:
    """Return exhaustive, shared Riva/NIM field dispositions.

    The dialect selects wire spelling at its adapter boundary; this common
    contract makes every field explicit before either transport projects it.
    """
    if dialect not in {"grpc", "nim"}:
        raise ValueError(f"unknown compatibility dialect: {dialect}")
    return {
        field: _DISPOSITIONS.get(field, "invalid_config_field")
        for field in values
    }


def _read_config_value(value: str) -> dict[str, Any]:
    """Decode one inline JSON object or a path to one UTF-8 JSON object."""
    candidate = value.strip()
    if candidate.startswith("{"):
        encoded = candidate
    else:
        path = Path(candidate).expanduser()
        try:
            encoded = path.read_text(encoding="utf-8")
        except OSError as error:
            raise ValueError(
                f"application-plugin config path is not readable: {path}"
            ) from error
    try:
        decoded = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError(
            "application-plugin config must be valid JSON"
        ) from error
    if not isinstance(decoded, dict):
        raise ValueError("application-plugin config must be one JSON object")
    return decoded


# @spec ING-VEH-009, ING-VEH-012, ING-SHIM-002
def load_plugin_config(
    value: str | None,
) -> tuple[FrontendConfig, DeploymentMetadata]:
    """Load and strictly validate the selected plugin's opaque JSON value."""
    if value is None or not value.strip():
        raise ValueError("riva_frontend configuration is required")
    decoded = _read_config_value(value)
    frontend_names = {field.name for field in fields(FrontendConfig)}
    omission_defaults: dict[str, object] = {
        "grpc_keepalive_seconds": None,
        "resampler_identifier": RESAMPLER_ID,
        "session_idle_timeout": 60.0,
    }
    metadata_names = {
        "deployment_image",
        "pin",
        "precision_policy",
    }
    unknown = set(decoded) - frontend_names - metadata_names
    missing = (frontend_names - set(omission_defaults) | metadata_names) - set(
        decoded
    )
    if unknown:
        raise ValueError(
            f"unknown configuration fields: {', '.join(sorted(unknown))}"
        )
    if missing:
        raise ValueError(
            f"missing configuration fields: {', '.join(sorted(missing))}"
        )
    explicit_null = sorted(
        name for name, field_value in decoded.items() if field_value is None
    )
    if explicit_null:
        raise ValueError(
            "configuration fields must not be null: " + ", ".join(explicit_null)
        )
    resolved = {**omission_defaults, **decoded}
    frontend = FrontendConfig(
        **{name: resolved[name] for name in frontend_names}
    )
    validate_frontend_config(frontend)
    metadata_values = {
        name: decoded[name]
        for name in ("deployment_image", "pin", "precision_policy")
    }
    for name, metadata_value in metadata_values.items():
        if not isinstance(metadata_value, str) or not metadata_value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    metadata = DeploymentMetadata(
        image=metadata_values["deployment_image"],
        pin=metadata_values["pin"],
        precision_policy=metadata_values["precision_policy"],
    )
    return frontend, metadata
