"""Finite frontend configuration and dialect-disposition contract."""

import json
import math
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from vllm_riva_frontend.frontend import RESAMPLER_ID

#: A NIM-owned model-selection identifier.  This deployment's own
#: server-authored provenance must never publish one; see
#: ``build_deployment_provenance``, which makes that structurally true
#: for *keys* rather than filtering for it, and ``load_plugin_config``
#: / ``_validate_provenance_value``, which does the same for *values*.
FORBIDDEN_PROVENANCE_KEY = "selectedModelProfileId"

#: A bare hex token, either case, spanning the whole value -- the shape a
#: NIM profile hash takes.  Deliberately does not match a "sha256:<hex>"
#: image digest, which is colon-prefixed rather than a full-string match.
_BARE_HEX_HASH = re.compile(r"^[0-9A-Fa-f]{8,64}$")
_SCHEME_SEPARATOR = "://"


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
    """Deployment-owned identifiers exposed by compatibility metadata.

    Only ``load_plugin_config`` constructs this; its fields' *values*,
    not only their allowlisted set of keys, are validated there (see
    ``_validate_provenance_value``) before this dataclass can exist.
    """

    image: str
    pin: str
    precision_policy: str


# @spec ING-SHIM-002
def _validate_provenance_value(name: str, value: str) -> None:
    """Reject a server-authored provenance value shaped like NIM identity.

    Applied to every provenance-bound field (image, pin, precision_policy,
    resampler) at configuration ingestion -- the only place any of these
    values ever originates -- so the allowlist-by-construction claim in
    ``build_deployment_provenance`` is honest about values, not only keys:
    a scheme-smuggled registry reference (``ngc://...`` or any other
    ``scheme://`` form) or a bare NIM profile hash cannot reach
    /v1/metadata through any of the four fields, fail-closed at startup.

    A ``sha256:<hex>`` image digest remains legal: it is colon-prefixed,
    not ``://``-scheme-prefixed, and (with that prefix included) is not a
    full-string bare-hex value either.
    """
    if value != value.strip():
        raise ValueError(f"{name} must not carry leading/trailing whitespace")
    if _SCHEME_SEPARATOR in value:
        raise ValueError(f"{name} must not contain a scheme reference (://)")
    if _BARE_HEX_HASH.fullmatch(value):
        raise ValueError(f"{name} must not be a bare hex identifier")


# @spec ING-SHIM-002
def build_deployment_provenance(
    metadata: DeploymentMetadata, *, resampler_identifier: str
) -> dict[str, str]:
    """Build the one server-authored provenance object by allowlist.

    This is allowlist construction, not filtered copying, for *keys*: the
    return value can only ever contain these four factual identifiers,
    because the signature takes exactly those typed fields and nothing
    else to draw from.  No compatibility surface may build its own
    server-authored provenance by copying an arbitrary mapping.

    The *values* of those four fields are constrained separately, at
    configuration ingestion (``load_plugin_config``, via
    ``_validate_provenance_value``): a NIM registry identity or a bare
    NIM profile hash is rejected there, fail-closed, before a
    ``DeploymentMetadata`` can exist to be threaded through here.  This
    function trusts that gate rather than re-validating.
    """
    return {
        "image": metadata.image,
        "pin": metadata.pin,
        "precision_policy": metadata.precision_policy,
        "resampler": resampler_identifier,
    }


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
    # Every provenance-bound value, not only the three DeploymentMetadata
    # fields: build_deployment_provenance also threads resampler_identifier
    # into /v1/metadata, so it is validated by the same shared rule here.
    provenance_bound_values = {
        **metadata_values,
        "resampler_identifier": frontend.resampler_identifier,
    }
    for name, value in provenance_bound_values.items():
        _validate_provenance_value(name, value)
    metadata = DeploymentMetadata(
        image=metadata_values["deployment_image"],
        pin=metadata_values["pin"],
        precision_policy=metadata_values["precision_policy"],
    )
    return frontend, metadata
