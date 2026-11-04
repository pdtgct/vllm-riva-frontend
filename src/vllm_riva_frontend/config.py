"""Finite frontend configuration and dialect-disposition contract."""

import json
import math
import os
import re
from collections.abc import Mapping
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

#: The endpoint-plugin entry-point name this package registers under
#: (``vllm.endpoint_plugins`` / ``VLLM_PLUGINS``), named in every
#: configuration-loading error so a multi-plugin host log is unambiguous
#: about which plugin's configuration failed. Core has no
#: ``--application-plugin-config``; the value arrives from
#: ``VLLM_RIVA_FRONTEND_CONFIG`` or ``--additional-config``'s
#: ``riva_frontend`` key (see ``resolve_config_value``).
_PLUGIN_KEY = "riva_frontend"
_CONFIG_ENV = "VLLM_RIVA_FRONTEND_CONFIG"
_CADENCE = re.compile(r"^[1-9][0-9]*ms$")

#: The qualified zero-config deployment profile (D5): every
#: ``FrontendConfig`` field plus the three ``DeploymentMetadata`` source
#: fields, resolved to the values this project already deploys with at
#: ``deploy/riva_frontend/home.values.yaml``.  ``load_plugin_config`` uses
#: this profile to fill any field the caller does not explicitly supply,
#: including when no configuration is supplied at all -- an explicit value
#: is still validated and can still be rejected (D5 does not relax
#: validation, only what may be omitted).
_DEFAULT_PROFILE: dict[str, object] = {
    "grpc_bind": "0.0.0.0:50051",
    "grpc_receive_max_bytes": 33619968,
    "grpc_config_envelope_max_bytes": 65536,
    "unary_max_encoded_audio_bytes": 33554432,
    "unary_max_decoded_duration_seconds": 600.0,
    "max_riff_header_bytes": 1048576,
    "load_shed_max_sessions": 64,
    "pre_submit_max_samples": 65536,
    "preconfiguration_timeout": 30.0,
    "session_idle_timeout": 60.0,
    "session_finalization_timeout": 180.0,
    "session_cleanup_timeout": 30.0,
    "plugin_shutdown_grace": 240.0,
    "ws_receive_max_bytes": 16777216,
    "ws_event_envelope_max_bytes": 8388608,
    "http_multipart_envelope_max_bytes": 33816576,
    "http_content_type_max_bytes": 4096,
    "http_request_header_max_bytes": 65536,
    "http_multipart_boundary_max_bytes": 200,
    "http_multipart_max_parts": 5,
    "http_multipart_max_header_bytes": 8192,
    "http_text_field_max_bytes": 4096,
    "http_request_timeout": 900.0,
    "grpc_keepalive_seconds": None,
    "max_session_duration": 600.0,
    "resampler_identifier": RESAMPLER_ID,
    # Streaming lookahead is configuration, not a call-site literal.
    # 560ms is the qualified profile's streaming arm (the same arm NIM
    # Realtime uses). Unary/offline stays on its own 1120ms arm.
    "streaming_cadence": "560ms",
    # Offline Recognize and POST /v1/audio/transcriptions shadow core's
    # transcription route when enabled. The default leaves those paths
    # to core.
    "serve_offline": False,
    # None selects the only installed vllm_riva.session_factories entry.
    "session_factory": None,
    # Provenance is a per-deployment FACT, not policy: a zero-config
    # deployment must never advertise another deployment's image or pin
    # through /v1/metadata. These two default to an honest marker; the
    # deployment values file supplies the real identity.
    "deployment_image": "unspecified",
    "pin": "unspecified",
    "precision_policy": "fp32-bringup-v1",
}

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
    streaming_cadence: str = "560ms"
    serve_offline: bool = False
    session_factory: str | None = None


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
            "streaming_cadence",
            "serve_offline",
            "session_factory",
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
    if not isinstance(config.streaming_cadence, str) or (
        _CADENCE.fullmatch(config.streaming_cadence) is None
    ):
        raise ValueError(
            "streaming_cadence must be a positive millisecond label "
            "such as 560ms"
        )
    if not isinstance(config.serve_offline, bool):
        raise ValueError("serve_offline must be a boolean")
    factory_name = config.session_factory
    if factory_name is not None and (
        not isinstance(factory_name, str)
        or not factory_name.strip()
        or factory_name != factory_name.strip()
    ):
        raise ValueError(
            "session_factory must be omitted or a non-empty entry-point name"
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
    """Decode one inline JSON object or a path to one UTF-8 JSON object.

    A value is inline JSON, never a path, the moment its (whitespace-
    trimmed) text starts with ``{`` -- malformed inline JSON then fails
    with its own error naming this plugin's entry-point key, rather than
    silently being reinterpreted as a (almost certainly nonexistent) path.
    """
    candidate = value.strip()
    if candidate.startswith("{"):
        try:
            decoded = json.loads(candidate)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{_PLUGIN_KEY} endpoint-plugin inline config is not "
                f"valid JSON: {error}"
            ) from error
    else:
        path = Path(candidate).expanduser()
        try:
            encoded = path.read_text(encoding="utf-8")
        except OSError as error:
            raise ValueError(
                f"{_PLUGIN_KEY} endpoint-plugin config path is not "
                f"readable: {path}"
            ) from error
        try:
            decoded = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{_PLUGIN_KEY} endpoint-plugin config file is not "
                f"valid JSON: {error}"
            ) from error
    if not isinstance(decoded, dict):
        raise ValueError(
            f"{_PLUGIN_KEY} endpoint-plugin config must be one JSON object"
        )
    return decoded


# @spec ING-VEH-009, ING-VEH-012, ING-SHIM-002
def load_plugin_config(
    value: str | None,
) -> tuple[FrontendConfig, DeploymentMetadata]:
    """Load and strictly validate the selected plugin's opaque JSON value.

    Absent configuration (``None``, or blank) is D5's zero-config launch:
    every field resolves to the qualified ``_DEFAULT_PROFILE`` and this
    never raises for that reason alone.  A field the caller does supply --
    whether alongside other fields, or as the whole configuration -- is
    still validated exactly as an explicit value always has been; a
    default never substitutes for, or silently repairs, a rejected
    explicit value.
    """
    decoded = (
        _read_config_value(value) if value is not None and value.strip() else {}
    )
    frontend_names = {field.name for field in fields(FrontendConfig)}
    metadata_names = {
        "deployment_image",
        "pin",
        "precision_policy",
    }
    unknown = set(decoded) - frontend_names - metadata_names
    if unknown:
        raise ValueError(
            f"unknown configuration fields: {', '.join(sorted(unknown))}"
        )
    explicit_null = sorted(
        name for name, field_value in decoded.items() if field_value is None
    )
    if explicit_null:
        raise ValueError(
            "configuration fields must not be null: " + ", ".join(explicit_null)
        )
    resolved = {**_DEFAULT_PROFILE, **decoded}
    frontend = FrontendConfig(
        **{name: resolved[name] for name in frontend_names}
    )
    validate_frontend_config(frontend)
    metadata_values = {
        name: resolved[name]
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


def resolve_config_value(
    args: object | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve the plugin config string from the server, then the environment.

    Core has no ``--application-plugin-config``. Precedence:

    1. ``args.additional_config["riva_frontend"]`` — a JSON object, inline
       JSON text, or a filesystem path. This is vLLM's ``--additional-config``
       Namespace field.
    2. ``VLLM_RIVA_FRONTEND_CONFIG`` — the same string forms (inline JSON or
       a path). Available before server args exist.

    A missing or blank value returns ``None``, which
    ``load_plugin_config`` treats as the qualified default profile.
    ``--additional-config`` wins when both channels are set.
    """
    from_args = _config_from_namespace(args)
    if from_args is not None:
        return from_args
    env = os.environ if environ is None else environ
    raw = env.get(_CONFIG_ENV)
    if raw is None or not raw.strip():
        return None
    return raw.strip()


def _config_from_namespace(args: object | None) -> str | None:
    """Read the ``riva_frontend`` key from a server Namespace, if present."""
    if args is None:
        return None
    additional = getattr(args, "additional_config", None)
    if not isinstance(additional, dict) or _PLUGIN_KEY not in additional:
        return None
    value = additional[_PLUGIN_KEY]
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, dict):
        return json.dumps(value)
    raise ValueError(
        f"{_PLUGIN_KEY} additional-config value must be a JSON object "
        "or a string (inline JSON or a file path)"
    )
