"""The frontend's configuration, the engine's stream deadlines and capacity.

Options come from ``--additional-config`` under the key ``riva_frontend``.
Validation is strict and reports every failure together, each naming the
plugin key and the option. The engine's stream deadlines are read from the
``stream_admission`` mapping core's state scheduler reads, and the engine's
capacity from the ``vllm_config`` core hands the plugin.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from typing import Any

from vllm_riva_frontend.frontend import RESAMPLER_ID

#: The ``--additional-config`` key of the frontend's options.
PLUGIN_KEY = "riva_frontend"
#: The ``--additional-config`` key of the engine's stream deadlines.
STREAM_ADMISSION_KEY = "stream_admission"


class StartupError(Exception):
    """A startup check failed; ``failures`` names every failed check.

    Defined here so that ``ConfigError`` can subclass it; ``lifecycle``
    exports it as the error ``init_state`` and ``start`` raise.
    """

    def __init__(self, failures: Iterable[str] | str) -> None:
        """Keep the failures, one message each, in order."""
        self.failures: list[str] = (
            [failures] if isinstance(failures, str) else list(failures)
        )
        super().__init__("; ".join(self.failures))


class ConfigError(StartupError):
    """The configuration is invalid; ``failures`` names every option."""


@dataclass(frozen=True)
class FrontendConfig:
    """One attribute per option; see the vehicle leaf's configuration table.

    ``max_concurrent_files`` and ``max_file_owners`` are ``None`` until
    ``init_state`` derives them from the engine's capacity.
    ``file_execution_timeout`` defaults to
    ``max(300, 2 * unary_max_decoded_duration_seconds)``.
    """

    grpc_bind: str = "0.0.0.0:50051"
    grpc_receive_max_bytes: int = 33_619_968
    grpc_config_envelope_max_bytes: int = 65_536
    grpc_keepalive_seconds: float | None = None
    unary_max_encoded_audio_bytes: int = 33_554_432
    unary_max_decoded_duration_seconds: float = 600
    max_riff_header_bytes: int = 1_048_576
    load_shed_max_sessions: int = 64
    max_file_owners: int | None = None
    max_concurrent_files: int | None = None
    file_gate_wait_timeout: float = 30
    pre_submit_max_samples: int = 65_536
    preconfiguration_timeout: float = 30
    first_audio_timeout: float = 10
    session_idle_timeout: float = 45
    lease_call_timeout: float = 45
    session_finalization_timeout: float = 30
    session_cleanup_timeout: float = 30
    max_session_duration: float | None = None
    ws_receive_max_bytes: int = 16_777_216
    ws_event_envelope_max_bytes: int = 8_388_608
    http_multipart_envelope_max_bytes: int = 33_816_576
    http_content_type_max_bytes: int = 4_096
    http_multipart_boundary_max_bytes: int = 200
    http_multipart_max_parts: int = 32
    http_multipart_max_header_bytes: int = 8_192
    http_text_field_max_bytes: int = 4_096
    http_body_read_timeout: float = 120
    file_execution_timeout: float | None = None
    response_send_timeout: float = 60
    session_factory: str | None = None
    resampler_identifier: str = RESAMPLER_ID
    deployment_image: str = "unspecified"
    pin: str = "unspecified"
    precision_policy: str = "fp32-bringup-v1"


_INTEGER_OPTIONS = frozenset(
    {
        "grpc_receive_max_bytes",
        "grpc_config_envelope_max_bytes",
        "unary_max_encoded_audio_bytes",
        "max_riff_header_bytes",
        "load_shed_max_sessions",
        "max_file_owners",
        "max_concurrent_files",
        "pre_submit_max_samples",
        "ws_receive_max_bytes",
        "ws_event_envelope_max_bytes",
        "http_multipart_envelope_max_bytes",
        "http_content_type_max_bytes",
        "http_multipart_boundary_max_bytes",
        "http_multipart_max_parts",
        "http_multipart_max_header_bytes",
        "http_text_field_max_bytes",
    }
)
_STRING_OPTIONS = frozenset(
    {
        "grpc_bind",
        "session_factory",
        "resampler_identifier",
        "deployment_image",
        "pin",
        "precision_policy",
    }
)
_OPTIONS = tuple(field.name for field in fields(FrontendConfig))
_NUMBER_OPTIONS = frozenset(_OPTIONS) - _INTEGER_OPTIONS - _STRING_OPTIONS

#: Values the metadata route publishes as this deployment's provenance.
_PROVENANCE_OPTIONS = (
    "deployment_image",
    "pin",
    "precision_policy",
    "resampler_identifier",
)
#: A bare hexadecimal token: the shape of a model-profile hash.
_BARE_HEX = re.compile(r"[0-9A-Fa-f]{8,64}")


def _named(option: str) -> str:
    return f"{PLUGIN_KEY}.{option}"


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _value_failure(option: str, value: object) -> str | None:
    """The failure of one option's value on its own, or ``None``."""
    if option in _INTEGER_OPTIONS:
        if type(value) is not int:
            return f"{_named(option)} must be an integer, not {value!r}"
        if value <= 0:
            return f"{_named(option)} must be positive, not {value!r}"
        return None
    if option in _NUMBER_OPTIONS:
        if not _is_number(value):
            return f"{_named(option)} must be a number, not {value!r}"
        assert isinstance(value, (int, float))
        if not math.isfinite(value) or value <= 0:
            return (
                f"{_named(option)} must be finite and positive, not {value!r}"
            )
        return None
    if not isinstance(value, str) or not value:
        return f"{_named(option)} must be a non-empty string, not {value!r}"
    if option == "grpc_bind":
        return _bind_failure(value)
    if option in _PROVENANCE_OPTIONS:
        return _provenance_failure(option, value)
    return None


def _bind_failure(value: str) -> str | None:
    host, separator, port = value.rpartition(":")
    if not separator or not host or not port.isdecimal():
        return f"{_named('grpc_bind')} must be host:port, not {value!r}"
    # The bound port must be known, so port 0 is refused.
    if not 1 <= int(port) <= 65535:
        return f"{_named('grpc_bind')} port must be 1 to 65535, not {port}"
    return None


# @spec ING-EP-052
def _provenance_failure(option: str, value: str) -> str | None:
    if value != value.strip():
        return f"{_named(option)} must not carry surrounding whitespace"
    if "://" in value:
        return f"{_named(option)} must not be a registry reference (://)"
    # A commit SHA is the deployment's own pin; elsewhere a bare hex token
    # would present a profile hash as the deployment's identity.
    if option != "pin" and _BARE_HEX.fullmatch(value):
        return f"{_named(option)} must not be a bare hexadecimal token"
    if option == "resampler_identifier" and value != RESAMPLER_ID:
        return (
            f"{_named(option)} must be the built-in resampler "
            f"{RESAMPLER_ID!r}, not {value!r}"
        )
    return None


def _relation_failures(values: Mapping[str, Any]) -> list[str]:
    """The failed relations among options that are each valid on their own."""
    failures: list[str] = []
    receive = values["grpc_receive_max_bytes"]
    audio = values["unary_max_encoded_audio_bytes"]
    envelope = values["grpc_config_envelope_max_bytes"]
    if receive != audio + envelope:
        failures.append(
            f"{_named('grpc_receive_max_bytes')} ({receive}) must equal "
            f"unary_max_encoded_audio_bytes + grpc_config_envelope_max_bytes "
            f"({audio + envelope})"
        )
    if values["ws_receive_max_bytes"] < values["ws_event_envelope_max_bytes"]:
        failures.append(
            f"{_named('ws_receive_max_bytes')} must be at least "
            "ws_event_envelope_max_bytes"
        )
    if values["max_riff_header_bytes"] > audio:
        failures.append(
            f"{_named('max_riff_header_bytes')} must be at most "
            "unary_max_encoded_audio_bytes"
        )
    failures.extend(
        file_cap_failures(
            values["max_concurrent_files"],
            values["max_file_owners"],
            values["load_shed_max_sessions"],
        )
    )
    return failures


def file_cap_failures(
    max_concurrent_files: int | None,
    max_file_owners: int | None,
    load_shed_max_sessions: int,
) -> list[str]:
    """The failed relations of the file sub-cap; unset values are skipped."""
    failures: list[str] = []
    if max_file_owners is None:
        return failures
    if max_concurrent_files is not None and (
        max_file_owners < max_concurrent_files
    ):
        failures.append(
            f"{_named('max_file_owners')} ({max_file_owners}) must be at "
            f"least max_concurrent_files ({max_concurrent_files})"
        )
    if max_file_owners >= load_shed_max_sessions:
        failures.append(
            f"{_named('max_file_owners')} ({max_file_owners}) must be below "
            f"load_shed_max_sessions ({load_shed_max_sessions})"
        )
    return failures


# @spec ING-EP-018, ING-EP-046, ING-EP-047, ING-EP-048, ING-EP-049
# @spec ING-EP-050, ING-EP-051, ING-EP-053, ING-EP-032, ING-ADM-017
# @spec ING-NIMWS-012
def parse_config(additional_config: Mapping[str, Any] | None) -> FrontendConfig:
    """Read and validate the ``riva_frontend`` options.

    An absent key or option means its default. An unknown key, an explicit
    ``null``, a number that is not finite and positive, a non-integer for an
    integer option and every failed relation fail, all reported together in
    one ``ConfigError``. ``max_concurrent_files`` and ``max_file_owners``
    stay ``None`` unless set: they are derived from capacity later.
    """
    section: object = {}
    if additional_config is not None and PLUGIN_KEY in additional_config:
        section = additional_config[PLUGIN_KEY]
    if not isinstance(section, Mapping):
        raise ConfigError([f"{PLUGIN_KEY} must be a JSON object"])
    failures: list[str] = []
    values: dict[str, Any] = {}
    for option, value in section.items():
        if option not in _OPTIONS:
            failures.append(f"{_named(str(option))} is not an option")
        elif value is None:
            failures.append(f"{_named(option)} must not be null")
        else:
            failure = _value_failure(option, value)
            if failure is None:
                values[option] = value
            else:
                failures.append(failure)
    if failures:
        raise ConfigError(failures)
    config = FrontendConfig(**values)
    relations = _relation_failures(dataclasses.asdict(config))
    if relations:
        raise ConfigError(relations)
    if config.file_execution_timeout is None:
        derived = max(300.0, 2 * config.unary_max_decoded_duration_seconds)
        config = dataclasses.replace(config, file_execution_timeout=derived)
    return config


# ------------------------------------------------- engine stream deadlines


@dataclass(frozen=True)
class StreamDeadlines:
    """The engine's deadlines for an admitted stream, in seconds."""

    never_fed_timeout: float
    parked_idle_timeout: float
    draining_timeout: float


#: The engine's defaults when ``stream_admission`` leaves a key out.
DEFAULT_STREAM_DEADLINES = StreamDeadlines(15.0, 60.0, 40.0)


# @spec ING-EP-013
def read_stream_deadlines(
    additional_config: Mapping[str, Any] | None,
) -> StreamDeadlines:
    """Read the engine's stream deadlines from ``stream_admission``.

    A missing section or key takes the engine's default; a present value
    that is not a finite positive number fails, because the frontend would
    not know the deadline the engine enforces.
    """
    section: object = {}
    if additional_config is not None and STREAM_ADMISSION_KEY in (
        additional_config
    ):
        section = additional_config[STREAM_ADMISSION_KEY]
    if not isinstance(section, Mapping):
        raise ConfigError([f"{STREAM_ADMISSION_KEY} must be a JSON object"])
    values: dict[str, float] = {}
    failures: list[str] = []
    for field in fields(StreamDeadlines):
        default = getattr(DEFAULT_STREAM_DEADLINES, field.name)
        value = section.get(field.name, default)
        if not _is_number(value) or not math.isfinite(value) or value <= 0:
            failures.append(
                f"{STREAM_ADMISSION_KEY}.{field.name} must be a finite "
                f"positive number, not {value!r}"
            )
        else:
            values[field.name] = float(value)
    if failures:
        raise ConfigError(failures)
    return StreamDeadlines(**values)


#: Each frontend bound and the engine deadline it must stay below.
_BOUNDS = (
    ("first_audio_timeout", "never_fed_timeout"),
    ("session_idle_timeout", "parked_idle_timeout"),
    ("lease_call_timeout", "parked_idle_timeout"),
    ("session_finalization_timeout", "draining_timeout"),
)


# @spec ING-EP-014, ING-EP-032
def check_bounds(
    config: FrontendConfig, deadlines: StreamDeadlines
) -> list[str]:
    """Return a failure for every frontend bound not below its deadline."""
    failures: list[str] = []
    for option, deadline in _BOUNDS:
        bound = getattr(config, option)
        engine = getattr(deadlines, deadline)
        if not bound < engine:
            failures.append(
                f"{_named(option)} ({bound} s) must be below "
                f"{STREAM_ADMISSION_KEY}.{deadline} ({engine} s)"
            )
    return failures


# ------------------------------------------------------------- capacity


@dataclass(frozen=True)
class Capacity:
    """The engine's state-slot count and active-request limit."""

    slots: int
    active_limit: int


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


# @spec ING-EP-030, ING-EP-029
def read_capacity(vllm_config: Any) -> Capacity:
    """Read the state-slot count and the active-request limit.

    Slots are the page count less the reserved null page; the active limit
    is ``max_num_active_seqs``, or ``max_num_seqs`` when that is unset. An
    unknown page count or an unreadable limit fails.
    """
    failures: list[str] = []
    cache = getattr(vllm_config, "cache_config", None)
    pages = getattr(cache, "num_gpu_blocks", None)
    if not _positive_int(pages):
        failures.append(
            "vllm_config.cache_config.num_gpu_blocks is "
            f"{pages!r}: the state-slot count is not known"
        )
    scheduler = getattr(vllm_config, "scheduler_config", None)
    active = getattr(scheduler, "max_num_active_seqs", None)
    if active is None:
        active = getattr(scheduler, "max_num_seqs", None)
    if not _positive_int(active):
        failures.append(
            "vllm_config.scheduler_config.max_num_seqs (or "
            f"max_num_active_seqs) is {active!r}: the active-request limit "
            "cannot be read"
        )
    if failures:
        raise StartupError(failures)
    assert isinstance(pages, int) and isinstance(active, int)
    return Capacity(slots=pages - 1, active_limit=active)


# @spec ING-ADM-014, ING-ADM-010
def derive_file_caps(
    config: FrontendConfig, capacity: Capacity, service_present: bool
) -> tuple[int, int, bool]:
    """Return ``(max_concurrent_files, max_file_owners, files_enabled)``.

    Without core's transcription service, or on a server whose smaller of
    slots and active limit is below 2 with no configured gate, files are
    disabled and both caps are 0. A configured gate must be below both
    limits, and the sub-cap at least the gate and below the ingress cap;
    otherwise ``ConfigError``.
    """
    if not service_present:
        return 0, 0, False
    smaller = min(capacity.slots, capacity.active_limit)
    concurrent = config.max_concurrent_files
    if concurrent is None:
        if smaller < 2:
            return 0, 0, False
        concurrent = min(4, max(1, smaller // 4))
    elif not concurrent < smaller:
        raise ConfigError(
            [
                f"{_named('max_concurrent_files')} ({concurrent}) must be "
                f"below both the state-slot count ({capacity.slots}) and the "
                f"active-request limit ({capacity.active_limit})"
            ]
        )
    owners = config.max_file_owners
    if owners is None:
        owners = concurrent + 2
    failures = file_cap_failures(
        concurrent, owners, config.load_shed_max_sessions
    )
    if failures:
        raise ConfigError(failures)
    return concurrent, owners, True


# ----------------------------------------------------------- provenance


@dataclass(frozen=True)
class DeploymentProvenance:
    """This deployment's own identifiers, as the metadata route shows them."""

    image: str
    pin: str
    precision_policy: str
    resampler: str


# @spec ING-SHIM-002, ING-SHIM-012
def build_deployment_provenance(config: FrontendConfig) -> DeploymentProvenance:
    """Build the provenance from exactly its four typed, validated options."""
    return DeploymentProvenance(
        image=config.deployment_image,
        pin=config.pin,
        precision_policy=config.precision_policy,
        resampler=config.resampler_identifier,
    )
