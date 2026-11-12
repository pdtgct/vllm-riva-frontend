"""The ``riva_frontend`` configuration: strict parsing, defaults, relations."""

from __future__ import annotations

import math
from typing import Any

import pytest

GRPC_DEFAULTS = {
    "grpc_bind": "0.0.0.0:50051",
    "grpc_receive_max_bytes": 33_619_968,
    "grpc_config_envelope_max_bytes": 65_536,
    "unary_max_encoded_audio_bytes": 33_554_432,
    "unary_max_decoded_duration_seconds": 600,
    "max_riff_header_bytes": 1_048_576,
    "pre_submit_max_samples": 65_536,
}
TIMEOUT_DEFAULTS = {
    "preconfiguration_timeout": 30,
    "session_cleanup_timeout": 30,
    "file_gate_wait_timeout": 30,
    "http_body_read_timeout": 120,
    "response_send_timeout": 60,
}
HTTP_WS_DEFAULTS = {
    "ws_receive_max_bytes": 16_777_216,
    "ws_event_envelope_max_bytes": 8_388_608,
    "http_multipart_envelope_max_bytes": 33_816_576,
    "http_content_type_max_bytes": 4_096,
    "http_multipart_boundary_max_bytes": 200,
    "http_multipart_max_parts": 32,
    "http_multipart_max_header_bytes": 8_192,
    "http_text_field_max_bytes": 4_096,
}


def parse(options: dict[str, Any] | None) -> Any:
    from vllm_riva_frontend.config import parse_config

    return parse_config({} if options is None else {"riva_frontend": options})


def parse_failures(options: dict[str, Any]) -> list[str]:
    from vllm_riva_frontend.config import ConfigError

    with pytest.raises(ConfigError) as info:
        parse(options)
    failures = list(info.value.failures)
    assert failures
    return failures


# ------------------------------------------------------------ strictness


# @spec ING-EP-018
def test_absent_key_and_empty_section_both_mean_defaults() -> None:
    from vllm_riva_frontend.config import PLUGIN_KEY

    assert PLUGIN_KEY == "riva_frontend"
    assert parse(None) == parse({})


# @spec ING-EP-018
def test_unknown_keys_fail_and_every_one_is_named() -> None:
    failures = parse_failures({"surprise": 1, "also_unknown": True})
    assert any("surprise" in f for f in failures)
    assert any("also_unknown" in f for f in failures)
    assert all("riva_frontend" in f for f in failures)


# @spec ING-EP-018
@pytest.mark.parametrize(
    "option",
    [
        "grpc_keepalive_seconds",
        "max_session_duration",
        "session_factory",
        "load_shed_max_sessions",
        "preconfiguration_timeout",
        "deployment_image",
        "max_concurrent_files",
    ],
)
def test_explicit_null_fails_naming_the_option(option: str) -> None:
    failures = parse_failures({option: None})
    assert any(option in f and "riva_frontend" in f for f in failures)


# @spec ING-EP-018
@pytest.mark.parametrize("value", [0, -1, -0.5, math.inf, -math.inf, math.nan])
@pytest.mark.parametrize(
    "option",
    [
        "preconfiguration_timeout",
        "session_cleanup_timeout",
        "grpc_keepalive_seconds",
        "max_session_duration",
        "load_shed_max_sessions",
    ],
)
def test_non_finite_or_non_positive_numbers_fail(
    option: str, value: float
) -> None:
    failures = parse_failures({option: value})
    assert any(option in f for f in failures)


# @spec ING-EP-018
@pytest.mark.parametrize(
    "option",
    [
        "load_shed_max_sessions",
        "grpc_receive_max_bytes",
        "pre_submit_max_samples",
        "http_multipart_max_parts",
        "max_file_owners",
    ],
)
def test_integer_option_given_a_non_integer_fails(option: str) -> None:
    failures = parse_failures({option: 2.5})
    assert any(option in f for f in failures)


# @spec ING-EP-018
def test_all_failures_are_reported_together() -> None:
    failures = parse_failures(
        {
            "surprise": 1,
            "preconfiguration_timeout": None,
            "session_cleanup_timeout": -3,
            "load_shed_max_sessions": 1.5,
        }
    )
    for option in (
        "surprise",
        "preconfiguration_timeout",
        "session_cleanup_timeout",
        "load_shed_max_sessions",
    ):
        assert any(option in f for f in failures), option


# @spec ING-EP-018, ING-EP-107
async def test_init_state_fails_on_invalid_configuration(
    clock, fake_factory, helpers
) -> None:
    failures = await helpers.startup_failures(
        clock,
        fake_factory,
        options={"surprise": 1, "load_shed_max_sessions": 0},
    )
    assert helpers.mentions(failures, "riva_frontend", "surprise")
    assert helpers.mentions(failures, "load_shed_max_sessions")


# @spec ING-EP-046
def test_there_is_no_plugin_shutdown_grace_option() -> None:
    from vllm_riva_frontend.config import FrontendConfig

    assert not hasattr(parse({}), "plugin_shutdown_grace")
    assert "plugin_shutdown_grace" not in getattr(
        FrontendConfig, "__dataclass_fields__", {}
    )
    failures = parse_failures({"plugin_shutdown_grace": 40})
    assert any("plugin_shutdown_grace" in f for f in failures)


# -------------------------------------------------------------- defaults


# @spec ING-EP-047
def test_grpc_and_listener_defaults() -> None:
    config = parse({})
    for option, value in GRPC_DEFAULTS.items():
        assert getattr(config, option) == value, option
    assert config.grpc_keepalive_seconds is None
    assert config.max_session_duration is None


# @spec ING-EP-048
def test_timeout_defaults_and_derived_file_execution_timeout() -> None:
    config = parse({})
    for option, value in TIMEOUT_DEFAULTS.items():
        assert getattr(config, option) == value, option
    assert config.file_execution_timeout == 1200
    assert (
        parse(
            {"unary_max_decoded_duration_seconds": 100}
        ).file_execution_timeout
        == 300
    )
    assert (
        parse(
            {"unary_max_decoded_duration_seconds": 200}
        ).file_execution_timeout
        == 400
    )


# @spec ING-EP-049
def test_websocket_and_http_limit_defaults() -> None:
    config = parse({})
    for option, value in HTTP_WS_DEFAULTS.items():
        assert getattr(config, option) == value, option


# @spec ING-EP-032
def test_frontend_bound_defaults() -> None:
    config = parse({})
    assert config.first_audio_timeout == 10
    assert config.session_idle_timeout == 45
    assert config.lease_call_timeout == 45
    assert config.session_finalization_timeout == 30


# @spec ING-EP-053
def test_provenance_defaults_and_built_in_resampler() -> None:
    from vllm_riva_frontend.frontend import RESAMPLER_ID

    config = parse({})
    assert config.deployment_image == "unspecified"
    assert config.pin == "unspecified"
    assert config.precision_policy == "fp32-bringup-v1"
    assert config.resampler_identifier == "scipy-poly-v1" == RESAMPLER_ID


# @spec ING-EP-053
async def test_a_different_resampler_identifier_fails_startup(
    clock, fake_factory, helpers
) -> None:
    failures = await helpers.startup_failures(
        clock, fake_factory, options={"resampler_identifier": "other-v2"}
    )
    assert helpers.mentions(failures, "resampler_identifier")


# @spec ING-EP-051
def test_session_factory_defaults_to_unset_and_reads_a_name() -> None:
    assert parse({}).session_factory is None
    assert parse({"session_factory": "nemotron"}).session_factory == "nemotron"


# @spec ING-ADM-018
def test_ingress_cap_default_is_64() -> None:
    assert parse({}).load_shed_max_sessions == 64


# ------------------------------------------------------------- relations


# @spec ING-EP-050
@pytest.mark.parametrize(
    "options, named",
    [
        ({"grpc_receive_max_bytes": 33_619_967}, "grpc_receive_max_bytes"),
        (
            {"grpc_config_envelope_max_bytes": 1024},
            "grpc_receive_max_bytes",
        ),
        (
            {"ws_receive_max_bytes": 1024, "ws_event_envelope_max_bytes": 2048},
            "ws_receive_max_bytes",
        ),
        (
            {
                "max_riff_header_bytes": 4096,
                "unary_max_encoded_audio_bytes": 2048,
                "grpc_receive_max_bytes": 2048 + 65_536,
            },
            "max_riff_header_bytes",
        ),
        ({"grpc_bind": "no-port"}, "grpc_bind"),
        ({"grpc_bind": "127.0.0.1:0"}, "grpc_bind"),
        ({"grpc_bind": "127.0.0.1:65536"}, "grpc_bind"),
        ({"grpc_bind": "127.0.0.1:"}, "grpc_bind"),
        ({"grpc_bind": "127.0.0.1:http"}, "grpc_bind"),
    ],
)
async def test_relations_fail_startup_naming_the_option(
    clock, fake_factory, helpers, options, named
) -> None:
    failures = await helpers.startup_failures(
        clock, fake_factory, options=options
    )
    assert helpers.mentions(failures, named)


# @spec ING-EP-050
async def test_relations_hold_at_the_edges(
    clock, fake_factory, helpers
) -> None:
    plugin = helpers.make_frontend(
        clock, factories={"nemotron": fake_factory.constructor}
    )
    engine = helpers.engine_with(
        options={
            "unary_max_encoded_audio_bytes": 4096,
            "grpc_config_envelope_max_bytes": 512,
            "grpc_receive_max_bytes": 4608,
            "max_riff_header_bytes": 4096,
            "ws_receive_max_bytes": 2048,
            "ws_event_envelope_max_bytes": 2048,
        }
    )
    await helpers.initialize(plugin, engine)
    assert plugin.config.max_riff_header_bytes == 4096


# ------------------------------------------------------------ provenance


PROVENANCE = ("deployment_image", "pin", "precision_policy")


# @spec ING-EP-052
@pytest.mark.parametrize("option", [*PROVENANCE, "resampler_identifier"])
@pytest.mark.parametrize(
    "value", [" padded", "padded ", "\tpadded", "ngc://nim/model", "a://b"]
)
async def test_whitespace_or_scheme_provenance_fails_startup(
    clock, fake_factory, helpers, option, value
) -> None:
    failures = await helpers.startup_failures(
        clock, fake_factory, options={option: value}
    )
    assert helpers.mentions(failures, option)


# @spec ING-EP-052
@pytest.mark.parametrize(
    "option", ["deployment_image", "precision_policy", "resampler_identifier"]
)
@pytest.mark.parametrize("value", ["deadbeef", "DEADBEEF", "ab" * 32])
async def test_bare_hex_provenance_fails_startup_except_pin(
    clock, fake_factory, helpers, option, value
) -> None:
    failures = await helpers.startup_failures(
        clock, fake_factory, options={option: value}
    )
    assert helpers.mentions(failures, option)


# @spec ING-EP-052
@pytest.mark.parametrize(
    "options",
    [
        {"pin": "0123456789abcdef0123456789abcdef01234567"},
        {"pin": "DEADBEEF"},
        {"deployment_image": "registry.example/riva:1.0"},
        {"deployment_image": "sha256:0123abcd"},
        {"deployment_image": "deadbee"},
        {"precision_policy": "ab" * 32 + "c"},
    ],
)
def test_legitimate_provenance_shapes_are_accepted(options) -> None:
    config = parse(options)
    for option, value in options.items():
        assert getattr(config, option) == value


# @spec ING-EP-052, ING-EP-053
def test_deployment_provenance_is_built_from_the_four_fields() -> None:
    import dataclasses

    from vllm_riva_frontend.config import build_deployment_provenance

    config = parse({"deployment_image": "img:1", "pin": "abc1234def"})
    provenance = build_deployment_provenance(config)
    fields = (
        dataclasses.astuple(provenance)
        if dataclasses.is_dataclass(provenance)
        else tuple(vars(provenance).values())
    )
    assert sorted(map(str, fields)) == sorted(
        ["img:1", "abc1234def", "fp32-bringup-v1", "scipy-poly-v1"]
    )
