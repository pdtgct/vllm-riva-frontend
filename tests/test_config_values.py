"""Finite configuration and Riva/NIM disposition parity contracts."""

import json
from dataclasses import replace

import pytest

from vllm_riva_frontend.config import (
    DeploymentMetadata,
    FrontendConfig,
    build_deployment_provenance,
    dispositioned_fields,
    is_positive_finite,
    load_plugin_config,
    validate_frontend_config,
)


def _valid_config() -> FrontendConfig:
    return FrontendConfig(
        grpc_bind="127.0.0.1:50051",
        grpc_receive_max_bytes=1152,
        grpc_config_envelope_max_bytes=128,
        unary_max_encoded_audio_bytes=1024,
        unary_max_decoded_duration_seconds=60.0,
        max_riff_header_bytes=512,
        load_shed_max_sessions=4,
        pre_submit_max_samples=16000,
        preconfiguration_timeout=5.0,
        session_idle_timeout=60.0,
        session_finalization_timeout=30.0,
        session_cleanup_timeout=7.0,
        plugin_shutdown_grace=40.0,
        ws_receive_max_bytes=2048,
        ws_event_envelope_max_bytes=1024,
        http_multipart_envelope_max_bytes=1152,
        http_content_type_max_bytes=128,
        http_request_header_max_bytes=256,
        http_multipart_boundary_max_bytes=64,
        http_multipart_max_parts=4,
        http_multipart_max_header_bytes=128,
        http_text_field_max_bytes=128,
        http_request_timeout=60.0,
        grpc_keepalive_seconds=20.0,
        max_session_duration=600.0,
        resampler_identifier="scipy-poly-v1",
    )


# @spec ENV-MOD-003, ING-VEH-012
def test_all_timeout_and_resource_values_are_positive_finite() -> None:
    assert all(
        is_positive_finite(value)
        for value in vars(_valid_config()).values()
        if isinstance(value, (int, float))
    )
    assert not is_positive_finite(0)
    assert not is_positive_finite(float("inf"))


# @spec ENV-MOD-003, ING-GRPC-008, ING-HTTP-006, ING-NIMWS-010
def test_cross_field_bounds_validate_before_listener_bind() -> None:
    config = _valid_config()
    assert config.grpc_receive_max_bytes == (
        config.unary_max_encoded_audio_bytes
        + config.grpc_config_envelope_max_bytes
    )
    assert (
        config.http_multipart_envelope_max_bytes
        > config.unary_max_encoded_audio_bytes
    )
    assert config.ws_receive_max_bytes >= config.ws_event_envelope_max_bytes
    assert (
        config.http_content_type_max_bytes
        <= config.http_request_header_max_bytes
    )
    assert validate_frontend_config(config) is None


# @spec ENV-MOD-003, ING-GRPC-008, ING-NIMWS-010
def test_cross_field_mismatch_is_rejected_before_listener_bind() -> None:
    with pytest.raises(ValueError, match="grpc_receive_max_bytes"):
        validate_frontend_config(
            replace(_valid_config(), grpc_receive_max_bytes=1024)
        )


# @spec ENV-MOD-003, ING-FE-004
def test_grpc_bind_lifetime_values_and_resampler_pin_validate() -> None:
    assert validate_frontend_config(_valid_config()) is None
    with pytest.raises(ValueError, match="grpc_bind"):
        validate_frontend_config(replace(_valid_config(), grpc_bind="bad"))
    with pytest.raises(ValueError, match="resampler_identifier"):
        validate_frontend_config(
            replace(_valid_config(), resampler_identifier="other-resampler")
        )
    with pytest.raises(ValueError, match="max_session_duration"):
        validate_frontend_config(
            replace(_valid_config(), max_session_duration=0.0)
        )


# @spec ING-GRPC-005, ING-NIMWS-008
def test_grpc_and_nim_share_field_dispositions_without_silent_ignore() -> None:
    fields = {"language_code": "en-US", "speaker_diarization": {"enable": True}}
    assert dispositioned_fields(fields, dialect="grpc") == dispositioned_fields(
        fields, dialect="nim"
    )


# @spec ENV-MOD-003, ING-LIFE-005, ING-LIFE-013
# @spec ING-LIFE-014, ING-LIFE-015
def test_preconfiguration_and_cleanup_timeouts_are_distinct() -> None:
    config = _valid_config()
    assert (
        len(
            {
                config.preconfiguration_timeout,
                config.session_idle_timeout,
                config.session_finalization_timeout,
                config.session_cleanup_timeout,
            }
        )
        == 4
    )


# @spec ING-VEH-009, ING-VEH-012, ING-SHIM-002
def test_plugin_config_loads_identically_inline_and_from_path(tmp_path) -> None:
    frontend = _valid_config()
    raw = {
        **vars(frontend),
        "deployment_image": "sha256:test-image",
        "pin": "vllm==0.24.0",
        "precision_policy": "nemotron-asr-fp32-v1",
    }
    encoded = json.dumps(raw)
    path = tmp_path / "riva-frontend.json"
    path.write_text(encoded, encoding="utf-8")

    expected_metadata = DeploymentMetadata(
        image="sha256:test-image",
        pin="vllm==0.24.0",
        precision_policy="nemotron-asr-fp32-v1",
    )
    assert load_plugin_config(encoded) == (frontend, expected_metadata)
    assert load_plugin_config(str(path)) == (frontend, expected_metadata)


# @spec ING-VEH-009, ING-VEH-012
def test_plugin_config_rejects_unknown_and_invalid_explicit_fields() -> None:
    with pytest.raises(ValueError, match="unknown configuration fields"):
        load_plugin_config(
            json.dumps(
                {
                    **vars(_valid_config()),
                    "deployment_image": "image",
                    "pin": "pin",
                    "precision_policy": "policy",
                    "surprise": True,
                }
            )
        )
    with pytest.raises(ValueError, match="deployment_image"):
        load_plugin_config(
            json.dumps(
                {
                    **vars(_valid_config()),
                    "deployment_image": "",
                    "pin": "pin",
                    "precision_policy": "policy",
                }
            )
        )


# @spec ENV-MOD-003, ING-VEH-012
def test_every_field_may_be_omitted_but_null_is_never_omission() -> None:
    """D5: any field may be omitted (defaults fill it); null never can be.

    A default silently filling an *omitted* key is the zero-config
    contract; a default silently repairing an *explicit* null would not
    be -- ``load_plugin_config`` must keep rejecting the latter even for
    fields that now carry a declared default.
    """
    raw = {
        **vars(_valid_config()),
        "deployment_image": "image",
        "pin": "pin",
        "precision_policy": "policy",
    }
    for optional in (
        "grpc_keepalive_seconds",
        "resampler_identifier",
        "session_idle_timeout",
        "max_session_duration",
        "deployment_image",
    ):
        omitted = dict(raw)
        del omitted[optional]
        frontend, metadata = load_plugin_config(json.dumps(omitted))
        if optional == "grpc_keepalive_seconds":
            assert frontend.grpc_keepalive_seconds is None
        elif optional == "resampler_identifier":
            assert frontend.resampler_identifier == "scipy-poly-v1"
        elif optional == "session_idle_timeout":
            assert frontend.session_idle_timeout == 60.0
        elif optional == "max_session_duration":
            assert frontend.max_session_duration == 600.0
        else:
            assert metadata.image == (
                "unspecified"
            )

    for name in ("grpc_keepalive_seconds", "deployment_image", "pin"):
        explicit_null = dict(raw)
        explicit_null[name] = None
        with pytest.raises(ValueError, match="must not be null"):
            load_plugin_config(json.dumps(explicit_null))


# @spec ENV-MOD-003, ING-VEH-009, ING-VEH-012
def test_zero_config_resolves_the_full_qualified_default_profile() -> None:
    """(a) No configuration at all still constructs the deployed profile."""
    frontend, metadata = load_plugin_config(None)
    assert frontend.grpc_bind == "0.0.0.0:50051"
    assert frontend.grpc_keepalive_seconds is None
    assert frontend.resampler_identifier == "scipy-poly-v1"
    assert frontend.session_idle_timeout == 60.0
    assert frontend.max_session_duration == 600.0
    assert metadata == DeploymentMetadata(
        image="unspecified",
        pin=(
            "unspecified"
        ),
        precision_policy="fp32-bringup-v1",
    )
    assert validate_frontend_config(frontend) is None

    # A blank (whitespace-only) value is treated the same as no value.
    blank_frontend, blank_metadata = load_plugin_config("   ")
    assert blank_frontend == frontend
    assert blank_metadata == metadata


# @spec ENV-MOD-003, ING-VEH-009, ING-VEH-012
def test_inline_json_overrides_exactly_the_given_keys() -> None:
    """(b) A partial inline override changes only the given keys."""
    frontend, metadata = load_plugin_config(
        json.dumps(
            {"grpc_bind": "127.0.0.1:60051", "load_shed_max_sessions": 8}
        )
    )
    assert frontend.grpc_bind == "127.0.0.1:60051"
    assert frontend.load_shed_max_sessions == 8
    # Everything else still resolves to the default profile.
    assert frontend.session_idle_timeout == 60.0
    assert frontend.plugin_shutdown_grace == 240.0
    assert frontend.max_session_duration == 600.0
    assert metadata.pin == (
        "unspecified"
    )


# @spec ING-VEH-009, ING-VEH-012
def test_malformed_inline_json_names_the_plugin_key_and_does_not_fall_back(
    tmp_path,
) -> None:
    """(c) A malformed inline value fails as inline JSON, not as a path."""
    # The candidate is not readable as a path either, so a fall-back
    # would raise a *different* ("... path is not readable") error; the
    # inline-specific error proves no fall-back happened.
    with pytest.raises(
        ValueError,
        match=r"riva_frontend application-plugin inline config is not "
        r"valid JSON",
    ):
        load_plugin_config("{not valid json")

    # The same distinction holds for a genuinely malformed config file.
    bad_path = tmp_path / "riva-frontend.json"
    bad_path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(
        ValueError,
        match=r"riva_frontend application-plugin config file is not "
        r"valid JSON",
    ):
        load_plugin_config(str(bad_path))


# @spec ING-VEH-009, ING-VEH-012
def test_path_shaped_value_still_loads_from_file(tmp_path) -> None:
    """(d) A path-shaped value (not starting with '{') still reads a file."""
    frontend, metadata = load_plugin_config(
        json.dumps({"load_shed_max_sessions": 5})
    )
    path = tmp_path / "riva-frontend.json"
    path.write_text(
        json.dumps({"load_shed_max_sessions": 5}), encoding="utf-8"
    )
    from_path_frontend, from_path_metadata = load_plugin_config(str(path))
    assert from_path_frontend == frontend
    assert from_path_metadata == metadata


# @spec ENV-MOD-003, ING-VEH-017
def test_shutdown_grace_covers_normal_finalization_then_cleanup() -> None:
    config = _valid_config()
    with pytest.raises(ValueError, match="session_finalization_timeout"):
        validate_frontend_config(
            replace(
                config,
                plugin_shutdown_grace=(
                    config.session_finalization_timeout
                    + config.session_cleanup_timeout
                    - 0.1
                ),
            )
        )


# @spec ING-GRPC-005, ING-NIMWS-008
def test_unsupported_sections_share_capability_disposition() -> None:
    values = {
        "diarization_config": {"enable_speaker_diarization": True},
        "speech_contexts": ["term"],
    }
    assert dispositioned_fields(values, dialect="grpc") == {
        "diarization_config": "unsupported_capability",
        "speech_contexts": "unsupported_capability",
    }


# @spec ING-SHIM-002
def test_deployment_provenance_is_built_by_allowlist_not_by_copy() -> None:
    """The only four fields this function can ever emit, and no others.

    Unlike a filter over an arbitrary mapping, there is no input shape
    here that could smuggle a fifth key through: the signature accepts
    exactly a ``DeploymentMetadata`` and a resampler string.
    """
    metadata = DeploymentMetadata(
        image="sha256:test",
        pin="vllm==0.24.0",
        precision_policy="nemotron-asr-fp32-v1",
    )
    provenance = build_deployment_provenance(
        metadata, resampler_identifier="scipy-poly-v1"
    )
    assert provenance == {
        "image": "sha256:test",
        "pin": "vllm==0.24.0",
        "precision_policy": "nemotron-asr-fp32-v1",
        "resampler": "scipy-poly-v1",
    }
    assert set(provenance) == {"image", "pin", "precision_policy", "resampler"}


# @spec ING-SHIM-002
def test_configuration_ingestion_rejects_a_scheme_smuggled_image() -> None:
    """A NIM registry reference must fail startup, not reach /v1/metadata.

    Allowlisting the four provenance *keys* (see
    test_deployment_provenance_is_built_by_allowlist_not_by_copy) does not
    by itself constrain their *values*; this pins that the value itself
    is schema-validated where it originates, at configuration ingestion.
    """
    raw = {
        **vars(_valid_config()),
        "deployment_image": "ngc://nim/model",
        "pin": "vllm==0.24.0",
        "precision_policy": "nemotron-asr-fp32-v1",
    }
    with pytest.raises(ValueError, match="deployment_image"):
        load_plugin_config(json.dumps(raw))


# @spec ING-SHIM-002
def test_configuration_ingestion_rejects_a_bare_hex_pin() -> None:
    """A NIM profile hash must fail startup no matter which field carries it."""
    raw = {
        **vars(_valid_config()),
        "deployment_image": "sha256:test",
        "pin": "deadbeef",
        "precision_policy": "nemotron-asr-fp32-v1",
    }
    with pytest.raises(ValueError, match="pin"):
        load_plugin_config(json.dumps(raw))


# @spec ING-SHIM-002
def test_configuration_ingestion_rejects_uppercase_bare_hex() -> None:
    """Hex case must not bypass the bare-hash rejection."""
    raw = {
        **vars(_valid_config()),
        "deployment_image": "sha256:test",
        "pin": "DEADBEEF",
        "precision_policy": "nemotron-asr-fp32-v1",
    }
    with pytest.raises(ValueError, match="pin"):
        load_plugin_config(json.dumps(raw))


# @spec ING-SHIM-002
def test_configuration_ingestion_rejects_whitespace_padded_values() -> None:
    """Padding must not bypass validation; whitespace itself is rejected."""
    raw = {
        **vars(_valid_config()),
        "deployment_image": "sha256:test",
        "pin": " deadbeef ",
        "precision_policy": "nemotron-asr-fp32-v1",
    }
    with pytest.raises(ValueError, match="pin"):
        load_plugin_config(json.dumps(raw))


# @spec ING-SHIM-002
def test_configuration_ingestion_accepts_the_real_legit_provenance_shapes() -> (
    None
):
    """The rule that rejects ngc:// and bare hex does not over-restrict.

    A real ``sha256:<hex>`` image digest and a real pin string like
    ``vllm==0.24.0`` are exactly the shapes production configuration
    uses, and must still load.
    """
    raw = {
        **vars(_valid_config()),
        "deployment_image": "sha256:test",
        "pin": "vllm==0.24.0",
        "precision_policy": "nemotron-asr-fp32-v1",
    }
    frontend, metadata = load_plugin_config(json.dumps(raw))
    assert frontend == _valid_config()
    assert metadata == DeploymentMetadata(
        image="sha256:test",
        pin="vllm==0.24.0",
        precision_policy="nemotron-asr-fp32-v1",
    )
