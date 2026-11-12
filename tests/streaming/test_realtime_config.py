"""Realtime session configuration: format, deferred header, field rules."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
from support.audio import (
    float_from_pcm16,
    g711_bytes,
    pcm16_ramp,
    wav_bytes,
    wav_pcm16,
)
from support.fake_session import FakeSessionFactory
from support.realtime import (
    RealtimeClient,
    append_event,
    default_session,
    error_code,
    error_fields,
    update_event,
)

from .conftest import (
    SERVED,
    lease_names,
    opens,
    served_lease,
    serving,
)

DONE = {"type": "input_audio_buffer.done"}
SAMPLES = pcm16_ramp(1600, start=-800, step=1)


async def _connected(served) -> RealtimeClient:
    client = RealtimeClient(served.app, served.clock)
    assert await client.connect()
    assert (await client.next())["type"] == "conversation.created"
    return client


async def _rejections(served, session: dict[str, Any]) -> list[dict[str, Any]]:
    """Send ``session`` first, then a corrected one; return the errors.

    The connection must survive the rejection and the corrected update must
    configure (ING-ERR-034); nothing may open before it.
    """
    client = await _connected(served)
    await client.send(update_event(session))
    await client.send(update_event(default_session()))
    errors = []
    while True:
        event = await client.next()
        if event["type"] == "transcription_session.updated":
            break
        assert event["type"] == "error", event
        errors.append(event)
        assert opens(served) == []
    assert len(opens(served)) == 1
    return errors


def _with(path: str, value: Any) -> dict[str, Any]:
    session = default_session()
    target = session
    keys = path.split(".")
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    return session


ACCEPTED_CELLS = [
    ("pcm16", 16_000, 1),
    ("pcm16", 8_000, 1),
    ("g711_ulaw", 8_000, 1),
    ("g711_alaw", 8_000, 1),
]
# The cell and the one field the rejection names (implementation layout,
# Shapes: Realtime events): the rate for a known format at a rate off its
# cells, the channel count for non-mono, the format for an unknown name.
REJECTED_CELLS = [
    (("g711_ulaw", 16_000, 1), "sample_rate_hz"),
    (("g711_alaw", 16_000, 1), "sample_rate_hz"),
    (("pcm16", 16_000, 2), "num_channels"),
    (("pcm16", 8_000, 2), "num_channels"),
    (("pcm16", 44_100, 1), "sample_rate_hz"),
    (("pcm24", 16_000, 1), "input_audio_format"),
    (("opus", 16_000, 1), "input_audio_format"),
]


def _cell(fmt: str, rate: int, channels: int) -> dict[str, Any]:
    return default_session(
        input_audio_format=fmt,
        input_audio_params={"sample_rate_hz": rate, "num_channels": channels},
    )


@pytest.mark.parametrize("cell", ACCEPTED_CELLS, ids=str)
# @spec ING-NIMWS-007
async def test_accept_matrix_cells_configure(served, cell):
    client = RealtimeClient(served.app, served.clock)
    await client.open_session(_cell(*cell))
    assert len(opens(served)) == 1


@pytest.mark.parametrize(("cell", "field"), REJECTED_CELLS, ids=str)
# @spec ING-NIMWS-007
# @spec ING-ERR-034
async def test_off_matrix_cells_are_unsupported_format_naming_the_fields(
    served, cell, field
):
    errors = await _rejections(served, _cell(*cell))
    assert [error_code(e) for e in errors] == ["unsupported_format"]
    assert error_fields(errors[0]) == [field], errors[0]


# ------------------------------------------------------ deferred header


def _none(rate: int = 16_000, channels: int = 1) -> dict[str, Any]:
    return default_session(
        input_audio_format="none",
        input_audio_params={"sample_rate_hz": rate, "num_channels": channels},
    )


# @spec ING-NIMWS-009
async def test_none_format_resolves_and_strips_the_header_after_open(served):
    client = RealtimeClient(served.app, served.clock)
    await client.open_session(_none())
    lease = served_lease(served)
    data = wav_pcm16(SAMPLES)
    await client.send(append_event(data[:30]))
    await client.send({"type": "input_audio_buffer.commit"})
    await client.until_type("input_audio_buffer.committed")
    assert "feed" not in lease_names(served)
    await client.send(append_event(data[30:]))
    await client.send(DONE)
    before = await client.assert_closes(1000)
    assert not [e for e in before if e["type"] == "error"]
    np.testing.assert_array_equal(lease.fed_samples, float_from_pcm16(SAMPLES))


# @spec ING-NIMWS-009
async def test_none_format_resolves_a_g711_header(served):
    client = RealtimeClient(served.app, served.clock)
    await client.open_session(_none(rate=8_000))
    lease = served_lease(served)
    payload = g711_bytes(SAMPLES[:800], "alaw")
    await client.send(
        append_event(wav_bytes(payload, encoding="alaw", rate=8_000))
    )
    await client.send(DONE)
    await client.assert_closes(1000)
    assert sum(lease.accepted) > 0


@pytest.mark.parametrize(
    ("declared", "header", "field"),
    [
        ((16_000, 1), {"rate": 8_000}, "sample_rate_hz"),
        ((16_000, 1), {"rate": 44_100}, "sample_rate_hz"),
        ((16_000, 1), {"channels": 2}, "num_channels"),
    ],
    ids=str,
)
# @spec ING-NIMWS-009
async def test_none_format_mismatch_or_off_matrix_header_fails_naming_fields(
    served, declared, header, field
):
    client = RealtimeClient(served.app, served.clock)
    await client.open_session(_none(*declared))
    channels = header.get("channels", 1)
    data = wav_pcm16(np.repeat(SAMPLES, channels), **header)
    await client.send(append_event(data))
    before = await client.assert_closes(1008)
    errors = [e for e in before if e["type"] == "error"]
    assert [error_code(e) for e in errors] == ["unsupported_format"]
    assert error_fields(errors[0]) == [field], errors[0]
    assert "feed" not in lease_names(served)


# @spec ING-NIMWS-009
async def test_a_declared_format_is_never_overridden_by_a_header(served):
    client = RealtimeClient(served.app, served.clock)
    await client.open_session()
    lease = served_lease(served)
    data = wav_pcm16(SAMPLES)
    await client.send(append_event(data))
    await client.send(DONE)
    await client.assert_closes(1000)
    expected = float_from_pcm16(np.frombuffer(data, dtype="<i2"))
    np.testing.assert_array_equal(lease.fed_samples, expected)


# @spec ING-NIMWS-009
def test_none_is_not_an_honoured_openai_format():
    from vllm_riva_frontend.frontend import ACCEPT_MATRIX

    assert "'none'" not in repr(ACCEPT_MATRIX)
    assert '"none"' not in repr(ACCEPT_MATRIX)


# ------------------------------------------------------ field rules

REJECTED_FIELDS: list[tuple[str, Any, str, tuple[str, ...]]] = [
    (
        "input_audio_transcription.prompt",
        "be terse",
        "invalid_config_field",
        ("prompt",),
    ),
    (
        "modalities",
        ["text", "audio"],
        "invalid_config_field",
        ("modalities",),
    ),
    (
        "input_audio_transcription.model",
        "someone/else",
        "invalid_config_field",
        ("model",),
    ),
    (
        "input_audio_transcription.language",
        "fr-FR",
        "unknown_locale",
        ("language",),
    ),
    (
        "recognition_config.enable_profanity_filter",
        True,
        "invalid_config_field",
        ("enable_profanity_filter",),
    ),
    (
        "recognition_config.enable_word_time_offsets",
        True,
        "invalid_config_field",
        ("enable_word_time_offsets",),
    ),
    (
        "recognition_config.max_alternatives",
        2,
        "invalid_config_field",
        ("max_alternatives",),
    ),
    (
        "speaker_diarization.enable_speaker_diarization",
        True,
        "unsupported_capability",
        ("speaker_diarization",),
    ),
    (
        "word_boosting",
        {
            "enable_word_boosting": True,
            "word_boosting_list": [{"phrases": ["vLLM"], "boost": 4.0}],
        },
        "unsupported_capability",
        ("word_boosting",),
    ),
    (
        "endpointing_config",
        {"start_history": 100},
        "unsupported_capability",
        ("endpointing_config",),
    ),
    (
        "endpointing_config",
        {"stop_threshold_eou": 0.5},
        "unsupported_capability",
        ("endpointing_config",),
    ),
    (
        "custom_configuration",
        {"key": "value"},
        "invalid_config_field",
        ("custom_configuration",),
    ),
    ("surprise_option", 1, "invalid_config_field", ("surprise_option",)),
    (
        "recognition_config.surprise_flag",
        True,
        "invalid_config_field",
        ("surprise_flag",),
    ),
]


@pytest.mark.parametrize(
    ("path", "value", "code", "names"),
    REJECTED_FIELDS,
    ids=lambda v: str(v)[:30],
)
# @spec ING-NIMWS-022
# @spec ING-NIMWS-014
# @spec ING-EOU-003
async def test_rejected_session_fields_name_the_dialect_field(
    served, path, value, code, names
):
    errors = await _rejections(served, _with(path, value))
    assert len(errors) == 1, errors
    assert error_code(errors[0]) == code, errors[0]
    assert error_fields(errors[0]) == list(names), errors[0]


ACCEPTED_FIELDS: list[tuple[str, Any]] = [
    ("id", "sess_123"),
    ("object", "realtime.transcription_session"),
    ("client_secret", {"value": "ignored", "expires_at": 0}),
    ("provenance", {"anything": "echoed"}),
    ("input_audio_transcription.model", ""),
    ("input_audio_transcription.model", SERVED),
    ("input_audio_transcription.prompt", ""),
    ("recognition_config.enable_automatic_punctuation", True),
    ("recognition_config.enable_verbatim_transcripts", True),
    ("speaker_diarization.max_speaker_count", 0),
    ("word_boosting.word_boosting_list", []),
    ("endpointing_config", {}),
]


@pytest.mark.parametrize(("path", "value"), ACCEPTED_FIELDS, ids=str)
# @spec ING-NIMWS-022
async def test_echo_keys_and_default_shaped_sections_configure(
    served, path, value
):
    client = RealtimeClient(served.app, served.clock)
    await client.open_session(_with(path, value))
    assert len(opens(served)) == 1


# @spec ING-NIMWS-022
async def test_initial_configuration_is_all_or_nothing(served):
    session = _with("input_audio_transcription.prompt", "terse")
    session["custom_configuration"] = {"k": "v"}
    session["recognition_config"]["enable_profanity_filter"] = True
    errors = await _rejections(served, session)
    assert len(errors) == 3, errors
    for name in ("prompt", "custom_configuration", "enable_profanity_filter"):
        assert any(error_fields(e) == [name] for e in errors), (name, errors)


# @spec ING-NIMWS-022
# @spec ING-NIMWS-013
async def test_default_object_round_trips_and_needs_a_locale_without_default(
    clock,
):
    from vllm_riva_frontend.nim_ws import bootstrap_session

    async with serving(clock=clock) as served:
        client = RealtimeClient(served.app, served.clock)
        await client.open_session(
            copy.deepcopy(bootstrap_session(SERVED, "auto"))
        )
        assert opens(served) == [{"cadence": "560ms", "locale": "auto"}]

    factory = FakeSessionFactory(default_locale=None)
    async with serving(clock=clock, factory=factory) as served:
        errors = await _rejections(served, bootstrap_session(SERVED, None))
        assert len(errors) == 1
        assert error_fields(errors[0]) == ["language"], errors[0]
        assert opens(served)[0]["locale"] == "en-US"


# @spec ING-NIMWS-022
def test_validate_session_update_reports_every_field():
    from vllm_riva_frontend.nim_ws import validate_session_update

    session = _with("input_audio_transcription.prompt", "terse")
    session["modalities"] = ["audio"]
    errors = validate_session_update(
        session,
        served_model=SERVED,
        locales=frozenset({"en-US", "auto"}),
        default_locale="auto",
        first=True,
    )
    named = {name for error in errors for name in error.fields}
    assert named == {"prompt", "modalities"}
    assert (
        validate_session_update(
            default_session(),
            served_model=SERVED,
            locales=frozenset({"en-US", "auto"}),
            default_locale="auto",
            first=True,
        )
        == []
    )
