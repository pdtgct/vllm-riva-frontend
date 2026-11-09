"""``RecognitionConfig`` field disposition on the first streaming message."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import grpc
import pytest
from riva.client.proto import riva_asr_pb2 as rasr
from support.audio import pcm16_bytes, pcm16_sine
from support.fake_session import FakeSessionFactory
from support.riva import (
    ALAW,
    FLAC,
    LINEAR_PCM,
    MULAW,
    OGGOPUS,
    audio_request,
    config_request,
    recognition_config,
)

from .conftest import (
    SERVED,
    first_message_outcome,
    opened,
    opens,
    serving,
    start_stream,
    until,
)

REPO = Path(__file__).resolve().parents[2]
INVALID = grpc.StatusCode.INVALID_ARGUMENT
UNIMPLEMENTED = grpc.StatusCode.UNIMPLEMENTED

DESCRIPTOR_FIELDS = {
    field.name for field in rasr.RecognitionConfig.DESCRIPTOR.fields
}


def _endpointing(**fields: Any) -> rasr.EndpointingConfig:
    return rasr.EndpointingConfig(**fields)


#: Field -> configurations whose first message must open the session.
ACCEPTED: dict[str, list[dict[str, Any]]] = {
    "encoding": [
        {"encoding": LINEAR_PCM, "sample_rate_hertz": 16_000},
        {"encoding": LINEAR_PCM, "sample_rate_hertz": 8_000},
        {"encoding": MULAW, "sample_rate_hertz": 8_000},
        {"encoding": ALAW, "sample_rate_hertz": 8_000},
    ],
    "sample_rate_hertz": [
        {"sample_rate_hertz": 16_000},
        {"sample_rate_hertz": 8_000},
    ],
    "language_code": [{"language_code": "en-US"}, {"language_code": ""}],
    "max_alternatives": [{"max_alternatives": 0}, {"max_alternatives": 1}],
    "audio_channel_count": [
        {"audio_channel_count": 0},
        {"audio_channel_count": 1},
    ],
    "enable_automatic_punctuation": [
        {"enable_automatic_punctuation": True},
        {"enable_automatic_punctuation": False},
    ],
    "verbatim_transcripts": [
        {"verbatim_transcripts": True},
        {"verbatim_transcripts": False},
    ],
    "profanity_filter": [{"profanity_filter": False}],
    "enable_word_time_offsets": [{"enable_word_time_offsets": False}],
    "enable_separate_recognition_per_channel": [
        {"enable_separate_recognition_per_channel": False}
    ],
    "model": [{"model": ""}, {"model": SERVED}],
    "speech_contexts": [{"speech_contexts": []}],
    "diarization_config": [
        {"diarization_config": rasr.SpeakerDiarizationConfig()}
    ],
    "custom_configuration": [{"custom_configuration": {}}],
    "endpointing_config": [{"endpointing_config": _endpointing()}],
}

#: (field, configuration, status, catalog code); the field is named
REJECTED: list[tuple[str, dict[str, Any], grpc.StatusCode, str]] = [
    ("encoding", {"encoding": FLAC}, INVALID, "unsupported_format"),
    ("encoding", {"encoding": OGGOPUS}, INVALID, "unsupported_format"),
    (
        "sample_rate_hertz",
        {"sample_rate_hertz": 0},
        INVALID,
        "unsupported_format",
    ),
    (
        "sample_rate_hertz",
        {"sample_rate_hertz": 44_100},
        INVALID,
        "unsupported_format",
    ),
    (
        "sample_rate_hertz",
        {"encoding": MULAW, "sample_rate_hertz": 16_000},
        INVALID,
        "unsupported_format",
    ),
    ("language_code", {"language_code": "fr-FR"}, INVALID, "unknown_locale"),
    (
        "max_alternatives",
        {"max_alternatives": 2},
        INVALID,
        "invalid_config_field",
    ),
    (
        "audio_channel_count",
        {"audio_channel_count": 2},
        INVALID,
        "unsupported_format",
    ),
    (
        "profanity_filter",
        {"profanity_filter": True},
        INVALID,
        "invalid_config_field",
    ),
    (
        "enable_word_time_offsets",
        {"enable_word_time_offsets": True},
        INVALID,
        "invalid_config_field",
    ),
    (
        "enable_separate_recognition_per_channel",
        {"enable_separate_recognition_per_channel": True},
        INVALID,
        "invalid_config_field",
    ),
    ("model", {"model": "someone/else"}, INVALID, "invalid_config_field"),
    (
        "speech_contexts",
        {"speech_contexts": [rasr.SpeechContext(phrases=["vLLM"], boost=4.0)]},
        UNIMPLEMENTED,
        "unsupported_capability",
    ),
    (
        "diarization_config",
        {
            "diarization_config": rasr.SpeakerDiarizationConfig(
                enable_speaker_diarization=True
            )
        },
        UNIMPLEMENTED,
        "unsupported_capability",
    ),
    (
        "diarization_config",
        {
            "diarization_config": rasr.SpeakerDiarizationConfig(
                max_speaker_count=2
            )
        },
        UNIMPLEMENTED,
        "unsupported_capability",
    ),
    (
        "custom_configuration",
        {"custom_configuration": {"key": "value"}},
        INVALID,
        "invalid_config_field",
    ),
    *[
        (
            "endpointing_config",
            {"endpointing_config": _endpointing(**{name: value})},
            UNIMPLEMENTED,
            "unsupported_capability",
        )
        for name, value in (
            ("start_history", 100),
            ("start_threshold", 0.5),
            ("stop_history", 200),
            ("stop_threshold", 0.25),
            ("stop_history_eou", 300),
            ("stop_threshold_eou", 0.75),
        )
    ],
]


def _config(case: dict[str, Any]) -> rasr.RecognitionConfig:
    case = dict(case)
    encoding = case.pop("encoding", LINEAR_PCM)
    rate = case.pop(
        "sample_rate_hertz", 16_000 if encoding == LINEAR_PCM else 8_000
    )
    return recognition_config(encoding=encoding, sample_rate_hertz=rate, **case)


def _id(value: Any) -> str:
    return repr(value)[:40]


# @spec ING-GRPC-005
# @spec ING-GRPC-025
def test_the_disposition_cases_here_cover_every_descriptor_field():
    covered = set(ACCEPTED) | {case[0] for case in REJECTED}
    assert covered == DESCRIPTOR_FIELDS


# @spec ING-GRPC-025
def test_every_recognition_config_field_of_the_pinned_stubs_has_a_disposition():
    from vllm_riva_frontend.grpc import FIELD_DISPOSITIONS

    missing = DESCRIPTOR_FIELDS - set(FIELD_DISPOSITIONS)
    unknown = set(FIELD_DISPOSITIONS) - DESCRIPTOR_FIELDS
    assert not missing, f"fields without a disposition: {sorted(missing)}"
    assert not unknown, f"dispositions for no field: {sorted(unknown)}"
    for name in DESCRIPTOR_FIELDS:
        assert FIELD_DISPOSITIONS[name] is not None, name


@pytest.mark.parametrize(
    ("name", "case"),
    [(name, case) for name, cases in ACCEPTED.items() for case in cases],
    ids=_id,
)
# @spec ING-GRPC-005
# @spec ING-GRPC-006
# @spec ING-GRPC-017
# @spec ING-GRPC-024
# @spec ING-EOU-003
async def test_honoured_and_model_intrinsic_values_open_the_session(
    served, name, case
):
    call = await first_message_outcome(served, config_request(_config(case)))
    assert call.status == grpc.StatusCode.OK, (name, call.details)
    assert len(opens(served)) == 1


@pytest.mark.parametrize(("name", "case", "status", "code"), REJECTED, ids=_id)
# @spec ING-GRPC-005
# @spec ING-GRPC-017
# @spec ING-GRPC-018
# @spec ING-GRPC-022
# @spec ING-GRPC-023
# @spec ING-GRPC-024
# @spec ING-EOU-003
async def test_rejected_values_name_the_field_before_open(
    served, name, case, status, code
):
    call = await first_message_outcome(served, config_request(_config(case)))
    call.assert_rejected(status, code, name)
    assert opens(served) == []


# @spec ING-GRPC-005
def test_validation_collects_one_error_per_rejected_field():
    from vllm_riva_frontend.grpc import validate_recognition_config

    config = recognition_config(
        profanity_filter=True,
        speech_contexts=[rasr.SpeechContext(phrases=["x"])],
        custom_configuration={"k": "v"},
        model="someone/else",
    )
    errors = validate_recognition_config(
        config,
        served_model=SERVED,
        locales=frozenset({"en-US", "auto"}),
        default_locale="auto",
    )
    named = {name for error in errors for name in error.fields}
    assert {
        "profanity_filter",
        "speech_contexts",
        "custom_configuration",
        "model",
    } <= named
    codes = {error.code for error in errors}
    assert {"invalid_config_field", "unsupported_capability"} <= codes


# @spec ING-GRPC-005
# @spec ING-GRPC-022
# @spec ING-GRPC-023
def test_rejection_status_follows_the_capability_class():
    from vllm_riva_frontend.grpc import validate_recognition_config

    def errors_for(**fields: Any):
        return validate_recognition_config(
            recognition_config(**fields),
            served_model=SERVED,
            locales=frozenset({"en-US", "auto"}),
            default_locale="auto",
        )

    (offsets,) = errors_for(enable_word_time_offsets=True)
    assert offsets.code == "invalid_config_field"
    assert "enable_word_time_offsets" in offsets.fields
    (alternatives,) = errors_for(max_alternatives=2)
    assert alternatives.code == "invalid_config_field"
    assert "max_alternatives" in alternatives.fields
    (channels,) = errors_for(audio_channel_count=2)
    assert channels.code == "unsupported_format"
    assert "audio_channel_count" in channels.fields
    (boost,) = errors_for(speech_contexts=[rasr.SpeechContext(phrases=["x"])])
    assert boost.code == "unsupported_capability"
    assert "speech_contexts" in boost.fields


# @spec ING-GRPC-006
async def test_punctuation_and_verbatim_do_not_alter_output(clock):
    finals = set()
    for punctuation in (True, False):
        for verbatim in (True, False):
            factory = FakeSessionFactory(final_text="Hello, world")
            async with serving(clock=clock, factory=factory) as served:
                call = start_stream(
                    served,
                    [
                        config_request(
                            recognition_config(
                                enable_automatic_punctuation=punctuation,
                                verbatim_transcripts=verbatim,
                            )
                        )
                    ],
                )
                lease = await opened(served, call)
                call.push(audio_request(pcm16_bytes(pcm16_sine(0.02))))
                await until(lambda lease=lease: lease.accepted, what="feed")
                call.half_close()
                await call.wait()
                assert call.status == grpc.StatusCode.OK, call.details
                assert len(call.finals) == 1
                finals.add(call.finals[0])
    assert finals == {"Hello, world"}


# @spec ING-GRPC-006
def test_user_documentation_states_the_punctuation_and_verbatim_deviation():
    texts = [
        path.read_text(encoding="utf-8")
        for path in [REPO / "README.md", *sorted((REPO / "docs").glob("*.md"))]
        if path.is_file()
    ]
    stating = [
        text
        for text in texts
        if "enable_automatic_punctuation" in text
        and "verbatim_transcripts" in text
    ]
    assert stating, "no user-facing document names both fields"


# @spec ING-GRPC-024
# @spec ING-CORE-015
async def test_empty_language_takes_the_default_locale(served):
    call = await first_message_outcome(
        served, config_request(recognition_config(language_code=""))
    )
    assert call.status == grpc.StatusCode.OK, call.details
    assert opens(served) == [{"cadence": "560ms", "locale": "auto"}]


# @spec ING-GRPC-024
async def test_any_factory_locale_is_accepted(served):
    for locale in ("en-US", "de-DE", "auto"):
        call = start_stream(
            served, [config_request(recognition_config(language_code=locale))]
        )
        lease = await opened(served, call, index=len(served.factory.leases))
        call.half_close()
        await call.wait()
        assert lease.locale == locale


# @spec ING-GRPC-024
# @spec ING-CORE-015
async def test_empty_language_without_a_default_locale_is_rejected(clock):
    factory = FakeSessionFactory(default_locale=None)
    async with serving(clock=clock, factory=factory) as served:
        call = await first_message_outcome(
            served, config_request(recognition_config(language_code=""))
        )
        call.assert_rejected(INVALID, "invalid_config_field", "language_code")
        assert opens(served) == []


# @spec ING-GRPC-017
async def test_zero_channels_and_alternatives_normalize_to_one(served):
    call = await first_message_outcome(
        served,
        config_request(
            recognition_config(audio_channel_count=0, max_alternatives=0)
        ),
    )
    assert call.status == grpc.StatusCode.OK, call.details
    assert len(opens(served)) == 1


# @spec ING-GRPC-016
async def test_runtime_config_rider_is_rejected_before_open(served):
    call = await first_message_outcome(
        served, config_request(runtime_config={"anything": "value"})
    )
    call.assert_rejected(INVALID, "invalid_config_field", "runtime_config")
    assert opens(served) == []


# @spec ING-GRPC-016
async def test_runtime_config_rider_on_a_later_message_ends_the_stream(served):
    call = start_stream(served, [config_request()])
    lease = await opened(served, call)
    rider = audio_request(pcm16_bytes(pcm16_sine(0.1)))
    rider.runtime_config.update({"anything": "value"})
    call.push(rider)
    await call.wait()
    call.assert_rejected(INVALID, "invalid_config_field", "runtime_config")
    # Terminal after ``open``: the open lease is aborted and released, and
    # the rider's audio is not fed.
    assert lease.names() == ["abort", "release"]
    assert lease.fed == []


# @spec ING-GRPC-016
async def test_empty_runtime_config_is_not_a_rejection(served):
    call = await first_message_outcome(
        served, config_request(runtime_config={})
    )
    assert call.status == grpc.StatusCode.OK, call.details
    assert len(opens(served)) == 1
