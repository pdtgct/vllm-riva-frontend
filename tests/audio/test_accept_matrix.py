"""The accept-matrix: four mono cells, everything else named and refused."""

from __future__ import annotations

import pytest
from support.riva import recognition_config

CELLS = {
    ("LINEAR_PCM", 16_000),
    ("LINEAR_PCM", 8_000),
    ("MULAW", 8_000),
    ("ALAW", 8_000),
}
SERVED = "nvidia/nemotron-asr-streaming-test"
LOCALES = frozenset({"en-US", "de-DE", "auto"})


def _validate(encoding, rate, channels=1):
    from vllm_riva_frontend.frontend import validate_format

    return validate_format(encoding, rate, channels)


def _riva_errors(**fields):
    from vllm_riva_frontend.grpc import validate_recognition_config

    return validate_recognition_config(
        recognition_config(**fields),
        served_model=SERVED,
        locales=LOCALES,
        default_locale="auto",
    )


# @spec ING-FE-016
def test_accept_matrix_is_exactly_the_four_mono_cells() -> None:
    from vllm_riva_frontend.frontend import ACCEPT_MATRIX

    # ``(encoding, sample_rate_hz, channels)``, mono only.
    assert frozenset((*cell, 1) for cell in CELLS) == ACCEPT_MATRIX


# @spec ING-FE-016
@pytest.mark.parametrize(("encoding", "rate"), sorted(CELLS))
def test_each_cell_is_accepted_mono(encoding, rate) -> None:
    assert _validate(encoding, rate, 1) is None


# @spec ING-FE-016, ING-ERR-003
def test_off_matrix_rate_names_the_rate() -> None:
    error = _validate("LINEAR_PCM", 44_100)
    assert error is not None
    assert error.fields == ("sample_rate_hz",)


# @spec ING-FE-016, ING-ERR-003
def test_non_mono_names_channels_even_for_a_valid_cell() -> None:
    error = _validate("LINEAR_PCM", 16_000, 2)
    assert error is not None
    assert error.fields == ("channels",)


# @spec ING-FE-016, ING-FE-014, ING-ERR-003
@pytest.mark.parametrize("encoding", ["FLAC", "OGGOPUS", "OPUS", "MP3"])
def test_compressed_encodings_are_refused_naming_the_encoding(
    encoding,
) -> None:
    error = _validate(encoding, 16_000)
    assert error is not None
    assert error.fields == ("encoding",)


# @spec ING-FE-014, ING-ERR-003
@pytest.mark.parametrize("encoding", ["MULAW", "ALAW"])
def test_16k_g711_is_refused_not_resampled(encoding) -> None:
    error = _validate(encoding, 16_000)
    assert error is not None
    # A known encoding at a rate off its cells names the rate.
    assert error.fields == ("sample_rate_hz",)


# @spec ING-FE-016, ING-FE-014
@pytest.mark.parametrize(
    ("fields", "named"),
    [
        ({"encoding": "FLAC"}, "encoding"),
        ({"encoding": "OGGOPUS"}, "encoding"),
        (
            {"encoding": "MULAW", "sample_rate_hertz": 16_000},
            "sample_rate_hertz",
        ),
        ({"sample_rate_hertz": 22_050}, "sample_rate_hertz"),
        ({"audio_channel_count": 2}, "audio_channel_count"),
    ],
)
def test_riva_servicer_refuses_off_matrix_audio_as_unsupported_format(
    fields, named
) -> None:
    (error,) = _riva_errors(**fields)
    assert error.code == "unsupported_format"
    assert error.fields == (named,)


# @spec ING-FE-016
@pytest.mark.parametrize(("encoding", "rate"), sorted(CELLS))
def test_riva_servicer_accepts_every_cell(encoding, rate) -> None:
    assert _riva_errors(encoding=encoding, sample_rate_hertz=rate) == []
