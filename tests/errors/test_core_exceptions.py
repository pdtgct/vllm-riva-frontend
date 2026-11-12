"""Core's raised exceptions and returned ``ErrorResponse`` for ``Recognize``.

The size-limit and decoder cases are produced by running core's own code
(``read_upload_with_limit``, ``load_audio`` and the transcription service's
decode step), so a change of core's published types or message texts fails
these tests rather than silently changing the projection.
"""

from __future__ import annotations

import asyncio
import io
from http import HTTPStatus
from types import SimpleNamespace

import pytest
from support.audio import pcm16_sine, wav_pcm16
from support.fake_transcription import error_response


def _classify(exc: BaseException):
    from vllm_riva_frontend.errors import classify_core_exception

    return classify_core_exception(exc)


def _wav() -> bytes:
    return wav_pcm16(pcm16_sine(0.5))


def _core_load_error(backend: str, **limits) -> ValueError:
    """The ``ValueError`` core's ``load_audio`` raises for ``limits``."""
    from vllm.multimodal.media.audio import load_audio

    try:
        load_audio(io.BytesIO(_wav()), sr=None, backend=backend, **limits)
    except ValueError as exc:
        return exc
    except Exception as exc:  # the backend is unusable here
        pytest.skip(f"audio backend {backend} unavailable: {exc!r}")
    pytest.fail(f"core accepted the audio under {limits}")


def _serving_decode_error(**overrides) -> ValueError:
    """Run the transcription service's own decode step on ``overrides``."""
    from vllm.entrypoints.speech_to_text.base.serving import (
        SpeechToTextBaseServing,
    )

    fake = SimpleNamespace(
        asr_config=SimpleNamespace(
            audio_resample_method=None,
            sample_rate=16_000,
            allow_audio_chunking=False,
            max_audio_clip_s=None,
        ),
        max_audio_decode_duration_s=overrides.pop("max_duration_s", None),
        max_audio_decode_bytes=overrides.pop("max_decode_bytes", None),
    )
    data = overrides.pop("data", _wav())
    try:
        SpeechToTextBaseServing._decode_and_chunk_speech(fake, data)
    except ValueError as exc:
        return exc
    pytest.fail("core's decode step accepted the audio")


# ---------------------------------------------------- VLLMValidationError


# @spec ING-ERR-052
def test_cores_upload_size_error_is_request_too_large() -> None:
    from starlette.datastructures import UploadFile
    from vllm.entrypoints.speech_to_text.base.utils import (
        read_upload_with_limit,
    )
    from vllm.exceptions import VLLMValidationError

    upload = UploadFile(io.BytesIO(b"x" * 4096), size=4096)
    with pytest.raises(VLLMValidationError) as raised:
        asyncio.run(read_upload_with_limit(upload, max_size_mb=0.001))
    error = _classify(raised.value)
    assert error.code == "request_too_large"


# @spec ING-ERR-052
def test_cores_streamed_upload_size_error_is_request_too_large() -> None:
    from starlette.datastructures import UploadFile
    from vllm.entrypoints.speech_to_text.base.utils import (
        read_upload_with_limit,
    )
    from vllm.exceptions import VLLMValidationError

    upload = UploadFile(io.BytesIO(b"x" * 4096))  # no size: chunked read
    with pytest.raises(VLLMValidationError) as raised:
        asyncio.run(read_upload_with_limit(upload, max_size_mb=0.001))
    assert _classify(raised.value).code == "request_too_large"


# @spec ING-ERR-052
def test_other_validation_error_is_invalid_config_field_naming_it() -> None:
    from vllm.exceptions import VLLMValidationError

    exc = VLLMValidationError(
        "Stream options can only be defined when `stream=True`.",
        parameter="stream_include_usage",
    )
    error = _classify(exc)
    assert error.code == "invalid_config_field"
    assert "stream_include_usage" in error.fields


# ------------------------------------------------------ decoder ValueError


# @spec ING-ERR-053
@pytest.mark.parametrize("backend", ["auto", "soundfile", "torchcodec"])
def test_cores_decode_duration_limit_is_request_too_large(backend) -> None:
    exc = _core_load_error(backend, max_duration_s=0.1)
    assert _classify(exc).code == "request_too_large"


# @spec ING-ERR-053
@pytest.mark.parametrize("backend", ["auto", "soundfile", "torchcodec"])
def test_cores_decode_size_limit_is_request_too_large(backend) -> None:
    exc = _core_load_error(backend, max_decode_bytes=100)
    assert _classify(exc).code == "request_too_large"


# @spec ING-ERR-053
def test_cores_service_decode_duration_limit_is_request_too_large() -> None:
    exc = _serving_decode_error(max_duration_s=0.1)
    assert _classify(exc).code == "request_too_large"


# @spec ING-ERR-053
def test_cores_corrupt_audio_error_is_invalid_audio() -> None:
    exc = _serving_decode_error(data=b"not audio at all" * 16)
    assert _classify(exc).code == "invalid_audio"


# @spec ING-ERR-053
def test_cores_unsupported_audio_error_is_invalid_audio(monkeypatch) -> None:
    from vllm.entrypoints.speech_to_text.base import serving

    def unreadable(*args, **kwargs):
        raise RuntimeError("decoder blew up")

    # core wraps any non-ValueError decoder failure in its own message
    monkeypatch.setattr(serving, "load_audio", unreadable)
    exc = _serving_decode_error()
    assert _classify(exc).code == "invalid_audio"


# --------------------------------------------------------------- fallback


# @spec ING-ERR-039
@pytest.mark.parametrize(
    "message",
    [
        "Unsupported language: 'xx'",
        "prompt is not supported by this model",
        # no length refusal is projected (ING-FILE-081): core's own message
        # about the model length is one more unmatched ``ValueError``
        "file exceeds max_model_len: 84000 tokens > 76800",
        "",
    ],
)
def test_unmatched_value_error_is_invalid_config_field(message) -> None:
    error = _classify(ValueError(message))
    assert error.code == "invalid_config_field"
    assert error.code != "internal"


# @spec ING-ERR-040
def test_engine_dead_error_is_service_unavailable() -> None:
    from vllm.v1.engine.exceptions import EngineDeadError

    assert _classify(EngineDeadError()).code == "service_unavailable"


# @spec ING-ERR-040
@pytest.mark.parametrize(
    "exc", [RuntimeError("boom"), KeyError("x"), TypeError("y")]
)
def test_any_other_exception_is_internal(exc) -> None:
    assert _classify(exc).code == "internal"


# ------------------------------------------------------- ErrorResponse


def _project_response(response):
    from vllm_riva_frontend.errors import project_error_response

    return project_error_response(response)


# @spec ING-ERR-041
@pytest.mark.parametrize(
    "status", [HTTPStatus.BAD_REQUEST, HTTPStatus.UNPROCESSABLE_ENTITY]
)
def test_client_error_response_is_invalid_config_field_naming_param(
    status,
) -> None:
    error = _project_response(
        error_response("bad language", status, param="language")
    )
    assert error.code == "invalid_config_field"
    assert error.fields == ("language",)


# @spec ING-ERR-041
def test_client_error_response_without_param_names_audio() -> None:
    error = _project_response(error_response("bad", HTTPStatus.BAD_REQUEST))
    assert error.code == "invalid_config_field"
    assert error.fields == ("audio",)


# @spec ING-ERR-041
@pytest.mark.parametrize(
    ("status", "code"),
    [
        (HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request_too_large"),
        (HTTPStatus.NOT_IMPLEMENTED, "unsupported_capability"),
        (HTTPStatus.SERVICE_UNAVAILABLE, "service_unavailable"),
        (HTTPStatus.INTERNAL_SERVER_ERROR, "internal"),
        (HTTPStatus.BAD_GATEWAY, "internal"),
        (HTTPStatus.GATEWAY_TIMEOUT, "internal"),
    ],
)
def test_error_response_is_projected_by_status(status, code) -> None:
    error = _project_response(
        error_response("x", status, err_type="InternalServerError")
    )
    assert error.code == code
