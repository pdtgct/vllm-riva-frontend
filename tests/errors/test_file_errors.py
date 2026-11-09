"""Errors of the complete-file paths: ``Recognize`` and the file route.

``Recognize`` is called on the servicer directly; the file route through
``test_app`` (core's routers, exception handlers and middleware). Core's
transcription service is ``FakeTranscriptionServing``, scripted to raise or
return what core's would.
"""

from __future__ import annotations

import asyncio
from http import HTTPStatus
from typing import Any

import httpx
import pytest
from support.audio import pcm16_bytes, pcm16_sine, wav_bytes
from support.fake_session import Block, Raise
from support.fake_transcription import error_response
from support.riva import recognize, recognize_request

MODEL = "nvidia/nemotron-asr-streaming-test"


async def _recognize(plugin, request):
    """``Recognize`` on the servicer (``support.riva.recognize``)."""
    outcome = await recognize(plugin.servicer, request)
    assert outcome.error is None, outcome.error
    return outcome


# ------------------------------------------------------------- Recognize


# @spec ING-ERR-052, ING-ERR-053, ING-ERR-054, ING-ERR-039, ING-ERR-040
@pytest.mark.parametrize(
    ("raised", "status", "code"),
    [
        ("size", "RESOURCE_EXHAUSTED", "request_too_large"),
        ("budget", "RESOURCE_EXHAUSTED", "request_too_large"),
        ("value", "INVALID_ARGUMENT", "invalid_config_field"),
        ("dead", "UNAVAILABLE", "service_unavailable"),
        ("other", "INTERNAL", "internal"),
    ],
)
async def test_recognize_projects_cores_raised_exceptions(
    raised, status, code, started, fake_transcription, helpers
) -> None:
    from vllm.exceptions import VLLMValidationError
    from vllm.v1.engine.exceptions import EngineDeadError

    exc = {
        "size": VLLMValidationError(
            "Maximum file size exceeded", parameter="audio_filesize_mb"
        ),
        "budget": ValueError("file exceeds max_model_len: 9 > 8"),
        "value": ValueError("something core did not like"),
        "dead": EngineDeadError(),
        "other": RuntimeError("boom"),
    }[raised]
    fake_transcription.script(Raise(exc))
    request = recognize_request(pcm16_bytes(pcm16_sine(0.2)))
    outcome = await _recognize(started, request)
    assert outcome.response is None
    assert outcome.status.name == status
    assert outcome.details.startswith(f"{code}: ")


# @spec ING-ERR-041
@pytest.mark.parametrize(
    ("status", "grpc", "code"),
    [
        (HTTPStatus.BAD_REQUEST, "INVALID_ARGUMENT", "invalid_config_field"),
        (
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            "RESOURCE_EXHAUSTED",
            "request_too_large",
        ),
        (HTTPStatus.NOT_IMPLEMENTED, "UNIMPLEMENTED", "unsupported_capability"),
        (HTTPStatus.SERVICE_UNAVAILABLE, "UNAVAILABLE", "service_unavailable"),
        (HTTPStatus.INTERNAL_SERVER_ERROR, "INTERNAL", "internal"),
    ],
)
async def test_recognize_projects_a_returned_error_response(
    status, grpc, code, started, fake_transcription, helpers
) -> None:
    from support.fake_session import Return

    fake_transcription.script(
        Return(error_response("core said no", status, param="language"))
    )
    request = recognize_request(pcm16_bytes(pcm16_sine(0.2)))
    outcome = await _recognize(started, request)
    assert outcome.status.name == grpc
    assert outcome.details.startswith(f"{code}: ")
    if code == "invalid_config_field":
        assert "language" in outcome.details.split("(fields:", 1)[1]


# @spec ING-ERR-022, ING-ERR-029, ING-ERR-003
async def test_recognize_truncated_riff_data_is_invalid_audio(
    started, fake_transcription, helpers
) -> None:
    payload = pcm16_bytes(pcm16_sine(0.1))
    truncated = wav_bytes(payload, data_length=len(payload) + 640)
    request = recognize_request(
        truncated,
        encoding="ENCODING_UNSPECIFIED",
        sample_rate_hertz=0,
        audio_channel_count=0,
    )
    outcome = await _recognize(started, request)
    assert outcome.status.name == "INVALID_ARGUMENT"
    assert outcome.details.startswith("invalid_audio: ")
    assert fake_transcription.calls == []


# @spec ING-ERR-029, ING-ERR-003
async def test_recognize_rejection_names_every_offending_field(
    started, fake_transcription, helpers
) -> None:
    request = recognize_request(
        pcm16_bytes(pcm16_sine(0.1)),
        profanity_filter=True,
        enable_word_time_offsets=True,
    )
    outcome = await _recognize(started, request)
    assert outcome.status.name == "INVALID_ARGUMENT"
    head, _, names = outcome.details.partition(" (fields: ")
    assert head.startswith("invalid_config_field: ")
    assert names.endswith(")")
    assert "profanity_filter" in names
    assert "enable_word_time_offsets" in names


# -------------------------------------------------------- the file route


def _client(app) -> httpx.AsyncClient:
    from support.app import http_client

    return http_client(app)


def _form() -> dict[str, Any]:
    return {
        "files": {
            "file": ("clip.wav", wav_bytes(pcm16_bytes(pcm16_sine(0.2))))
        },
        "data": {"model": MODEL},
    }


# @spec ING-ERR-009, ING-ERR-055
async def test_route_malformed_content_type_is_400_in_cores_shape(
    started, test_app
) -> None:
    async with _client(test_app) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            content=b"not multipart",
            headers={"Content-Type": "text/plain"},
        )
    assert response.status_code == 400
    error = response.json()["error"]
    assert set(error) == {"message", "type", "param", "code"}
    assert error["message"].startswith("malformed_request: ")
    assert error["code"] == 400


# @spec ING-ERR-009, ING-ERR-055
async def test_route_oversized_content_type_is_413(started, test_app) -> None:
    value = "multipart/form-data; boundary=" + "a" * 5000
    async with _client(test_app) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            content=b"x",
            headers={"Content-Type": value},
        )
    assert response.status_code == 413
    assert response.json()["error"]["message"].startswith("request_too_large: ")


# @spec ING-ERR-009, ING-ERR-055, ING-ERR-046
async def test_route_before_start_is_503_service_unavailable(
    app_state, test_app
) -> None:
    async with _client(test_app) as client:
        response = await client.post("/v1/audio/transcriptions", **_form())
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["message"].startswith("service_unavailable: ")
    assert error["code"] == 503


# @spec ING-ERR-009
async def test_route_passes_cores_error_response_through_unchanged(
    started, test_app, fake_transcription
) -> None:
    from support.fake_session import Return

    core_error = error_response(
        "Unsupported language: 'xx'", HTTPStatus.BAD_REQUEST, param="language"
    )
    fake_transcription.script(Return(core_error))
    async with _client(test_app) as client:
        response = await client.post("/v1/audio/transcriptions", **_form())
    assert response.status_code == 400
    assert response.json() == core_error.model_dump()


# @spec ING-ERR-009
async def test_route_passes_cores_raised_validation_error_through(
    started, test_app, fake_transcription
) -> None:
    from vllm.exceptions import VLLMValidationError

    fake_transcription.script(
        Raise(VLLMValidationError("bad value", parameter="temperature"))
    )
    async with _client(test_app) as client:
        response = await client.post("/v1/audio/transcriptions", **_form())
    assert response.status_code == 400
    error = response.json()["error"]
    # core's own handler shaped it: no frontend catalog prefix
    assert not error["message"].startswith("invalid_config_field: ")
    assert "bad value" in error["message"]


# @spec ING-ERR-055, ING-ERR-046
async def test_route_stalled_body_is_408_request_timeout(
    started, test_app, clock, helpers
) -> None:
    from support.asgi import asgi_http

    hold = asyncio.Event()
    task = asyncio.create_task(
        asgi_http(
            test_app,
            "POST",
            "/v1/audio/transcriptions",
            headers=[
                ("content-type", "multipart/form-data; boundary=xyz"),
                ("content-length", "100"),
            ],
            body=[b"--xyz\r\n"],
            hold_body=hold,
        )
    )
    await asyncio.sleep(0.01)
    await clock.advance(121)  # http_body_read_timeout
    response = await asyncio.wait_for(task, 5)
    assert response.status == 408
    assert response.json()["error"]["message"].startswith("request_timeout: ")


# @spec ING-ERR-055, ING-ERR-046
async def test_route_core_execution_timeout_is_504(
    started, test_app, clock, fake_transcription, helpers
) -> None:
    fake_transcription.script(Block())
    async with _client(test_app) as client:
        request = asyncio.create_task(
            client.post("/v1/audio/transcriptions", **_form())
        )
        await helpers.until(lambda: len(fake_transcription.calls) == 1)
        await clock.advance(1201)  # file_execution_timeout default: 1200 s
        # Core's call never settles, so the handler answers when the cleanup
        # bound expires (ING-FILE-070: session_cleanup_timeout, 30 s).
        await clock.advance(31)
        # A bounded poll, not asyncio.wait_for: cancelling the request task
        # is a transport cancellation the owner absorbs while its cleanup
        # waits on virtual time, so a cancelling wait would never return.
        await helpers.until(request.done, what="the 504 answer")
        response = request.result()
    assert response.status_code == 504
    assert response.json()["error"]["message"].startswith("request_timeout: ")
