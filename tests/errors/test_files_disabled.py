"""``unsupported_capability`` (HTTP 501) when complete files are disabled.

A server whose smaller of state slots and active limit is below 2 disables
complete files with a warning (vehicle leaf); the file route and
``Recognize`` then refuse without creating an owner.
"""

from __future__ import annotations

import pytest
from support.audio import pcm16_bytes, pcm16_sine, wav_bytes

MODEL = "nvidia/nemotron-asr-streaming-test"


@pytest.fixture
def fake_engine():
    from support.fake_engine import FakeEngineClient
    from support.grpc_harness import free_port

    return FakeEngineClient(
        num_gpu_blocks=2,  # one state slot
        riva_frontend={"grpc_bind": f"127.0.0.1:{free_port()}"},
    )


# @spec ING-ERR-055, ING-ERR-009
async def test_route_with_files_disabled_is_501_unsupported_capability(
    started, test_app, fake_transcription
) -> None:
    from support.app import http_client
    from support.http import ROUTE

    assert started.files_enabled is False
    async with http_client(test_app) as client:
        response = await client.post(
            ROUTE,
            files={"file": ("a.wav", wav_bytes(pcm16_bytes(pcm16_sine(0.1))))},
            data={"model": MODEL},
        )
    assert response.status_code == 501
    error = response.json()["error"]
    assert set(error) == {"message", "type", "param", "code"}
    assert error["message"].startswith("unsupported_capability: ")
    assert error["code"] == 501
    assert fake_transcription.calls == []


# @spec ING-ERR-031
async def test_recognize_with_files_disabled_is_unimplemented(
    started, fake_transcription, helpers
) -> None:
    from support.riva import recognize, recognize_request

    outcome = await recognize(
        started.servicer, recognize_request(pcm16_bytes(pcm16_sine(0.1)))
    )
    assert outcome.status.name == "UNIMPLEMENTED"
    assert outcome.details.startswith("unsupported_capability: ")
    assert fake_transcription.calls == []
