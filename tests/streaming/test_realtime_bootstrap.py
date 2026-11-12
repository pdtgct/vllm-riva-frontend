"""The Realtime bootstrap and its default session object.

``POST /plugins/riva/v1/realtime/transcription_sessions`` (``BOOTSTRAP_PATH``).
"""

from __future__ import annotations

import asyncio

import pytest
from support.app import drain_plugin, http_client, plugins_ready
from support.fake_session import FakeSessionFactory
from support.realtime import BOOTSTRAP_PATH, RealtimeClient, update_event

from .conftest import SERVED, opens, serving


async def _bootstrap(served) -> dict:
    async with http_client(served.app) as client:
        response = await client.post(BOOTSTRAP_PATH, json={})
    assert response.status_code == 200, response.text
    return response.json()


# @spec ING-NIMWS-025
# @spec ING-NIMWS-020
async def test_bootstrap_returns_the_default_object_with_the_served_identity(
    served,
):
    body = await _bootstrap(served)
    assert body["input_audio_transcription"]["model"] == SERVED
    assert body["modalities"] == ["text"]
    assert body["input_audio_format"] == "pcm16"
    params = body["input_audio_params"]
    assert params["sample_rate_hz"] == 16_000
    assert params["num_channels"] == 1
    assert body["recognition_config"]["max_alternatives"] == 1
    assert body["speaker_diarization"]["enable_speaker_diarization"] is False
    assert body["word_boosting"]["enable_word_boosting"] is False
    assert body["endpointing_config"] == {}
    assert "client_secret" in body and body["client_secret"] is None


# @spec ING-NIMWS-013
async def test_bootstrap_language_is_the_factory_default_locale(served):
    body = await _bootstrap(served)
    assert body["input_audio_transcription"]["language"] == "auto"


# @spec ING-NIMWS-013
async def test_bootstrap_language_is_empty_without_a_default_locale(clock):
    factory = FakeSessionFactory(default_locale=None)
    async with serving(clock=clock, factory=factory) as served:
        body = await _bootstrap(served)
    assert body["input_audio_transcription"]["language"] == ""


# @spec ING-NIMWS-025
# @spec ING-NIMWS-013
def test_bootstrap_session_builder_matches_the_route():
    from vllm_riva_frontend.nim_ws import bootstrap_session

    with_default = bootstrap_session(SERVED, "en-US")
    assert with_default["input_audio_transcription"]["language"] == "en-US"
    assert with_default["input_audio_transcription"]["model"] == SERVED
    without = bootstrap_session(SERVED, None)
    assert without["input_audio_transcription"]["language"] == ""


# @spec ING-NIMWS-025
# @spec ING-LIFE-045
async def test_bootstrap_creates_no_owner_and_starts_no_timer(served):
    sleepers = served.clock.sleepers
    records = dict(served.plugin.registry.records)
    count = served.plugin.ingress.count
    await _bootstrap(served)
    await served.clock.settle(20)
    assert served.clock.sleepers == sleepers
    assert dict(served.plugin.registry.records) == records
    assert served.plugin.ingress.count == count
    assert served.factory.leases == []


# @spec ING-NIMWS-013
# @spec ING-NIMWS-022
async def test_the_bootstrap_object_round_trips_as_the_first_update(served):
    body = await _bootstrap(served)
    client = RealtimeClient(served.app, served.clock)
    assert await client.connect()
    assert (await client.next())["type"] == "conversation.created"
    await client.send(update_event(body))
    updated = await client.next()
    assert updated["type"] == "transcription_session.updated", updated
    assert opens(served) == [{"cadence": "560ms", "locale": "auto"}]


async def _bootstrap_object(served) -> dict:
    body = await _bootstrap(served)
    assert body["input_audio_transcription"]["model"] == SERVED
    assert body["input_audio_format"] == "pcm16"
    return body


@pytest.mark.parametrize(
    "state", ["not_started", "draining_flag", "draining", "past_settle_point"]
)
# @spec ING-NIMWS-025
async def test_bootstrap_at_plugin_path_returns_default_object_in_any_state(
    clock, state
):
    """HTTP 200 and the default object whatever the plugin's state."""
    start = state != "not_started"
    async with serving(clock=clock, start=start) as served:
        reference = None
        if state == "not_started":
            assert not plugins_ready(served.state)
        elif state == "draining_flag":
            reference = await _bootstrap_object(served)
            served.state.draining = True
        drain = None
        if state in ("draining", "past_settle_point"):
            reference = await _bootstrap_object(served)
            drain = asyncio.create_task(
                drain_plugin(served.plugin, served.state, 60.0)
            )
            await served.clock.settle(20)
            assert not served.plugin.admission_open
            if state == "past_settle_point":
                await served.clock.advance(49.5)  # past the settle point
        body = await _bootstrap_object(served)
        if reference is not None:
            # The default object, not a stable id: ``id`` is per session.
            def without_id(obj: dict) -> dict:
                return {k: v for k, v in obj.items() if k != "id"}

            assert without_id(body) == without_id(reference)
        assert served.factory.leases == []
        if drain is not None:
            await served.clock.advance(60)
            await asyncio.wait_for(drain, 5.0)
