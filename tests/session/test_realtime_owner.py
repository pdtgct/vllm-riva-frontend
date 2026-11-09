"""The stream owner behind the Realtime adapter.

Sessions are claimed through the test app's dispatcher
(``/v1/realtime?intent=transcription``) and driven with
``support.realtime.RealtimeClient``; the first update is the public default
session object with a language set.
"""

from __future__ import annotations

import json

import pytest
from support.asgi import BlockingSendWebSocket
from support.audio import pcm16_bytes, pcm16_sine
from support.fake_session import Block, Raise, Return, failed
from support.realtime import (
    COMPLETED,
    CREATED,
    DELTA,
    ERROR,
    REALTIME_URL,
    UPDATED,
    error_code,
)

PIECE = pcm16_bytes(pcm16_sine(0.1))


def _lease(fake_factory):
    assert fake_factory.leases, "no lease was opened"
    return fake_factory.leases[0]


async def _connected(realtime, ws=None):
    """A connected session past ``conversation.created``."""
    session = realtime(ws)
    assert await session.connect()
    assert (await session.next())["type"] == CREATED
    return session


async def _configured(realtime, first_update, language="en-US"):
    session = await _connected(realtime)
    await session.update(first_update(language))
    ack = await session.next()
    assert ack.get("type") == UPDATED, ack
    return session, ack


def _of(events, kind):
    return [e for e in events if e.get("type") == kind]


# ------------------------------------------------------- configuration


# @spec ING-LIFE-001
async def test_an_append_before_configuration_is_protocol_order(
    realtime, fake_factory
):
    session = await _connected(realtime)
    await session.append(PIECE)
    await session.wait_closed()
    await session.drain_events()

    errors = _of(session.events, ERROR)
    assert len(errors) == 1
    assert error_code(errors[0]) == "protocol_order"
    assert session.close_code == 1008
    assert [c for c in fake_factory.calls if c.name == "open"] == []


# @spec ING-CORE-011
async def test_the_first_ack_follows_open_and_claims_no_admission(
    realtime, first_update, fake_factory
):
    block = Block()
    fake_factory.script("open", block)
    session = await _connected(realtime)
    await session.update(first_update())
    await block.entered.wait()
    assert await session.drain_events() == []

    block.release()
    ack = await session.next()
    assert ack.get("type") == UPDATED
    text = json.dumps(ack).lower()
    for word in ("admission", "admitted", "slot", "capacity", "queue"):
        assert word not in text


# @spec ING-CORE-021, ING-LIFE-022
async def test_a_retry_after_a_rejected_update_keeps_the_original_bound(
    realtime, first_update, fake_factory
):
    session = await _connected(realtime)
    await session.update(first_update("xx-XX"))
    first = await session.next()
    assert first.get("type") == ERROR
    assert error_code(first) == "unknown_locale"
    await session.advance(25)
    await session.update(first_update("yy-YY"))
    await session.next()
    await session.advance(5.1)  # t=30.1 of the one preconfiguration bound
    await session.wait_closed()
    await session.drain_events()

    terminal = session.terminal
    assert terminal is not None and terminal["type"] == ERROR
    assert error_code(terminal) == "configuration_timeout"
    assert session.close_code == 1011
    assert [c for c in fake_factory.calls if c.name == "open"] == []


# @spec ING-CORE-021
async def test_a_corrected_update_opens_and_audio_waits_for_open(
    realtime, first_update, fake_factory
):
    session = await _connected(realtime)
    await session.update(first_update("xx-XX"))
    await session.next()
    block = Block()
    fake_factory.script("open", block)
    await session.update(first_update("en-US"))
    await block.entered.wait()
    await session.append(PIECE)
    await session.settle()
    assert fake_factory.leases == []

    block.release()
    ack = await session.next()
    assert ack.get("type") == UPDATED
    await session.settle()
    assert sum(_lease(fake_factory).accepted) == 1600


# ------------------------------------------------------- finalization


# @spec ING-LIFE-002, ING-LIFE-025, ING-CORE-019
async def test_done_gives_true_suffix_deltas_and_one_completed_after_release(
    realtime, first_update, fake_factory
):
    session, _ = await _configured(realtime, first_update)
    lease = _lease(fake_factory)
    lease.script("feed", Return(["hello"]), Return([]), Return(["hello world"]))
    lease.script("flush", Return("hello world."))
    block = Block()
    lease.script("release", block)
    for _ in range(3):
        await session.append(PIECE)
    await session.done_event()
    await block.entered.wait()
    events = await session.drain_events()
    assert _of(events, COMPLETED) == []

    block.release()
    await session.wait_closed()
    await session.drain_events()
    events = session.events

    assert [e["delta"] for e in _of(events, DELTA)] == ["hello", " world"]
    completed = _of(events, COMPLETED)
    assert len(completed) == 1
    assert completed[0]["transcript"] == "hello world."
    assert completed[0]["is_last_result"] is True
    assert lease.names()[-3:] == ["flush", "finish", "release"]
    assert session.close_code == 1000


# ----------------------------------------------------- locale updates


# @spec ING-LIFE-007
async def test_a_locale_update_is_acknowledged_after_update_locale(
    realtime, first_update, fake_factory
):
    session, _ = await _configured(realtime, first_update)
    lease = _lease(fake_factory)
    block = Block()
    lease.script("update_locale", block)
    await session.update(first_update("de-DE"))
    await block.entered.wait()
    assert _of(await session.drain_events(), UPDATED) == []

    block.release()
    ack = await session.next()
    assert ack.get("type") == UPDATED
    assert ack["session"]["input_audio_transcription"]["language"] == "de-DE"
    assert [c.args for c in lease.calls if c.name == "update_locale"] == [
        ("de-DE",)
    ]


# @spec ING-LIFE-007
async def test_an_unknown_locale_update_keeps_the_prior_locale(
    realtime, first_update, fake_factory
):
    session, _ = await _configured(realtime, first_update)
    lease = _lease(fake_factory)
    await session.update(first_update("xx-XX"))
    event = await session.next()

    assert event.get("type") == ERROR
    assert error_code(event) == "unknown_locale"
    assert "update_locale" not in lease.names()
    assert lease.locale == "en-US"
    assert not session.done


# @spec ING-LIFE-007
async def test_updates_after_done_are_rejected_and_the_locale_is_frozen(
    realtime, first_update, fake_factory
):
    session, _ = await _configured(realtime, first_update)
    lease = _lease(fake_factory)
    block = Block()
    lease.script("flush", block)
    await session.done_event()
    await block.entered.wait()

    await session.update(first_update("de-DE"))
    await session.settle()
    block.release()
    await session.wait_closed()
    await session.drain_events()
    events = session.events

    assert "update_locale" not in lease.names()
    assert _of(events, COMPLETED) == []
    assert [error_code(e) for e in _of(events, ERROR)] == ["session_terminal"]
    assert session.close_code == 1008


# @spec ING-LIFE-008, ING-LIFE-009
async def test_a_changed_audio_format_is_config_change_rejected(
    realtime, first_update, fake_factory
):
    session, _ = await _configured(realtime, first_update)
    changed = first_update("en-US")
    changed["input_audio_format"] = "g711_ulaw"
    await session.update(changed)
    event = await session.next()

    assert event.get("type") == ERROR
    assert error_code(event) == "config_change_rejected"
    # The update still ends with its acknowledgement: the honoured subset is
    # empty, so it reflects the unchanged session.
    ack = await session.next()
    assert ack.get("type") == UPDATED
    assert ack["session"]["input_audio_format"] == "pcm16"
    assert len(fake_factory.leases) == 1
    lease = _lease(fake_factory)
    assert lease.names() == []
    # The front end is unchanged: PCM16 audio is still accepted as before.
    await session.append(PIECE)
    await session.settle()
    assert sum(lease.accepted) == 1600


# @spec ING-LIFE-009, ING-LIFE-008
async def test_rejections_precede_one_truthful_ack_of_the_honoured_subset(
    realtime, first_update, fake_factory
):
    session, first_ack = await _configured(realtime, first_update)
    changed = first_update("de-DE")
    changed["input_audio_format"] = "g711_ulaw"
    await session.update(changed)
    await session.settle()
    events = await session.drain_events()

    kinds = [e.get("type") for e in events]
    assert kinds[-1] == UPDATED
    assert kinds[:-1] and all(k == ERROR for k in kinds[:-1])
    assert {error_code(e) for e in events[:-1]} == {"config_change_rejected"}
    ack = events[-1]
    assert ack["session"]["input_audio_transcription"]["language"] == "de-DE"
    assert (
        ack["session"]["input_audio_format"]
        == first_ack["session"]["input_audio_format"]
    )
    text = json.dumps(ack).lower()
    assert "admission" not in text and "admitted" not in text
    lease = _lease(fake_factory)
    assert [c.args for c in lease.calls if c.name == "update_locale"] == [
        ("de-DE",)
    ]


# ----------------------------------------------------------- bootstrap


# @spec ING-LIFE-045
async def test_the_bootstrap_post_creates_no_owner_and_no_timer(
    started, test_app, clock
):
    from support.app import http_client

    sleepers = clock.sleepers
    async with http_client(test_app) as client:
        response = await client.post(
            "/v1/realtime/transcription_sessions", json={}
        )

    assert response.status_code == 200
    assert len(started.registry.records) == 0
    assert started.ingress.count == 0
    assert clock.sleepers == sleepers


# ---------------------------------------------------------- disconnects


# @spec ING-LIFE-039, ING-LIFE-038
async def test_a_disconnect_during_feed_is_observed_when_the_call_returns(
    realtime, first_update, fake_factory
):
    session, _ = await _configured(realtime, first_update)
    lease = _lease(fake_factory)
    block = Block()
    lease.script("feed", block)
    await session.append(PIECE)
    await block.entered.wait()

    await session.disconnect(1001)
    await session.settle()
    assert lease.names() == ["feed"]

    block.release(Return(["late"]))
    await session.settle()
    assert lease.names() == ["feed", "abort", "release"]
    assert _of(session.events, COMPLETED) == []


# @spec ING-LIFE-039
async def test_a_disconnect_during_feed_is_observed_at_the_call_bound(
    realtime, first_update, fake_factory
):
    session, _ = await _configured(realtime, first_update)
    lease = _lease(fake_factory)
    block = Block()
    lease.script("feed", block)
    await session.append(PIECE)
    await block.entered.wait()

    await session.disconnect(1001)
    await session.advance(45.1)

    assert lease.names() == ["feed", "abort", "release"]


# @spec ING-LIFE-042
async def test_an_update_locale_over_lease_call_timeout_breaks_the_lease(
    realtime, first_update, fake_factory
):
    session, _ = await _configured(realtime, first_update)
    lease = _lease(fake_factory)
    lease.script("update_locale", Block())
    await session.update(first_update("de-DE"))
    await session.settle()

    await session.advance(45.1)
    await session.wait_closed()
    await session.drain_events()

    assert lease.names() == ["update_locale", "abort", "release"]
    terminal = session.terminal
    assert terminal is not None and terminal["type"] == ERROR
    assert error_code(terminal) == "service_unavailable"
    assert session.close_code == 1011


# ---------------------------------------------------------------- sends


class TestSendBound:
    @pytest.fixture
    def riva_options(self):
        return {"response_send_timeout": 5}

    # @spec ING-LIFE-051
    async def test_a_stalled_send_ends_the_owner_as_a_transport_failure(
        self, realtime, first_update, fake_factory, test_app
    ):
        ws = BlockingSendWebSocket(test_app, REALTIME_URL)
        session = await _connected(realtime, ws)
        ws.block = True
        await session.update(first_update())
        await session.settle()
        assert ws.stalled == 1
        lease = _lease(fake_factory)

        await session.advance(4.9)
        assert lease.names() == []
        await session.advance(0.2)

        assert lease.names() == ["abort", "release"]
        ws.unblocked.set()
        await session.settle()
        # No further sends after the transport failure.
        assert ws.stalled == 1


# @spec ING-LIFE-021
async def test_realtime_engine_expired_is_idle_timeout(
    realtime, first_update, fake_factory
):
    session, _ = await _configured(realtime, first_update)
    lease = _lease(fake_factory)
    lease.script("feed", Raise(failed("expired")))
    await session.append(PIECE)
    await session.wait_closed()
    await session.drain_events()

    terminal = session.terminal
    assert terminal is not None and terminal["type"] == ERROR
    assert error_code(terminal) == "idle_timeout"
    assert session.close_code == 1011
    assert lease.names() == ["feed", "abort", "release"]
