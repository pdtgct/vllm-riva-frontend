"""The stream owner behind the Realtime adapter.

Sessions are claimed through the test app's dispatcher
(``/v1/realtime?intent=transcription``) and driven with
``support.realtime.RealtimeClient``; the first update is the public default
session object with a language set.
"""

from __future__ import annotations

import asyncio
import contextlib
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


# ------------------------------------------------------- sends and drain


# @spec ING-LIFE-051, ING-LIFE-053
async def test_a_send_stalled_when_the_drain_begins_ends_by_d(
    realtime, first_update, fake_factory, test_app, drain_trigger, clock
):
    ws = BlockingSendWebSocket(test_app, REALTIME_URL)
    session = await _connected(realtime, ws)
    ws.block = True
    await session.update(first_update())
    await session.settle()
    assert ws.stalled == 1
    lease = _lease(fake_factory)

    _, deadline = await drain_trigger()
    assert lease.names() == []
    # The send's own bound (60 s) is far off; D, set after it began, rules.
    await session.advance(deadline - clock.monotonic() + 0.01)

    assert lease.names() == ["abort", "release"]
    await ws.wait_closed()
    ws.unblocked.set()
    await session.settle()
    assert ws.stalled == 1  # no further sends, the close included


# @spec ING-LIFE-051, ING-ERR-059
async def test_a_send_starting_after_d_to_a_stalled_transport_does_not_wait(
    realtime, first_update, fake_factory, test_app, drain_trigger, clock
):
    ws = BlockingSendWebSocket(test_app, REALTIME_URL)
    session = await _connected(realtime, ws)
    await session.update(first_update())
    assert (await session.next()).get("type") == UPDATED
    lease = _lease(fake_factory)
    lease.script("flush", Block())
    _, deadline = await drain_trigger()
    await session.settle()
    assert "flush" in lease.names()
    ws.block = True

    await session.advance(deadline - clock.monotonic() + 0.01)
    await session.settle()

    # The abort row's error event was offered once, at D, and not waited for.
    assert ws.done
    assert ws.stalled == 1
    assert ws.parked[0]["type"] == "websocket.send"
    assert lease.names()[-3:] == ["flush", "abort", "release"]


# ------------------------------------------------------ cleanup ownership


# @spec ING-LIFE-058, ING-EP-060, ING-EP-059
async def test_a_second_cancellation_waits_for_cleanup(
    realtime, first_update, fake_factory, started
):
    session, _ = await _configured(realtime, first_update)
    lease = _lease(fake_factory)
    block = Block()
    lease.script("abort", block)
    started.registry.cancel_owners()
    await block.entered.wait()

    # Later cancellations reach the surface while it waits for cleanup.
    try:
        for _ in range(2):
            started.registry.cancel_owners()
            await session.settle()

        # The record and the admission outlive the running cleanup.
        assert not session.done
        assert len(started.registry.records) == 1
        assert started.ingress.count == 1
    finally:
        block.release()
    await session.wait_closed()
    await session.drain_events()
    terminal = session.terminal
    assert (
        terminal is not None and error_code(terminal) == "service_unavailable"
    )
    assert session.close_code == 1012
    assert lease.names()[-2:] == ["abort", "release"]
    assert len(started.registry.records) == 0
    assert started.ingress.count == 0


# @spec ING-LIFE-058, ING-LIFE-035, ING-EP-060
async def test_a_continuation_holds_the_record_and_admission_after_the_answer(
    realtime, first_update, fake_factory, started
):
    session, _ = await _configured(realtime, first_update)
    lease = _lease(fake_factory)
    lease.script("feed", Raise(failed("expired")))
    abort = Block(delay_cancel=True)
    lease.script("abort", abort)
    await session.append(PIECE)
    await abort.entered.wait()

    # Past the cleanup budget the session is answered; the abort that
    # delays its cancellation stays with the record's continuation.
    try:
        await session.advance(started.config.session_cleanup_timeout + 0.1)
        await session.wait_closed()
        await session.drain_events()
        assert error_code(session.terminal) == "idle_timeout"
        assert session.close_code == 1011
        records = list(started.registry.records.values())
        assert len(records) == 1, "the record went before its continuation"
        assert records[0].continuation is not None
        assert not records[0].continuation.done()
        assert started.ingress.count == 1
    finally:
        abort.release()
    await session.settle()
    assert records[0].continuation.done()
    assert lease.names() == ["feed", "abort", "release"]
    assert len(started.registry.records) == 0
    assert started.ingress.count == 0


# ------------------------------------- terminal sends, refusals and the drain


class StubbornSendWebSocket(BlockingSendWebSocket):
    """A ``BlockingSendWebSocket`` whose parked sends resist cancellation.

    A parked send that is cancelled counts the cancellation, then either
    delays it until ``settled`` is set (``suppress=False``) or swallows it
    and returns as if it had been sent (``suppress=True``).
    """

    def __init__(self, *args, suppress: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.suppress = suppress
        self.settled = asyncio.Event()
        self.cancellations = 0

    async def _send(self, message) -> None:
        try:
            await super()._send(message)
        except asyncio.CancelledError:
            self.cancellations += 1
            if self.suppress:
                return
            await self.settled.wait()
            raise


async def _final_stalled(realtime, first_update, fake_factory, ws):
    """A session whose ``completed`` event is parked after ``release``."""
    session = await _connected(realtime, ws)
    await session.update(first_update())
    assert (await session.next()).get("type") == UPDATED
    lease = _lease(fake_factory)
    await session.append(PIECE)
    await session.settle()
    ws.block = True
    await session.done_event()
    await session.settle()
    assert lease.names()[-3:] == ["flush", "finish", "release"]
    assert ws.stalled == 1
    assert json.loads(ws.parked[0]["text"])["type"] == COMPLETED
    return session, lease


# @spec ING-LIFE-051, ING-EP-060
async def test_a_final_stalled_before_the_drain_ends_by_d(
    realtime, first_update, fake_factory, test_app, drain_trigger, started
):
    ws = BlockingSendWebSocket(test_app, REALTIME_URL)
    session, _ = await _final_stalled(realtime, first_update, fake_factory, ws)
    # The owner's cleanup has ended; the handler still holds the record.
    assert len(started.registry.records) == 1

    _, deadline = await drain_trigger()
    assert not ws.done
    await session.advance(deadline - started.clock.monotonic() + 0.01)

    assert ws.done, "the final's send outlived D"
    ws.unblocked.set()
    await session.settle()
    assert ws.stalled == 1  # no close after the failed send
    assert len(started.registry.records) == 0
    assert started.ingress.count == 0


# @spec ING-LIFE-051, ING-EP-059, ING-EP-060, ING-LIFE-058
async def test_a_terminal_error_stalled_before_the_drain_ends_by_d(
    realtime, first_update, fake_factory, test_app, drain_trigger, started
):
    ws = BlockingSendWebSocket(test_app, REALTIME_URL)
    session = await _connected(realtime, ws)
    await session.update(first_update())
    assert (await session.next()).get("type") == UPDATED
    lease = _lease(fake_factory)
    lease.script("feed", Raise(failed("expired")))
    ws.block = True
    await session.append(PIECE)
    await session.settle()
    assert lease.names() == ["feed", "abort", "release"]
    assert ws.stalled == 1
    assert json.loads(ws.parked[0]["text"])["type"] == ERROR

    # Lease cleanup has ended; the record stays while its error is sent.
    (record,) = started.registry.records.values()
    assert not record.task.done()
    assert not started.registry.settled()

    _, deadline = await drain_trigger()
    assert not ws.done
    await session.advance(deadline - started.clock.monotonic() + 0.01)

    assert ws.done, "the error's send outlived D"
    ws.unblocked.set()
    await session.settle()
    assert ws.stalled == 1  # neither the close nor anything else followed
    assert len(started.registry.records) == 0


# @spec ING-LIFE-051, ING-NIMWS-016, ING-ERR-015
async def test_a_blocked_refusal_ends_by_an_earlier_d(
    realtime, test_app, drain_trigger, started
):
    started.ingress.close()  # refused before any drain begins
    ws = BlockingSendWebSocket(test_app, REALTIME_URL)
    ws.block = True
    session = realtime(ws)
    assert await session.connect()
    await session.settle()
    assert ws.stalled == 1
    assert json.loads(ws.parked[0]["text"])["type"] == ERROR
    assert len(started.registry.records) == 0

    # The refusal's own bound (60 s) is far off; a drain then sets D = +3 s.
    _, deadline = await drain_trigger()
    assert not ws.done
    await session.advance(deadline - started.clock.monotonic() + 0.01)

    assert ws.done, "the refusal outlived D"
    ws.unblocked.set()
    await session.settle()
    assert ws.stalled == 1  # the 1013 close was not sent after it


# @spec ING-LIFE-051
async def test_a_final_due_after_d_is_not_started(
    realtime, first_update, fake_factory, test_app, drain_trigger, started
):
    ws = BlockingSendWebSocket(test_app, REALTIME_URL)
    session = await _connected(realtime, ws)
    await session.update(first_update())
    assert (await session.next()).get("type") == UPDATED
    lease = _lease(fake_factory)
    await session.append(PIECE)
    await session.settle()
    clock = started.clock
    # The shutdown finalization's ``release`` returns just after D.
    lease.after("release", lambda: clock.set(started.drain_deadline + 0.5))

    await drain_trigger()
    await session.settle()

    assert lease.names()[-3:] == ["flush", "finish", "release"]
    assert ws.done
    await session.drain_events()
    # Neither the ``completed`` event nor its close was started.
    assert COMPLETED not in session.types()
    assert session.close_code is None
    assert ws.stalled == 0


# @spec ING-LIFE-051
async def test_an_acknowledgement_due_after_d_is_not_started(
    realtime, first_update, fake_factory, test_app, app_state, started
):
    ws = BlockingSendWebSocket(test_app, REALTIME_URL)
    session = await _connected(realtime, ws)
    await session.update(first_update())
    assert (await session.next()).get("type") == UPDATED
    lease = _lease(fake_factory)
    drain = asyncio.create_task(started.drain(app_state, 8.0))
    try:
        await session.settle()
        deadline = started.drain_deadline
        assert deadline is not None
        clock = started.clock
        # The locale change succeeds just after D; its ack is then due.
        lease.after("update_locale", lambda: clock.set(deadline + 0.5))
        await session.update(first_update("de-DE"))
        await session.settle()

        assert "update_locale" in lease.names()
        assert ws.done
        await session.drain_events()
        assert session.types().count(UPDATED) == 1  # the first ack only
        assert session.close_code is None
        assert lease.names()[-2:] == ["abort", "release"]
    finally:
        drain.cancel()
        with contextlib.suppress(BaseException):
            await drain


# @spec ING-LIFE-051, ING-EP-059, ING-EP-060, ING-LIFE-037
async def test_a_send_delaying_its_cancellation_fails_at_its_bound(
    realtime, first_update, fake_factory, test_app, started, app_state
):
    ws = StubbornSendWebSocket(test_app, REALTIME_URL)
    session = await _connected(realtime, ws)
    ws.block = True
    await session.update(first_update())
    await session.settle()
    assert ws.stalled == 1
    lease = _lease(fake_factory)

    await session.advance(started.config.response_send_timeout + 0.1)

    # Failed at its bound: the handler waited for no settlement, and
    # cleanup began at once with its own budget.
    assert ws.cancellations == 1
    assert ws.done
    assert lease.names() == ["abort", "release"]
    (record,) = started.registry.records.values()
    (unsettled,) = record.unsettled_sends
    assert unsettled in record.children and not unsettled.done()
    assert started.ingress.count == 1

    # ``stop`` reaches the unsettled send through the record.
    from support.app import stop_plugins

    await stop_plugins(app_state)
    await session.settle()
    assert unsettled.cancelled()
    assert len(started.registry.records) == 0
    assert started.ingress.count == 0
    assert ws.stalled == 1


class TestLateSend:
    @pytest.fixture
    def riva_options(self):
        return {"response_send_timeout": 5}

    # @spec ING-LIFE-051
    async def test_a_send_completing_after_its_bound_has_failed(
        self, realtime, first_update, fake_factory, test_app, started
    ):
        ws = BlockingSendWebSocket(test_app, REALTIME_URL)
        session = await _connected(realtime, ws)
        ws.block = True
        await session.update(first_update())
        await session.settle()
        assert ws.stalled == 1
        lease = _lease(fake_factory)

        # The bound passes and the acknowledgement then completes, all
        # before the handler resumes.
        clock = started.clock
        clock.set(clock.monotonic() + 5.1)
        ws.unblocked.set()
        await session.settle()

        assert ws.done, "the handler took the late send as delivered"
        await session.drain_events()
        assert UPDATED in session.types(), "the send did complete, late"
        assert lease.names() == ["abort", "release"]
        assert ws.stalled == 1  # nothing further, the close included
        assert session.close_code is None
        assert len(started.registry.records) == 0
        assert started.ingress.count == 0


# @spec ING-LIFE-051, ING-EP-059, ING-EP-043
async def test_after_stop_a_resisting_send_neither_waits_nor_leaks(
    realtime, first_update, fake_factory, test_app, started, app_state
):
    ws = StubbornSendWebSocket(test_app, REALTIME_URL)
    session = await _connected(realtime, ws)
    ws.block = True
    await session.update(first_update())
    await session.settle()
    assert ws.stalled == 1
    lease = _lease(fake_factory)
    before = asyncio.all_tasks()

    from support.app import stop_plugins

    stopper = asyncio.ensure_future(stop_plugins(app_state))
    try:
        await session.settle()

        # No clock time passed: every send bound counted as passed.
        assert ws.done, "the handler waited after stop"
        # The terminal error got its one attempt; its close never started.
        assert ws.stalled == 2
        assert json.loads(ws.parked[1]["text"])["type"] == ERROR
        assert "abort" not in lease.names()
        assert lease.state == "torn_down"
        await session.advance(1.1)  # ``stop``'s one bounded wait
        assert stopper.done()
        # Every task started since ``stop`` is done or was cancelled.
        leftover = [
            task for task in asyncio.all_tasks() - before if not task.done()
        ]
        assert all(task.cancelling() for task in leftover), leftover
    finally:
        ws.settled.set()
        await session.settle()
    assert len(started.registry.records) == 0
    assert started.ingress.count == 0


# @spec ING-LIFE-051
async def test_a_send_suppressing_its_cancellation_still_failed(
    realtime, first_update, fake_factory, test_app, started
):
    ws = StubbornSendWebSocket(test_app, REALTIME_URL, suppress=True)
    session = await _connected(realtime, ws)
    ws.block = True
    await session.update(first_update())
    await session.settle()
    assert ws.stalled == 1
    lease = _lease(fake_factory)

    await session.advance(started.config.response_send_timeout + 0.1)

    # The send returned after its cancellation; it failed all the same.
    assert ws.cancellations == 1
    assert ws.done
    assert lease.names() == ["abort", "release"]
    ws.block = False
    await session.settle()
    assert ws.stalled == 1  # nothing further, the close included
    assert session.close_code is None
    assert len(started.registry.records) == 0
