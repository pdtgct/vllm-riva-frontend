"""Realtime projections: continuing rejections, the observer, size and send.

``invalid_config_field``, ``unsupported_capability``, ``unknown_locale`` and
``config_change_rejected`` answer an ``error`` and leave the session usable;
traffic after ``input_audio_buffer.done`` is classified by the receive
observer; oversized messages close 1009; and every send is bounded.
"""

from __future__ import annotations

import asyncio
import base64
import json

import pytest
from support.fake_session import Block, Return
from support.realtime import DELTA

SMALL_WS = {"ws_event_envelope_max_bytes": 4096, "ws_receive_max_bytes": 8192}


def _lease(fake_factory):
    assert fake_factory.leases, "open was not called"
    return fake_factory.leases[-1]


async def _done_with_flush_blocked(drv, fake_factory, helpers, seconds=0.1):
    lease = _lease(fake_factory)
    block = Block()
    lease.script("flush", block)
    await drv.send_audio(helpers.speech_bytes(seconds))
    await drv.finalize()
    await helpers.until(lambda: "flush" in lease.names())
    return lease, block


# ------------------------------------- rejections before configuration


# @spec ING-ERR-034, ING-ERR-003, ING-FE-016
async def test_format_rejection_before_configuration_allows_a_retry(
    started, stream, fake_factory, helpers
) -> None:
    drv = stream("realtime", started)
    assert await drv.connect()
    assert (await drv.receive())["type"] == "conversation.created"
    bad = drv.session()
    bad["input_audio_format"] = "g711_ulaw"  # G.711 at the default 16 kHz
    event = await drv.update(bad)
    assert event["type"] == "error"
    assert helpers.event_code(event) == "unsupported_format"
    # G.711 is 8 kHz only: the declared rate is the offending field.
    assert helpers.event_fields(event) == ["sample_rate_hz"]
    assert drv.close is None
    assert fake_factory.leases == []
    event = await drv.update(drv.session())
    assert event["type"] == "transcription_session.updated"
    assert drv.close is None
    assert len(fake_factory.leases) == 1


# @spec ING-ERR-034, ING-ERR-003
async def test_configuration_rejection_before_configuration_does_not_close(
    started, stream, fake_factory, helpers
) -> None:
    drv = stream("realtime", started)
    assert await drv.connect()
    assert (await drv.receive())["type"] == "conversation.created"
    bad = drv.session()
    bad["not_a_session_field"] = True
    event = await drv.update(bad)
    assert event["type"] == "error"
    assert helpers.event_code(event) == "invalid_config_field"
    assert helpers.event_fields(event) == ["not_a_session_field"]
    assert drv.close is None
    event = await drv.update(drv.session())
    assert event["type"] == "transcription_session.updated"


# ----------------------------------------------- continuing after config


# @spec ING-ERR-033
async def test_unknown_locale_update_continues_with_the_old_locale(
    started, stream, fake_factory, helpers
) -> None:
    drv = stream("realtime", started)
    await drv.configure()
    session = drv.session()
    session["input_audio_transcription"] = {
        **session.get("input_audio_transcription", {}),
        "language": "xx-XX",
    }
    event = await drv.update(session)
    assert event["type"] == "error"
    assert helpers.event_code(event) == "unknown_locale"
    assert drv.close is None
    lease = _lease(fake_factory)
    assert "update_locale" not in lease.names()
    await drv.finalize()
    outcome = await drv.outcome()
    assert outcome.close == 1000
    assert len(outcome.finals) == 1


# @spec ING-ERR-033
async def test_changed_immutable_field_is_config_change_rejected_and_continues(
    started, stream, fake_factory, helpers
) -> None:
    drv = stream("realtime", started)
    await drv.configure()
    session = drv.session()
    session["input_audio_format"] = "g711_alaw"
    event = await drv.update(session)
    assert event["type"] == "error"
    assert helpers.event_code(event) == "config_change_rejected"
    ack = await drv.receive()
    assert ack is not None and ack["type"] == "transcription_session.updated"
    assert drv.close is None
    await drv.finalize()
    outcome = await drv.outcome()
    assert outcome.close == 1000


# @spec ING-ERR-033
async def test_unknown_field_on_a_later_update_continues(
    started, stream, fake_factory, helpers
) -> None:
    drv = stream("realtime", started)
    await drv.configure()
    session = drv.session()
    session["not_a_session_field"] = 1
    events = [await drv.update(session)]
    while events[-1]["type"] != "transcription_session.updated":
        next_event = await drv.receive()
        assert next_event is not None
        events.append(next_event)
    errors = [e for e in events if e["type"] == "error"]
    assert [helpers.event_code(e) for e in errors] == ["invalid_config_field"]
    assert drv.close is None


# @spec ING-ERR-033, ING-ERR-034, ING-ERR-003
async def test_absent_capability_is_unsupported_capability_and_continues(
    started, stream, fake_factory, helpers
) -> None:
    drv = stream("realtime", started)
    assert await drv.connect()
    assert (await drv.receive())["type"] == "conversation.created"
    session = drv.session()
    section = dict(session["speaker_diarization"])
    toggles = [key for key in section if key.startswith("enable")]
    assert toggles, section
    for key in toggles:
        section[key] = True
    session["speaker_diarization"] = section
    first = await drv.update(session)
    assert first["type"] == "error"
    assert helpers.event_code(first) == "unsupported_capability"
    assert helpers.event_fields(first) == ["speaker_diarization"]
    assert drv.close is None
    assert fake_factory.leases == []
    event = await drv.update(drv.session())
    assert event["type"] == "transcription_session.updated"


# ----------------------------------------------------- invalid audio


# @spec ING-ERR-022, ING-ERR-011
async def test_undecodable_append_is_invalid_audio_failed_1008(
    started, stream, fake_factory, helpers
) -> None:
    drv = stream("realtime", started)
    await drv.configure()
    await drv.ws.send_json(
        {"type": "input_audio_buffer.append", "audio": "@@not base64@@"}
    )
    outcome = await drv.outcome()
    helpers.assert_row(
        outcome, "invalid_audio", "INVALID_ARGUMENT", helpers.FAILED_EVENT, 1008
    )
    assert _lease(fake_factory).names()[-2:] == ["abort", "release"]


# ---------------------------------------------------------- oversize


# @spec ING-ERR-026, ING-ERR-011
@pytest.mark.parametrize("riva_options", [SMALL_WS])
async def test_oversized_event_is_request_too_large_1009(
    started, stream, fake_factory, helpers
) -> None:
    drv = stream("realtime", started)
    await drv.configure()
    big = base64.b64encode(b"\x00" * 6000).decode()
    await drv.ws.send_json({"type": "input_audio_buffer.append", "audio": big})
    outcome = await drv.outcome()
    helpers.assert_row(
        outcome, "request_too_large", "RESOURCE_EXHAUSTED", "error", 1009
    )


# @spec ING-ERR-026
@pytest.mark.parametrize("riva_options", [SMALL_WS])
async def test_message_above_the_receive_cap_closes_1009(
    started, stream, fake_factory, helpers
) -> None:
    drv = stream("realtime", started)
    await drv.configure()
    await drv.ws.send_text("x" * 9000)
    outcome = await drv.outcome()
    assert outcome.close == 1009
    terminal = outcome.terminal
    if terminal is not None:
        assert helpers.event_code(terminal) == "request_too_large"


# ------------------------------------------- the receive observer


# @spec ING-ERR-057, ING-ERR-058, ING-ERR-011
@pytest.mark.parametrize(
    "late",
    [
        {"type": "input_audio_buffer.append", "audio": "AAAA"},
        {"type": "input_audio_buffer.commit"},
        {"type": "input_audio_buffer.clear"},
        {"type": "input_audio_buffer.done"},
    ],
    ids=["append", "commit", "clear", "done"],
)
async def test_traffic_after_done_is_session_terminal(
    late, started, stream, fake_factory, helpers
) -> None:
    drv = stream("realtime", started)
    await drv.configure()
    lease, _block = await _done_with_flush_blocked(drv, fake_factory, helpers)
    await drv.ws.send_json(late)
    outcome = await drv.outcome()
    helpers.assert_row(
        outcome, "session_terminal", "FAILED_PRECONDITION", "error", 1008
    )
    assert outcome.finals == []
    assert "finish" not in lease.names()
    assert lease.names()[-1] == "release"
    assert lease.names().count("flush") == 1


# @spec ING-ERR-057, ING-ERR-058
async def test_update_after_done_is_session_terminal(
    started, stream, fake_factory, helpers
) -> None:
    drv = stream("realtime", started)
    await drv.configure()
    await _done_with_flush_blocked(drv, fake_factory, helpers)
    await drv.ws.send_json(
        {"type": "transcription_session.update", "session": drv.session()}
    )
    outcome = await drv.outcome()
    assert helpers.event_code(outcome.terminal) == "session_terminal"
    assert outcome.close == 1008
    assert outcome.finals == []


# @spec ING-ERR-057
async def test_malformed_envelope_after_done_stays_invalid_event(
    started, stream, fake_factory, helpers
) -> None:
    drv = stream("realtime", started)
    await drv.configure()
    lease, _block = await _done_with_flush_blocked(drv, fake_factory, helpers)
    await drv.ws.send_text("{not json")
    outcome = await drv.outcome()
    assert helpers.event_code(outcome.terminal) == "invalid_event"
    assert outcome.close == 1008
    assert outcome.finals == []
    assert "finish" not in lease.names()


# @spec ING-ERR-057
@pytest.mark.parametrize("riva_options", [SMALL_WS])
async def test_oversized_message_after_done_stays_request_too_large(
    started, stream, fake_factory, helpers
) -> None:
    drv = stream("realtime", started)
    await drv.configure()
    # 0.05 s of PCM16 (about 2.2 KB as base64) keeps the setup append under
    # the 4096-byte envelope cap, so only the late message is oversized.
    await _done_with_flush_blocked(drv, fake_factory, helpers, seconds=0.05)
    await drv.ws.send_text("x" * 6000)
    outcome = await drv.outcome()
    assert outcome.close == 1009
    assert outcome.finals == []


# @spec ING-ERR-057
async def test_quiet_client_after_done_gets_its_one_final(
    started, stream, fake_factory, helpers
) -> None:
    drv = stream("realtime", started)
    await drv.configure()
    lease, block = await _done_with_flush_blocked(drv, fake_factory, helpers)
    block.release()
    outcome = await drv.outcome()
    assert outcome.close == 1000
    assert len(outcome.finals) == 1
    assert lease.names()[-3:] == ["flush", "finish", "release"]


# The observer's boundaries: the final-tail ``feed``, the instant ``release``
# succeeds, and the entry of the final's send.

LATE = {
    "append": {"type": "input_audio_buffer.append", "audio": "AAAA"},
    "commit": {"type": "input_audio_buffer.commit"},
    "update": {"type": "transcription_session.update", "session": {}},
}


def _client(test_app, clock):
    from support.asgi import BlockingSendWebSocket
    from support.realtime import REALTIME_URL, RealtimeClient

    ws = BlockingSendWebSocket(test_app, REALTIME_URL)
    return ws, RealtimeClient(test_app, clock, ws=ws)


async def _opened(client, plugin, fake_factory, helpers, **session):
    from support.realtime import default_session

    await client.open_session(
        default_session(
            plugin.served_model, fake_factory.default_locale, **session
        )
    )
    lease = _lease(fake_factory)
    await client.append(helpers.speech_bytes())
    await helpers.until(lambda: "feed" in lease.names())
    await client.settle()
    return lease


def _after_release(lease) -> list[str]:
    names = lease.names()
    return names[names.index("release") + 1 :] if "release" in names else []


# @spec ING-ERR-057, ING-ERR-058, ING-LIFE-026
@pytest.mark.parametrize("late", sorted(LATE))
async def test_traffic_during_the_final_tail_feed_is_session_terminal(
    late, started, test_app, clock, fake_factory, helpers
) -> None:
    ws, client = _client(test_app, clock)
    # 8 kHz input leaves the resampler's lookahead as a final tail to feed.
    lease = await _opened(
        client,
        started,
        fake_factory,
        helpers,
        input_audio_params={"sample_rate_hz": 8_000, "num_channels": 1},
    )
    tail = Block()
    lease.script("feed", tail)
    await client.done_event()
    await asyncio.wait_for(tail.entered.wait(), 2)
    assert lease.names().count("feed") == 2, "no tail feed after done"

    try:
        await client.send(LATE[late])
        await client.outcome()
    finally:
        tail.release()  # the owner cancels it; never left parked

    terminal = client.terminal
    assert terminal is not None and terminal["type"] == "error", client.events
    assert helpers.event_code(terminal) == "session_terminal"
    assert client.close_code == 1008
    assert client.of_type(helpers.COMPLETED_EVENT) == []
    # The open lease's running tail ``feed`` is cancelled; cleanup is
    # ``abort`` then ``release``, with no ``flush`` or ``finish``.
    assert lease.names()[-2:] == ["abort", "release"]
    assert "flush" not in lease.names() and "finish" not in lease.names()
    assert lease.names().count("release") == 1


# @spec ING-ERR-057, ING-ERR-058
async def test_traffic_seen_during_release_is_session_terminal(
    started, test_app, clock, fake_factory, helpers
) -> None:
    ws, client = _client(test_app, clock)
    lease = await _opened(client, started, fake_factory, helpers)
    # ``release`` is held after ``finish`` settled: the final's send has not
    # begun, so the observer is still reading.
    release = Block()
    lease.script("release", release)
    await client.done_event()
    try:
        await asyncio.wait_for(release.entered.wait(), 2)
        read = ws.consumed
        ws.push_json(LATE["append"])
        await clock.settle()
        assert ws.consumed == read + 1, "the observer did not read the message"
    finally:
        release.release()  # the running release is not cancelled
    await client.outcome()

    assert helpers.event_code(client.terminal) == "session_terminal"
    assert client.close_code == 1008
    assert client.of_type(helpers.COMPLETED_EVENT) == []
    assert lease.names()[-3:] == ["flush", "finish", "release"]
    assert "abort" not in lease.names()
    assert lease.names().count("release") == 1
    assert _after_release(lease) == [], "a lease call after release"
    # The running ``release`` completed: a cancelled one leaves the lease
    # unreleased and hands the session to the factory's cleanup.
    assert lease.state == "released", "the running release was cancelled"
    assert fake_factory.handed_over == [], "the running release was cancelled"


# @spec ING-ERR-057, ING-ERR-058
async def test_traffic_when_release_succeeds_makes_no_further_lease_call(
    started, test_app, clock, fake_factory, helpers
) -> None:
    # A race invariant: whichever of the observer and the final's send comes
    # first, no lease call follows ``release`` and one outcome stands.
    ws, client = _client(test_app, clock)
    lease = await _opened(client, started, fake_factory, helpers)
    # The late message arrives at the instant ``release`` succeeds, before
    # the adapter can begin the final's send.
    lease.after("release", lambda: ws.push_json(LATE["append"]))
    await client.done_event()
    await client.outcome()

    assert lease.names()[-3:] == ["flush", "finish", "release"]
    assert lease.names().count("release") == 1
    assert "abort" not in lease.names()
    assert _after_release(lease) == [], "a lease call after release"
    completed = client.of_type(helpers.COMPLETED_EVENT)
    if completed:
        # The final's send began before the observer saw the message: the
        # final is irrevocable and the message is not classified.
        assert len(completed) == 1
        assert client.close_code == 1000
        assert client.of_type("error") == []
    else:
        assert helpers.event_code(client.terminal) == "session_terminal"
        assert client.close_code == 1008


# @spec ING-ERR-057, ING-NIMWS-023
async def test_once_the_final_send_began_no_further_traffic_is_read(
    started, test_app, clock, fake_factory, helpers
) -> None:
    ws, client = _client(test_app, clock)
    lease = await _opened(client, started, fake_factory, helpers)
    # ``flush`` returns new text, so an adapter that sent a closing delta for
    # it would send that delta first.
    lease.script("flush", Return("the final words"))
    # Stall the first send after ``release``: the final's send has begun.
    lease.after("release", lambda: setattr(ws, "block", True))
    await client.done_event()
    await helpers.until(lambda: ws.stalled == 1)
    (stalled,) = ws.parked
    first = json.loads(stalled["text"])
    assert first["type"] == helpers.COMPLETED_EVENT, (
        "the first send after release is not the final's completed event"
    )
    assert first["transcript"] == "the final words"
    read = ws.consumed

    await client.send(LATE["append"])
    await client.send(LATE["update"])
    await clock.settle()
    assert ws.consumed == read, "traffic read after the final's send began"

    ws.block = False
    ws.unblocked.set()
    await client.outcome()
    assert len(client.of_type(helpers.COMPLETED_EVENT)) == 1
    assert client.of_type("error") == []
    assert client.close_code == 1000
    # No feed returned a hypothesis, so any delta would be a closing delta
    # for ``flush``'s text; the final transcript arrives only in completed.
    assert DELTA not in client.types(), "a closing delta was sent"
    assert lease.names()[-3:] == ["flush", "finish", "release"]
    assert _after_release(lease) == []


# @spec ING-ERR-057
async def test_the_observer_holds_at_most_one_message(
    started, test_app, clock, fake_factory, helpers
) -> None:
    ws, client = _client(test_app, clock)
    lease = await _opened(client, started, fake_factory, helpers)
    flush = Block()
    abort = Block()
    lease.script("flush", flush)
    lease.script("abort", abort)
    await client.done_event()
    await asyncio.wait_for(flush.entered.wait(), 2)
    read = ws.consumed

    # Two messages at once: the first decides; the second is never read.
    ws.push_json(LATE["append"])
    ws.push_text("{not json")
    try:
        await asyncio.wait_for(abort.entered.wait(), 2)
        await clock.settle()
        assert ws.consumed == read + 1, "the observer read past one message"
    finally:
        flush.release()  # the owner cancels it; never left parked
        abort.release()
    await client.outcome()
    assert helpers.event_code(client.terminal) == "session_terminal"
    assert client.close_code == 1008
    assert client.of_type(helpers.COMPLETED_EVENT) == []
    assert lease.names()[-3:] == ["flush", "abort", "release"]


# @spec ING-ERR-058
async def test_grpc_observer_sees_only_cancellation_after_half_close(
    started, stream, fake_factory, helpers
) -> None:
    drv = stream("grpc", started)
    await drv.configure()
    lease, _block = await _done_with_flush_blocked(drv, fake_factory, helpers)
    drv.context.cancel()
    outcome = await drv.outcome()
    assert outcome.status != "FAILED_PRECONDITION"
    assert outcome.code != "session_terminal"
    assert outcome.finals == []
    assert lease.names()[-1] == "release"


# ------------------------------------------------------- send bound


# @spec ING-ERR-051
async def test_a_send_beyond_the_send_bound_ends_with_no_further_sends(
    started, test_app, clock, fake_factory
) -> None:
    from support.asgi import BlockingSendWebSocket
    from support.realtime import REALTIME_URL, RealtimeClient, default_session

    ws = BlockingSendWebSocket(test_app, REALTIME_URL)
    client = RealtimeClient(test_app, clock, ws=ws)
    await client.open_session(
        default_session(started.served_model, fake_factory.default_locale)
    )
    ws.block = True  # the client stops reading
    await clock.advance(11)  # first-audio bound: the ``idle_timeout`` error
    await client.settle()
    assert ws.stalled == 1
    await clock.advance(61)  # ``response_send_timeout`` of that send
    await ws.wait_closed()
    # The transport-failure path: no further event and no close attempt.
    assert ws.stalled == 1
    lease = fake_factory.leases[0]
    assert lease.names()[-2:] == ["abort", "release"]


# @spec ING-ERR-051
async def test_pending_events_are_sent_before_the_close(
    started, stream, clock, fake_factory, helpers
) -> None:
    drv = stream("realtime", started)
    await drv.configure()
    await clock.advance(11)
    await drv.outcome()
    kinds = [m["type"] for m in drv.ws.received]
    assert kinds[-1] == "websocket.close"
    assert kinds[-2] == "websocket.send"
    error = json.loads(drv.ws.received[-2]["text"])
    assert helpers.event_code(error) == "idle_timeout"
