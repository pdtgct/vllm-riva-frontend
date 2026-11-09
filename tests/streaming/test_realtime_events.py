"""Realtime events mapped to the lease: lifecycle, projection, refusal."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
from support.app import drain_plugin
from support.audio import float_from_pcm16, pcm16_bytes, pcm16_ramp, pcm16_sine
from support.fake_session import (
    Block,
    FakeSessionFactory,
    Raise,
    Return,
    failed,
)
from support.realtime import (
    CLOSE,
    COMPLETED,
    DELTA,
    FAILED,
    RealtimeClient,
    append_event,
    default_session,
    error_authority,
    error_code,
    session_cell,
    update_event,
)

from .conftest import (
    PRE_SUBMIT,
    lease_names,
    opens,
    riva_options,
    served_lease,
    serving,
    until,
)

AUDIO = pcm16_bytes(pcm16_sine(0.02))  # 320 samples
DONE = {"type": "input_audio_buffer.done"}


async def _no_event(client: RealtimeClient, wait: float = 0.05) -> bool:
    try:
        await client.next(timeout=wait)
    except TimeoutError:
        return True
    return False


# @spec ING-NIMWS-021
async def test_conversation_created_precedes_any_client_event(served):
    client = RealtimeClient(served.app, served.clock)
    assert await client.connect()
    first = await client.next()
    assert first["type"] == "conversation.created"
    assert served.factory.leases == []


# @spec ING-NIMWS-023
async def test_first_update_opens_at_560ms_then_acknowledges(served):
    gate = Block()
    served.factory.script("open", gate)
    client = RealtimeClient(served.app, served.clock)
    assert await client.connect()
    await client.next()
    await client.send(update_event(default_session()))
    await asyncio.wait_for(gate.entered.wait(), 2.0)
    assert await _no_event(client), "updated sent before open completed"
    gate.release()
    updated = await client.next()
    assert updated["type"] == "transcription_session.updated"
    assert opens(served) == [{"cadence": "560ms", "locale": "en-US"}]


# @spec ING-NIMWS-023
# @spec ING-CORE-019
async def test_append_done_streams_true_deltas_and_one_completed(served):
    client = RealtimeClient(served.app, served.clock)
    await client.open_session()
    lease = served_lease(served)
    lease.script(
        "feed",
        Return(["hel"]),
        Return(["hel", "hello"]),
        Return([]),
        Return(["hello wor"]),
    )
    for _ in range(4):
        await client.send(append_event(AUDIO))
    await until(lambda: len(lease.accepted) >= 4, what="four feeds")
    await client.send(DONE)
    before = await client.assert_closes(1000)
    deltas = [e["delta"] for e in before if e["type"] == DELTA]
    assert deltas == ["hel", "lo", " wor"]
    completed = [e for e in before if e["type"] == COMPLETED]
    assert len(completed) == 1
    assert before[-1] is completed[0]
    assert completed[0]["is_last_result"] is True
    assert completed[0]["transcript"] == "hello world"
    names = lease_names(served)
    assert names[-3:] == ["flush", "finish", "release"]


# @spec ING-NIMWS-023
async def test_completed_is_sent_only_after_release(served):
    client = RealtimeClient(served.app, served.clock)
    await client.open_session()
    lease = served_lease(served)
    gate = Block()
    lease.script("release", gate)
    await client.send(append_event(AUDIO))
    await until(lambda: lease.accepted, what="feed")
    await client.send(DONE)
    await asyncio.wait_for(gate.entered.wait(), 2.0)
    assert await _no_event(client)
    gate.release()
    before = await client.assert_closes(1000)
    assert [e["type"] for e in before][-1] == COMPLETED


# @spec ING-NIMWS-023
# @spec ING-ERR-022
async def test_transcription_failure_is_a_failed_event_without_completed(
    served,
):
    client = RealtimeClient(served.app, served.clock)
    await client.open_session()
    lease = served_lease(served)
    lease.script("flush", Raise(failed("abnormal_end")))
    await client.send(append_event(AUDIO))
    await until(lambda: lease.accepted, what="feed")
    await client.send(DONE)
    before = await client.assert_closes(1011)
    assert [e["type"] for e in before].count(FAILED) == 1
    assert not [e for e in before if e["type"] == COMPLETED]
    assert "finish" not in lease_names(served)


# @spec ING-NIMWS-023
# @spec ING-LIFE-001
async def test_audio_before_configuration_is_a_protocol_error(served):
    client = RealtimeClient(served.app, served.clock)
    assert await client.connect()
    await client.next()
    await client.send(append_event(AUDIO))
    before = await client.assert_closes(1008)
    assert [error_code(e) for e in before if e["type"] == "error"] == [
        "protocol_order"
    ]
    assert opens(served) == []


MALFORMED = [
    ("text", "{not json"),
    ("json", []),
    ("json", "a string"),
    ("json", {"no_type": True}),
    ("json", {"type": 7}),
    ("json", {"type": "input_audio_buffer.explode"}),
    ("json", {"type": "input_audio_buffer.append", "audio": 12}),
    ("json", {"type": "input_audio_buffer.append"}),
    ("json", {"type": "transcription_session.update", "session": "x"}),
]


@pytest.mark.parametrize(("kind", "payload"), MALFORMED, ids=repr)
@pytest.mark.parametrize("configured", [False, True])
# @spec ING-NIMWS-023
# @spec ING-ERR-033
async def test_malformed_or_unknown_events_are_invalid_event_and_close_1008(
    served, kind, payload, configured
):
    client = RealtimeClient(served.app, served.clock)
    if configured:
        await client.open_session()
    else:
        assert await client.connect()
        await client.next()
    if kind == "text":
        await client.ws.send_text(payload)
    else:
        await client.send(payload)
    before = await client.assert_closes(1008)
    errors = [e for e in before if e["type"] == "error"]
    assert [error_code(e) for e in errors] == ["invalid_event"]
    if configured:
        assert served_lease(served).accepted == []
        assert "flush" not in lease_names(served)
    else:
        assert opens(served) == []


# @spec ING-NIMWS-023
# @spec ING-ERR-057
async def test_traffic_after_done_fails_the_session_with_no_second_feed(served):
    client = RealtimeClient(served.app, served.clock)
    await client.open_session()
    lease = served_lease(served)
    gate = Block()
    lease.script("flush", gate)
    await client.send(append_event(AUDIO))
    await until(lambda: lease.accepted, what="feed")
    await client.send(DONE)
    await asyncio.wait_for(gate.entered.wait(), 2.0)
    await client.send(append_event(AUDIO))
    await served.clock.settle(20)
    gate.release()
    before = await client.assert_closes(1008)
    assert [error_code(e) for e in before if e["type"] == "error"] == [
        "session_terminal"
    ]
    assert not [e for e in before if e["type"] == COMPLETED]
    names = lease_names(served)
    assert "feed" not in names[names.index("flush") :]


# @spec ING-NIMWS-023
async def test_a_later_locale_update_is_applied_then_acknowledged_truthfully(
    served,
):
    client = RealtimeClient(served.app, served.clock)
    await client.open_session()
    lease = served_lease(served)
    session = default_session()
    session["input_audio_transcription"]["language"] = "de-DE"
    await client.send(update_event(session))
    updated = await client.until_type("transcription_session.updated")
    assert updated["type"] == "transcription_session.updated", updated
    assert lease.names().count("update_locale") == 1
    assert lease.locale == "de-DE"
    assert (
        updated["session"]["input_audio_transcription"]["language"] == "de-DE"
    )


# @spec ING-NIMWS-005
async def test_commit_is_acknowledged_and_never_forces_an_engine_step(served):
    client = RealtimeClient(served.app, served.clock)
    await client.open_session()
    lease = served_lease(served)
    await client.send(append_event(AUDIO))
    await until(lambda: lease.accepted, what="feed")
    names = lease_names(served)
    await client.send({"type": "input_audio_buffer.commit"})
    ack = await client.until_type("input_audio_buffer.committed")
    assert ack["type"] == "input_audio_buffer.committed"
    assert lease_names(served) == names
    assert "flush" not in lease_names(served)


# ---------------------------------------------------------------- clear


# @spec ING-NIMWS-024
async def test_clear_waits_for_appends_ahead_of_it_and_never_rewinds(served):
    client = RealtimeClient(served.app, served.clock)
    await client.open_session()
    lease = served_lease(served)
    gate = Block()
    lease.script("feed", gate)
    await client.send(append_event(AUDIO))
    await client.send({"type": "input_audio_buffer.clear"})
    await asyncio.wait_for(gate.entered.wait(), 2.0)
    assert await _no_event(client), "cleared before the append's handoff"
    gate.release()
    ack = await client.until_type("input_audio_buffer.cleared")
    assert ack["type"] == "input_audio_buffer.cleared"
    assert lease.accepted == [320]
    assert "abort" not in lease.names()
    np.testing.assert_array_equal(
        lease.fed_samples, float_from_pcm16(np.frombuffer(AUDIO, "<i2"))
    )


# @spec ING-NIMWS-024
async def test_clear_discards_the_byte_residual(served):
    client = RealtimeClient(served.app, served.clock)
    await client.open_session()
    lease = served_lease(served)
    samples = pcm16_ramp(320, start=1000, step=3)
    await client.send(append_event(b"\x01\x02\x03"))  # one sample + 1 byte
    await client.send({"type": "input_audio_buffer.clear"})
    await client.until_type("input_audio_buffer.cleared")
    kept = len(lease.fed)
    await client.send(append_event(pcm16_bytes(samples)))
    await client.send(DONE)
    await client.assert_closes(1000)
    after = np.concatenate([np.asarray(p) for p in lease.fed[kept:]])
    np.testing.assert_array_equal(after, float_from_pcm16(samples))


async def _fed_after_clear(served, session, before: bytes, after: bytes):
    client = RealtimeClient(served.app, served.clock)
    await client.open_session(session)
    lease = served.factory.leases[-1]
    if before:
        await client.send(append_event(before))
        await client.send({"type": "input_audio_buffer.clear"})
        await client.until_type("input_audio_buffer.cleared")
    kept = len(lease.fed)
    await client.send(append_event(after))
    await client.send(DONE)
    await client.assert_closes(1000)
    pieces = [np.asarray(p) for p in lease.fed[kept:]]
    return np.concatenate(pieces) if pieces else np.zeros(0, np.float32)


# @spec ING-NIMWS-024
async def test_clear_resets_the_resampler_to_its_initial_state(served):
    session = default_session(
        input_audio_params={"sample_rate_hz": 8_000, "num_channels": 1}
    )
    first = pcm16_bytes(pcm16_sine(0.05, rate=8_000, frequency=300.0))
    second = pcm16_bytes(pcm16_sine(0.1, rate=8_000, frequency=700.0))
    cleared = await _fed_after_clear(served, session, first, second)
    fresh = await _fed_after_clear(served, session, b"", second)
    assert fresh.size > 0
    np.testing.assert_array_equal(cleared, fresh)


# @spec ING-NIMWS-024
# @spec ING-NIMWS-009
async def test_clearing_a_partial_riff_header_restarts_the_scan(served):
    from support.audio import wav_pcm16

    samples = pcm16_ramp(640, start=-300, step=1)
    data = wav_pcm16(samples)
    session = default_session(
        input_audio_format="none",
        input_audio_params={"sample_rate_hz": 16_000, "num_channels": 1},
    )
    client = RealtimeClient(served.app, served.clock)
    await client.open_session(session)
    lease = served_lease(served)
    await client.send(append_event(data[:20]))
    await client.send({"type": "input_audio_buffer.clear"})
    await client.until_type("input_audio_buffer.cleared")
    await client.send(append_event(data))
    await client.send(DONE)
    before = await client.assert_closes(1000)
    assert not [e for e in before if e["type"] == "error"]
    np.testing.assert_array_equal(lease.fed_samples, float_from_pcm16(samples))


# ------------------------------------------------------- split and pause


# @spec ING-NIMWS-018
async def test_a_large_append_is_split_into_bounded_feeds_while_reads_pause(
    served,
):
    count = 3 * PRE_SUBMIT + 5
    samples = pcm16_ramp(count, start=-4000, step=1)
    client = RealtimeClient(served.app, served.clock)
    await client.open_session()
    lease = served_lease(served)
    gate = Block()
    lease.script("feed", gate)
    await client.send(append_event(pcm16_bytes(samples)))
    await client.send({"type": "input_audio_buffer.commit"})
    await asyncio.wait_for(gate.entered.wait(), 2.0)
    assert await _no_event(client), "commit handled while feeding"
    gate.release()
    await client.until_type("input_audio_buffer.committed")
    await until(lambda: sum(lease.accepted) >= count, what="all slices")
    assert all(n <= PRE_SUBMIT for n in lease.accepted), lease.accepted
    np.testing.assert_array_equal(
        lease.fed_samples[:count], float_from_pcm16(samples)
    )
    await client.send(DONE)
    before = await client.assert_closes(1000)
    assert not [e for e in before if e["type"] == "error"]


#: The audio frontend's calls that decode or resample audio. ``reset`` (and
#: construction) only initialises buffers, so it is recorded but not counted.
PAYLOAD_CALLS = ("decode", "drain")


def _decode_spy(monkeypatch) -> list[str]:
    """Record every call of the audio frontend's decoding methods.

    The audio frontend is the only decoder and resampler of streaming audio
    (implementation layout), so wrapping its methods on the class sees every
    decode and resample a surface makes. The list records ``decode``,
    ``drain`` and ``reset``, the methods the layout names.
    """
    from vllm_riva_frontend.frontend import StreamingAudioFrontend

    calls: list[str] = []
    for name in ("decode", "drain", "reset"):
        original = getattr(StreamingAudioFrontend, name)

        def spy(self, *args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(StreamingAudioFrontend, name, spy)
    return calls


#: Messages above the 4096-byte envelope cap and within the 8192-byte
#: receive cap. 4000 bytes of audio are 5336 base64 characters. The second
#: is not base64 and the third is not JSON: an adapter that base64-decodes
#: or parses the message before its size check answers ``invalid_audio`` or
#: ``invalid_event`` instead of ``request_too_large``.
OVERSIZED = {
    "audio": json.dumps(append_event(b"\x00" * 4000)),
    "not-base64": json.dumps(
        {"type": "input_audio_buffer.append", "audio": "!" * 5000}
    ),
    "not-json": '{"type": "input_audio_buffer.append", "audio": "' + "A" * 5000,
}


# @spec ING-NIMWS-018, ING-NIMWS-015
@pytest.mark.parametrize("payload", sorted(OVERSIZED))
async def test_an_append_above_the_envelope_cap_is_refused_before_decode(
    clock, monkeypatch, payload
):
    decoded = _decode_spy(monkeypatch)
    options = riva_options(
        ws_event_envelope_max_bytes=4096, ws_receive_max_bytes=8192
    )
    async with serving(clock=clock, options=options) as served:
        client = RealtimeClient(served.app, served.clock)
        # 8 kHz input, so a decode would also resample.
        await client.open_session(session_cell("pcm16", 8_000))
        lease = served_lease(served)
        # Observe from here: configuring may build the frontend and reset it.
        decoded.clear()
        text = OVERSIZED[payload]
        assert 4096 < len(text.encode()) <= 8192
        await client.send_text(text)
        before = await client.assert_closes(1009)
        errors = [e for e in before if e["type"] == "error"]
        assert [error_code(e) for e in errors] == ["request_too_large"]
        assert not [e for e in before if e["type"] == FAILED]
        payload_calls = [name for name in decoded if name in PAYLOAD_CALLS]
        assert payload_calls == [], "decoded or resampled before the refusal"
        assert lease.fed == [], "the payload was decoded and fed"
        assert lease_names(served)[-2:] == ["abort", "release"]
        assert "feed" not in lease_names(served)


# --------------------------------------------------- refusal and shutdown


# @spec ING-NIMWS-016
# @spec ING-ERR-015
async def test_ingress_cap_refusal_accepts_errors_busy_and_closes_1013(clock):
    options = riva_options(load_shed_max_sessions=1)
    async with serving(
        clock=clock, options=options, tasks=("realtime",)
    ) as served:
        holder = RealtimeClient(served.app, served.clock)
        await holder.open_session()
        refused = RealtimeClient(served.app, served.clock)
        assert await refused.connect()
        before = await refused.assert_closes(1013)
        assert [e["type"] for e in before] == ["error"]
        assert error_code(before[0]) == "busy"
        assert error_authority(before[0]) == "load_shed", before[0]
        assert len(served.factory.leases) == 1
        assert len(served.plugin.registry.records) == 1


# @spec ING-NIMWS-016
async def test_a_draining_plugin_refuses_with_service_unavailable(served):
    drain = asyncio.create_task(drain_plugin(served.plugin, served.state, 60.0))
    await served.clock.settle(20)
    client = RealtimeClient(served.app, served.clock)
    assert await client.connect()
    before = await client.assert_closes(1013)
    assert [e["type"] for e in before] == ["error"]
    assert error_code(before[0]) == "service_unavailable"
    assert served.factory.leases == []
    assert not served.plugin.registry.records
    await served.clock.advance(60)
    await asyncio.wait_for(drain, 5.0)


# @spec ING-NIMWS-017
# @spec ING-ERR-027
async def test_shutdown_finalization_sends_completed_then_closes_1001(
    served_through_drain,
):
    served = served_through_drain
    client = RealtimeClient(served.app, served.clock)
    await client.open_session()
    lease = served_lease(served)
    await client.send(append_event(AUDIO))
    await until(lambda: lease.accepted, what="feed")
    drain = asyncio.create_task(drain_plugin(served.plugin, served.state, 60.0))
    await served.clock.advance(49.5)
    before = await client.assert_closes(1001)
    completed = [e for e in before if e["type"] == COMPLETED]
    assert len(completed) == 1 and before[-1] is completed[0]
    assert completed[0]["is_last_result"] is True
    assert completed[0]["transcript"] == "hello world"
    await served.clock.advance(10)
    await asyncio.wait_for(drain, 5.0)


@pytest.mark.parametrize("final_text", ["", "only text"])
# @spec ING-NIMWS-023
async def test_no_interim_completed_events_are_emitted(clock, final_text):
    factory = FakeSessionFactory(final_text=final_text)
    async with serving(clock=clock, factory=factory) as served:
        client = RealtimeClient(served.app, served.clock)
        await client.open_session()
        lease = served_lease(served)
        lease.script("feed", Return(["a"]), Return(["a b"]))
        await client.send(append_event(AUDIO))
        await client.send(append_event(AUDIO))
        await until(lambda: len(lease.accepted) >= 2, what="feeds")
        await client.send(DONE)
        before = await client.assert_closes(1000)
    assert [e["type"] for e in before].count(COMPLETED) == 1
    assert [e["delta"] for e in before if e["type"] == DELTA] == ["a", " b"]
    assert CLOSE not in [e["type"] for e in before]
