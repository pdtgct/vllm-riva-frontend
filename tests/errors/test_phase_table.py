"""The phase table, end to end on both streaming surfaces.

Each test drives one phase and cause on the Riva servicer and on the Riva
Realtime adapter and asserts that row's code, gRPC status, Realtime event
and close. Time is virtual: a bound expires when the test advances the
plugin's clock past it, never because the machine was slow. Defaults of the
vehicle leaf apply unless ``riva_options`` sets an option.
"""

from __future__ import annotations

import pytest
from support.fake_session import Block, Raise, failed, refused

SURFACES = ("grpc", "realtime")
ERROR = "error"
FAILED = "conversation.item.input_audio_transcription.failed"


@pytest.fixture(params=SURFACES)
def surface(request) -> str:
    return request.param


def _lease(fake_factory):
    assert fake_factory.leases, "open was not called"
    return fake_factory.leases[-1]


# ------------------------------------------------ refused before config


# @spec ING-ERR-015, ING-ERR-011, ING-ERR-046
async def test_refusal_while_not_serving_is_service_unavailable(
    surface, app_state, plugin, stream, fake_factory, helpers
) -> None:
    # admission opens last in ``start``: before it, the plugin is starting
    drv = stream(surface, plugin)
    if surface == "grpc":
        await drv.send_first(helpers.config_request())
    else:
        assert await drv.connect()
    outcome = await drv.outcome()
    helpers.assert_row(
        outcome, "service_unavailable", "UNAVAILABLE", ERROR, 1013
    )
    assert fake_factory.leases == []
    if surface == "realtime":
        types = [e["type"] for e in outcome.events]
        assert "conversation.created" not in types


# @spec ING-ERR-015, ING-ERR-011, ING-ERR-051
async def test_refusal_at_a_full_ingress_cap_is_busy_load_shed(
    surface, started, stream, fake_factory, helpers
) -> None:
    from vllm_riva_frontend.admission import OwnerKind

    held = []
    for _ in range(10_000):
        admission = started.ingress.try_admit(OwnerKind.stream)
        if admission is None:
            break
        held.append(admission)
    assert held, "the cap admitted nothing"
    try:
        drv = stream(surface, started)
        if surface == "grpc":
            await drv.send_first(helpers.config_request())
        else:
            assert await drv.connect()
        outcome = await drv.outcome()
    finally:
        for admission in held:
            admission.release()
    helpers.assert_row(outcome, "busy", "RESOURCE_EXHAUSTED", ERROR, 1013)
    assert fake_factory.leases == []
    if surface == "realtime":
        assert helpers.event_authority(outcome.terminal) == "load_shed"
        assert len(outcome.events) == 1
    else:
        assert outcome.details.startswith("busy: authority: load_shed")


# @spec ING-ERR-015, ING-ERR-008, ING-ERR-011, ING-ERR-051
@pytest.mark.parametrize("reason", ["capacity", "active_limit"])
async def test_engine_capacity_refusal_is_busy_provider(
    surface, reason, started, stream, fake_factory, helpers
) -> None:
    fake_factory.script("open", Raise(refused(reason)))
    drv = stream(surface, started)
    if surface == "grpc":
        await drv.send_first(helpers.config_request())
    else:
        assert await drv.connect()
        assert (await drv.receive())["type"] == "conversation.created"
        await drv.ws.send_json(
            {"type": "transcription_session.update", "session": drv.session()}
        )
    outcome = await drv.outcome()
    helpers.assert_row(outcome, "busy", "RESOURCE_EXHAUSTED", ERROR, 1013)
    if surface == "grpc":
        assert outcome.details.startswith("busy: authority: provider")
    if surface == "realtime":
        assert helpers.event_authority(outcome.terminal) == "provider"
        types = [e["type"] for e in outcome.events]
        assert "transcription_session.updated" not in types


# @spec ING-ERR-008
async def test_engine_paused_refusal_is_service_unavailable(
    started, stream, fake_factory, helpers
) -> None:
    fake_factory.script("open", Raise(refused("engine_paused")))
    drv = stream("grpc", started)
    await drv.send_first(helpers.config_request())
    outcome = await drv.outcome()
    assert outcome.status == "UNAVAILABLE"
    assert outcome.code == "service_unavailable"


# @spec ING-ERR-008
async def test_locale_refusal_from_open_is_unknown_locale(
    started, stream, fake_factory, helpers
) -> None:
    fake_factory.script("open", Raise(refused("locale")))
    drv = stream("grpc", started)
    await drv.send_first(helpers.config_request())
    outcome = await drv.outcome()
    assert outcome.status == "INVALID_ARGUMENT"
    assert outcome.code == "unknown_locale"


# @spec ING-ERR-008, ING-ERR-010
@pytest.mark.parametrize("reason", ["cadence", "request_id_in_use"])
async def test_server_side_refusals_are_internal(
    reason, started, stream, fake_factory, helpers
) -> None:
    fake_factory.script("open", Raise(refused(reason)))
    drv = stream("grpc", started)
    await drv.send_first(helpers.config_request())
    outcome = await drv.outcome()
    assert outcome.status == "INTERNAL"
    assert outcome.code == "internal"


# ------------------------------------------------------- protocol order


# @spec ING-ERR-016, ING-ERR-011, ING-ERR-029
async def test_wrong_first_message_is_protocol_order(
    surface, started, stream, fake_factory, helpers
) -> None:
    drv = stream(surface, started)
    if surface == "grpc":
        await drv.send_first(helpers.audio_request(helpers.speech_bytes()))
    else:
        await drv.send_first(
            {"type": "input_audio_buffer.append", "audio": "AAAA"}
        )
    outcome = await drv.outcome()
    helpers.assert_row(
        outcome, "protocol_order", "FAILED_PRECONDITION", ERROR, 1008
    )
    assert fake_factory.leases == []


# @spec ING-ERR-016, ING-ERR-011
async def test_later_streaming_config_is_terminal_protocol_order(
    started, stream, fake_factory, helpers
) -> None:
    drv = stream("grpc", started)
    await drv.configure()
    drv.requests.push(helpers.config_request())
    outcome = await drv.outcome()
    assert outcome.status == "FAILED_PRECONDITION"
    assert outcome.code == "protocol_order"
    names = _lease(fake_factory).names()
    assert names[-2:] == ["abort", "release"]
    assert "finish" not in names


# ------------------------------------------------------------ timeouts


# @spec ING-ERR-017, ING-ERR-011, ING-ERR-046
async def test_silent_preconfiguration_is_configuration_timeout(
    surface, started, stream, clock, fake_factory, helpers
) -> None:
    drv = stream(surface, started)
    if surface == "grpc":
        drv.begin()
    else:
        assert await drv.connect()
        assert (await drv.receive())["type"] == "conversation.created"
    await clock.advance(31)
    outcome = await drv.outcome()
    helpers.assert_row(
        outcome,
        "configuration_timeout",
        "DEADLINE_EXCEEDED",
        ERROR,
        1011,
    )
    assert fake_factory.leases == []


# @spec ING-ERR-017
async def test_open_beyond_the_preconfiguration_bound_is_configuration_timeout(
    started, stream, clock, fake_factory, helpers
) -> None:
    fake_factory.script("open", Block())
    drv = stream("grpc", started)
    await drv.send_first(helpers.config_request())
    await clock.advance(31)
    outcome = await drv.outcome()
    assert outcome.status == "DEADLINE_EXCEEDED"
    assert outcome.code == "configuration_timeout"


# @spec ING-ERR-018, ING-ERR-011
async def test_no_first_audio_is_idle_timeout(
    surface, started, stream, clock, fake_factory, helpers
) -> None:
    drv = stream(surface, started)
    await drv.configure()
    await clock.advance(11)
    outcome = await drv.outcome()
    helpers.assert_row(outcome, "idle_timeout", "ABORTED", ERROR, 1011)
    assert outcome.finals == []
    assert _lease(fake_factory).names()[-2:] == ["abort", "release"]


# @spec ING-ERR-018, ING-ERR-011
async def test_idle_after_audio_is_idle_timeout(
    surface, started, stream, clock, fake_factory, helpers
) -> None:
    drv = stream(surface, started)
    await drv.configure()
    await drv.send_audio(helpers.speech_bytes())
    await helpers.until(lambda: "feed" in _lease(fake_factory).names())
    await clock.advance(46)
    outcome = await drv.outcome()
    helpers.assert_row(outcome, "idle_timeout", "ABORTED", ERROR, 1011)


# @spec ING-ERR-018, ING-ERR-013, ING-ERR-011
async def test_engine_expired_is_idle_timeout(
    surface, started, stream, fake_factory, helpers
) -> None:
    drv = stream(surface, started)
    await drv.configure()
    _lease(fake_factory).script("feed", Raise(failed("expired")))
    await drv.send_audio(helpers.speech_bytes())
    outcome = await drv.outcome()
    helpers.assert_row(outcome, "idle_timeout", "ABORTED", ERROR, 1011)


# @spec ING-ERR-019, ING-ERR-011, ING-ERR-046
@pytest.mark.parametrize("riva_options", [{"max_session_duration": 5}])
async def test_max_session_duration_is_request_timeout(
    surface, started, stream, clock, fake_factory, helpers
) -> None:
    drv = stream(surface, started)
    await drv.configure()
    await drv.send_audio(helpers.speech_bytes())
    await clock.advance(6)
    outcome = await drv.outcome()
    helpers.assert_row(
        outcome, "request_timeout", "DEADLINE_EXCEEDED", ERROR, 1011
    )
    assert _lease(fake_factory).names()[-2:] == ["abort", "release"]


# @spec ING-ERR-020, ING-ERR-014, ING-ERR-011
async def test_feed_beyond_lease_call_timeout_is_service_unavailable(
    surface, started, stream, clock, fake_factory, helpers
) -> None:
    drv = stream(surface, started)
    await drv.configure()
    lease = _lease(fake_factory)
    lease.script("feed", Block())
    await drv.send_audio(helpers.speech_bytes())
    await helpers.until(lambda: "feed" in lease.names())
    await clock.advance(46)
    outcome = await drv.outcome()
    # the cancelled feed fails the lease with ``cancelled``; the owner's
    # selected cause (the per-operation bound) is what the client sees
    helpers.assert_row(
        outcome, "service_unavailable", "UNAVAILABLE", ERROR, 1011
    )
    assert lease.names()[-2:] == ["abort", "release"]


# @spec ING-ERR-020
async def test_update_locale_beyond_lease_call_timeout_is_service_unavailable(
    started, stream, clock, fake_factory, helpers
) -> None:
    drv = stream("realtime", started)
    await drv.configure()
    lease = _lease(fake_factory)
    lease.script("update_locale", Block())
    session = drv.session()
    session["input_audio_transcription"] = {
        **session.get("input_audio_transcription", {}),
        "language": "de-DE",
    }
    await drv.ws.send_json(
        {"type": "transcription_session.update", "session": session}
    )
    await helpers.until(lambda: "update_locale" in lease.names())
    await clock.advance(46)
    outcome = await drv.outcome()
    helpers.assert_row(
        outcome, "service_unavailable", "UNAVAILABLE", ERROR, 1011
    )


# @spec ING-ERR-048, ING-ERR-011
async def test_finalization_beyond_its_bound_is_finalization_timeout(
    surface, started, stream, clock, fake_factory, helpers
) -> None:
    drv = stream(surface, started)
    await drv.configure()
    lease = _lease(fake_factory)
    lease.script("flush", Block())
    await drv.send_audio(helpers.speech_bytes())
    await drv.finalize()
    await helpers.until(lambda: "flush" in lease.names())
    await clock.advance(31)
    outcome = await drv.outcome()
    helpers.assert_row(
        outcome,
        "finalization_timeout",
        "DEADLINE_EXCEEDED",
        ERROR,
        1011,
    )
    assert outcome.finals == []
    assert "finish" not in lease.names()


# -------------------------------------------------- lease call failures


# @spec ING-ERR-023, ING-ERR-005, ING-ERR-011
async def test_flush_failure_is_internal_with_no_final(
    surface, started, stream, fake_factory, helpers
) -> None:
    drv = stream(surface, started)
    await drv.configure()
    lease = _lease(fake_factory)
    lease.script("flush", Raise(RuntimeError("flush broke")))
    await drv.send_audio(helpers.speech_bytes())
    await drv.finalize()
    outcome = await drv.outcome()
    helpers.assert_row(outcome, "internal", "INTERNAL", FAILED, 1011)
    assert outcome.finals == []


# @spec ING-ERR-023, ING-ERR-011
async def test_finish_failure_is_internal_with_no_final(
    surface, started, stream, fake_factory, helpers
) -> None:
    drv = stream(surface, started)
    await drv.configure()
    lease = _lease(fake_factory)
    lease.script("finish", Raise(RuntimeError("finish broke")))
    await drv.finalize()
    outcome = await drv.outcome()
    helpers.assert_row(outcome, "internal", "INTERNAL", FAILED, 1011)
    assert outcome.finals == []
    assert "abort" not in lease.names()  # flushed: release only


# @spec ING-ERR-023, ING-ERR-035, ING-ERR-011
async def test_release_failure_is_internal_with_no_final(
    surface, started, stream, fake_factory, helpers
) -> None:
    drv = stream(surface, started)
    await drv.configure()
    _lease(fake_factory).script("release", Raise(RuntimeError("leaked")))
    await drv.finalize()
    outcome = await drv.outcome()
    helpers.assert_row(outcome, "internal", "INTERNAL", FAILED, 1011)
    assert outcome.finals == []


# @spec ING-ERR-023, ING-ERR-005, ING-ERR-010
async def test_unexpected_feed_exception_is_internal(
    surface, started, stream, fake_factory, helpers
) -> None:
    drv = stream(surface, started)
    await drv.configure()
    _lease(fake_factory).script("feed", Raise(KeyError("unexpected")))
    await drv.send_audio(helpers.speech_bytes())
    outcome = await drv.outcome()
    helpers.assert_row(outcome, "internal", "INTERNAL", FAILED, 1011)


# @spec ING-ERR-024, ING-ERR-010, ING-ERR-011
async def test_engine_dead_in_feed_is_service_unavailable_and_escalates(
    surface, started, stream, fake_factory, app_state, helpers
) -> None:
    drv = stream(surface, started)
    await drv.configure()
    _lease(fake_factory).script("feed", Raise(failed("engine_dead")))
    await drv.send_audio(helpers.speech_bytes())
    outcome = await drv.outcome()
    helpers.assert_row(
        outcome, "service_unavailable", "UNAVAILABLE", FAILED, 1011
    )
    await helpers.until(lambda: app_state.request_shutdown.requested)


# @spec ING-ERR-024, ING-ERR-035
async def test_engine_dead_in_release_is_service_unavailable(
    surface, started, stream, fake_factory, helpers
) -> None:
    drv = stream(surface, started)
    await drv.configure()
    _lease(fake_factory).script("release", Raise(failed("engine_dead")))
    await drv.finalize()
    outcome = await drv.outcome()
    helpers.assert_row(
        outcome, "service_unavailable", "UNAVAILABLE", FAILED, 1011
    )
    assert outcome.finals == []


# @spec ING-ERR-013, ING-FE-015
async def test_session_input_capacity_is_buffer_overflow(
    started, stream, fake_factory, helpers
) -> None:
    drv = stream("grpc", started)
    await drv.configure()
    _lease(fake_factory).script("feed", Raise(failed("input_capacity")))
    await drv.send_audio(helpers.speech_bytes())
    outcome = await drv.outcome()
    assert outcome.status == "RESOURCE_EXHAUSTED"
    assert outcome.code == "buffer_overflow"


# @spec ING-ERR-014
async def test_cancelled_lease_with_no_frontend_cause_is_internal(
    started, stream, fake_factory, helpers
) -> None:
    drv = stream("grpc", started)
    await drv.configure()
    _lease(fake_factory).script("feed", Raise(failed("cancelled")))
    await drv.send_audio(helpers.speech_bytes())
    outcome = await drv.outcome()
    assert outcome.status == "INTERNAL"
    assert outcome.code == "internal"


# ------------------------------------------------- primary cause stands


# @spec ING-ERR-005, ING-ERR-035
async def test_cleanup_failure_never_replaces_the_selected_cause(
    surface, started, stream, clock, fake_factory, helpers
) -> None:
    drv = stream(surface, started)
    await drv.configure()
    lease = _lease(fake_factory)
    lease.script("abort", Raise(RuntimeError("abort broke")))
    lease.script("release", Raise(failed("engine_dead")))
    await clock.advance(11)
    outcome = await drv.outcome()
    helpers.assert_row(outcome, "idle_timeout", "ABORTED", ERROR, 1011)


# @spec ING-ERR-005, ING-ERR-010
async def test_factory_open_exception_without_seam_error_is_internal(
    started, stream, fake_factory, helpers
) -> None:
    fake_factory.script("open", Raise(RuntimeError("factory broke")))
    drv = stream("grpc", started)
    await drv.send_first(helpers.config_request())
    outcome = await drv.outcome()
    assert outcome.status == "INTERNAL"
    assert outcome.code == "internal"


# ---------------------------------------------------- deadline precedence


# @spec ING-ERR-056, ING-ERR-046
async def test_earlier_rpc_deadline_wins_as_request_timeout(
    started, stream, clock, fake_factory, helpers
) -> None:
    drv = stream("grpc", started, time_remaining=5.0)
    await drv.configure()
    await clock.advance(6)
    outcome = await drv.outcome()
    assert outcome.status == "DEADLINE_EXCEEDED"
    assert outcome.code == "request_timeout"
    assert outcome.finals == []


# @spec ING-ERR-056
async def test_phase_deadline_wins_a_tie_with_the_rpc_deadline(
    started, stream, clock, fake_factory, helpers
) -> None:
    # the first-audio bound (10 s from open) and the RPC deadline fall due
    # at the same instant: the phase deadline wins
    drv = stream("grpc", started, time_remaining=10.0)
    await drv.configure()
    await clock.advance(10)
    outcome = await drv.outcome()
    assert outcome.status == "ABORTED"
    assert outcome.code == "idle_timeout"


# @spec ING-ERR-056
async def test_later_rpc_deadline_does_not_preempt_the_phase(
    started, stream, clock, fake_factory, helpers
) -> None:
    drv = stream("grpc", started, time_remaining=600.0)
    await drv.configure()
    await clock.advance(11)
    outcome = await drv.outcome()
    assert outcome.code == "idle_timeout"


# @spec ING-ERR-056
async def test_selected_abnormal_cause_is_kept_when_a_later_deadline_passes(
    started, stream, clock, fake_factory, helpers
) -> None:
    drv = stream("grpc", started, time_remaining=60.0)
    await drv.configure()
    lease = _lease(fake_factory)
    # the abort of the idle-timeout cleanup hangs past the RPC deadline
    lease.script("abort", Block())
    await clock.advance(11)
    await helpers.until(lambda: "abort" in lease.names())
    await clock.advance(60)
    outcome = await drv.outcome()
    assert outcome.code == "idle_timeout"
