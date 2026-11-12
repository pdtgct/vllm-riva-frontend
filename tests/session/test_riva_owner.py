"""The stream owner behind ``StreamingRecognize``: order, finals, disconnects.

Each test drives the servicer with ``support.riva.StreamingCall`` and
observes the wire result, the fake lease's record and the plugin's owner
registry.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import grpc
import pytest
from support.audio import pcm16_bytes, pcm16_sine, wav_pcm16
from support.fake_session import Block, Raise, Return, failed
from support.grpc_harness import RpcAbort
from support.riva import catalog_code

from .conftest import lease_calls


def _lease(fake_factory):
    assert fake_factory.leases, "no lease was opened"
    return fake_factory.leases[0]


# ------------------------------------------------------- first message


# @spec ING-LIFE-001
async def test_audio_before_configuration_is_protocol_order(riva, fake_factory):
    call = riva()
    call.audio(pcm16_bytes(pcm16_sine(0.1)))
    await call.wait()

    call.assert_status(grpc.StatusCode.FAILED_PRECONDITION, "protocol_order")
    assert [c for c in fake_factory.calls if c.name == "open"] == []


# @spec ING-CORE-021, ING-CORE-011
async def test_only_the_configuration_is_read_before_open_returns(
    riva, fake_factory
):
    block = Block()
    fake_factory.script("open", block)
    call = riva()
    call.configure()
    call.audio(pcm16_bytes(pcm16_sine(0.1)))
    await block.entered.wait()
    await call.settle()

    assert call.requests.pulls == 1
    assert len(call.requests.delivered) == 1

    block.release()
    await call.settle()
    # No invented configuration or admission acknowledgement.
    assert call.responses == []
    assert len(call.requests.delivered) == 2
    call.half_close()
    await call.wait()
    call.assert_ok()


class TestFlowControl:
    @pytest.fixture
    def riva_options(self):
        return {"pre_submit_max_samples": 16_000}

    # @spec ING-CORE-021, ING-CORE-032, ING-CORE-009
    async def test_reading_stops_while_feed_awaits_the_engine(
        self, riva, fake_factory
    ):
        piece = pcm16_bytes(pcm16_sine(0.1))  # 1600 samples
        call = riva()
        call.configure()
        await call.settle()
        lease = _lease(fake_factory)
        block = Block()
        lease.script("feed", block)
        for _ in range(40):
            call.audio(piece)
        await block.entered.wait()
        await call.settle()

        read = (len(call.requests.delivered) - 1) * 1600
        handed = sum(lease.accepted)
        assert read - handed <= 16_000 + 1600
        assert len(call.requests.delivered) < 41

        block.release(Return([]))
        await call.settle()
        call.half_close()
        await call.wait()
        assert sum(lease.accepted) == 40 * 1600


# ------------------------------------------------------ finalization


# @spec ING-LIFE-002, ING-LIFE-025, ING-LIFE-017
async def test_half_close_gives_one_final_after_finish_and_release(
    riva, fake_factory
):
    call = riva()
    call.configure()
    await call.settle()
    lease = _lease(fake_factory)
    lease.script("feed", Return(["an interim"]))
    lease.script("flush", Return("the final transcript"))
    block = Block()
    lease.script("release", block)
    call.audio(pcm16_bytes(pcm16_sine(0.2)))
    call.half_close()
    await block.entered.wait()
    await call.settle()

    assert call.finals == []
    assert not call.done

    block.release()
    await call.wait()
    assert call.finals == ["the final transcript"]
    assert lease.names()[-3:] == ["flush", "finish", "release"]
    assert lease.names().count("flush") == 1
    assert lease.names().count("finish") == 1
    assert len(fake_factory.leases) == 1
    call.assert_ok()


# @spec ING-LIFE-003
async def test_zero_audio_half_close_gives_one_empty_final(riva, fake_factory):
    fake_factory.final_text = ""
    call = riva()
    call.configure()
    call.half_close()
    await call.wait()

    lease = _lease(fake_factory)
    assert [n for n in lease.names() if n != "feed"] == [
        "flush",
        "finish",
        "release",
    ]
    assert sum(lease.accepted) == 0
    assert call.finals == [""]
    call.assert_ok()


# @spec ING-LIFE-017
async def test_the_resampler_tail_is_fed_before_flush(riva, fake_factory):
    from vllm_riva_frontend.frontend import (
        ACCEPT_MATRIX,
        StreamingAudioFrontend,
    )

    assert ("LINEAR_PCM", 8000, 1) in ACCEPT_MATRIX
    data = pcm16_bytes(pcm16_sine(0.3, rate=8000)) + b"\x01"  # odd residual
    oracle = StreamingAudioFrontend(encoding="LINEAR_PCM", sample_rate_hz=8000)
    expected = sum(
        len(piece) for piece in oracle.decode(data, max_samples=1 << 20)
    ) + len(oracle.drain())

    call = riva()
    call.configure(sample_rate_hertz=8000)
    call.audio(data)
    call.half_close()
    await call.wait()

    lease = _lease(fake_factory)
    names = lease.names()
    flush_at = names.index("flush")
    assert "feed" not in names[flush_at:]
    assert sum(lease.accepted) == expected
    call.assert_ok()


# @spec ING-LIFE-017, ING-LIFE-026, ING-LIFE-027
async def test_a_non_string_flush_result_fails_the_stream(riva, fake_factory):
    call = riva()
    call.configure()
    await call.settle()
    lease = _lease(fake_factory)
    lease.script("flush", Return(None))
    call.half_close()
    await call.wait()

    # The call returned, so the lease is flushed: release only, since a
    # flushed lease accepts neither ``abort`` nor, after a failed flush,
    # ``finish``.
    call.assert_status(grpc.StatusCode.INTERNAL, "internal")
    assert call.finals == []
    assert lease.names() == ["flush", "release"]
    assert lease.state == "released"


# @spec ING-LIFE-026
async def test_a_failed_flush_aborts_then_releases_without_final(
    riva, fake_factory
):
    call = riva()
    call.configure()
    await call.settle()
    lease = _lease(fake_factory)
    lease.script("flush", Raise(failed("abnormal_end")))
    call.half_close()
    await call.wait()

    assert lease.names() == ["flush", "abort", "release"]
    assert call.finals == []
    call.assert_status(grpc.StatusCode.INTERNAL, "internal")


# @spec ING-LIFE-027, ING-LIFE-020
async def test_finish_failure_releases_only_and_emits_no_final(
    riva, fake_factory
):
    call = riva()
    call.configure()
    await call.settle()
    lease = _lease(fake_factory)
    lease.script("finish", Raise(failed("abnormal_end")))
    call.half_close()
    await call.wait()

    assert lease.names() == ["flush", "finish", "release"]
    assert call.finals == []
    call.assert_status(grpc.StatusCode.INTERNAL, "internal")


# @spec ING-LIFE-020
async def test_release_failure_on_the_normal_path_emits_no_final(
    riva, fake_factory
):
    call = riva()
    call.configure()
    await call.settle()
    lease = _lease(fake_factory)
    lease.script("release", Raise(failed("state")))
    call.half_close()
    await call.wait()

    assert call.finals == []
    call.assert_status(grpc.StatusCode.INTERNAL, "internal")


# @spec ING-LIFE-020
async def test_engine_dead_in_cleanup_keeps_the_primary_and_is_a_fault(
    riva, fake_factory, started, app_state
):
    call = riva()
    call.configure()
    await call.settle()
    lease = _lease(fake_factory)
    lease.script("feed", Raise(failed("expired")))
    lease.script("release", Raise(failed("engine_dead")))
    call.audio(pcm16_bytes(pcm16_sine(0.1)))
    await call.wait()

    call.assert_status(grpc.StatusCode.ABORTED, "idle_timeout")
    assert lease.names() == ["feed", "abort", "release"]
    assert started.fault is not None
    assert app_state.request_shutdown.calls >= 1


# @spec ING-LIFE-020, ING-LIFE-056
async def test_a_cleanup_error_never_replaces_the_primary_code(
    riva, fake_factory, started
):
    call = riva()
    call.configure()
    await call.settle()
    lease = _lease(fake_factory)
    lease.script("feed", Raise(failed("expired")))
    lease.script("abort", Raise(failed("abnormal_end")))
    call.audio(pcm16_bytes(pcm16_sine(0.1)))
    await call.wait()

    call.assert_status(grpc.StatusCode.ABORTED, "idle_timeout")
    assert lease.names()[-2:] == ["abort", "release"]
    assert started.fault is None


# @spec ING-LIFE-008
async def test_a_later_streaming_config_is_terminal_protocol_order(
    riva, fake_factory
):
    call = riva()
    call.configure()
    call.audio(pcm16_bytes(pcm16_sine(0.1)))
    call.configure()
    await call.wait()

    call.assert_status(grpc.StatusCode.FAILED_PRECONDITION, "protocol_order")
    lease = _lease(fake_factory)
    assert lease.names()[-2:] == ["abort", "release"]
    assert "finish" not in lease.names()
    assert len(fake_factory.leases) == 1


# --------------------------------------------------------- hypotheses


# @spec ING-CORE-019
@pytest.mark.parametrize("interim", [True, False])
async def test_changed_hypotheses_are_interim_results_only_when_requested(
    riva, fake_factory, interim
):
    call = riva()
    call.configure(interim_results=interim)
    await call.settle()
    lease = _lease(fake_factory)
    lease.script("feed", Return(["hello"]), Return([]), Return(["hello world"]))
    lease.script("flush", Return("hello world."))
    piece = pcm16_bytes(pcm16_sine(0.1))
    for _ in range(3):
        call.audio(piece)
    call.half_close()
    await call.wait()

    assert lease.names().count("feed") >= 3
    expected = ["hello", "hello world"] if interim else []
    assert call.interims == expected
    assert call.finals == ["hello world."]


# ------------------------------------------------------ deferred RIFF


def _deferred(call):
    call.configure(encoding="ENCODING_UNSPECIFIED", sample_rate_hertz=0)


# @spec ING-CORE-021
async def test_deferred_riff_feeds_nothing_until_the_header_resolves(
    riva, fake_factory
):
    wav = wav_pcm16(pcm16_sine(0.2))
    call = riva()
    _deferred(call)
    call.audio(wav[:20])
    await call.settle()

    lease = _lease(fake_factory)
    assert "feed" not in lease.names()

    call.audio(wav[20:])
    call.half_close()
    await call.wait()
    assert sum(lease.accepted) == 3200
    call.assert_ok()


# @spec ING-LIFE-024
async def test_finalization_with_no_riff_bytes_is_an_empty_final(
    riva, fake_factory
):
    fake_factory.final_text = ""
    call = riva()
    _deferred(call)
    call.half_close()
    await call.wait()

    lease = _lease(fake_factory)
    assert [n for n in lease.names() if n != "feed"] == [
        "flush",
        "finish",
        "release",
    ]
    assert call.finals == [""]
    call.assert_ok()


# @spec ING-LIFE-024
async def test_finalization_inside_a_partial_riff_header_is_unsupported_format(
    riva, fake_factory
):
    wav = wav_pcm16(pcm16_sine(0.2))
    call = riva()
    _deferred(call)
    call.audio(wav[:30])
    call.half_close()
    await call.wait()

    call.assert_status(grpc.StatusCode.INVALID_ARGUMENT, "unsupported_format")
    lease = _lease(fake_factory)
    assert lease.names() == ["abort", "release"]
    assert call.finals == []


# --------------------------------------------------------- ownership


# @spec ING-LIFE-063
async def test_one_owner_is_registered_before_the_first_read_until_release(
    riva, fake_factory, started
):
    call = riva()
    await call.requests.waiting.wait()
    assert len(started.registry.records) == 1

    call.configure()
    await call.settle()
    lease = _lease(fake_factory)
    block = Block()
    lease.script("release", block)
    call.half_close()
    await block.entered.wait()
    await call.settle()
    assert len(started.registry.records) == 1

    block.release()
    await call.wait()
    await call.settle()
    assert len(started.registry.records) == 0
    assert started.ingress.count == 0


# ------------------------------------------------------- disconnects


# @spec ING-LIFE-038, ING-LIFE-039, ING-LIFE-035
async def test_grpc_cancellation_during_feed_is_observed_at_once(
    riva, fake_factory, clock
):
    call = riva()
    call.configure()
    await call.settle()
    lease = _lease(fake_factory)
    block = Block()
    lease.script("feed", block)
    call.audio(pcm16_bytes(pcm16_sine(0.1)))
    await block.entered.wait()
    before = clock.monotonic()

    call.cancel()
    await call.settle()

    assert clock.monotonic() == before  # no virtual time passed
    assert lease.names() == ["feed", "abort", "release"]
    assert lease.state == "released"
    assert call.finals == []


# @spec ING-LIFE-038
async def test_grpc_cancellation_of_a_flushed_lease_releases_only(
    riva, fake_factory
):
    call = riva()
    call.configure()
    await call.settle()
    lease = _lease(fake_factory)
    block = Block()
    lease.script("finish", block)
    call.half_close()
    await block.entered.wait()

    call.cancel()
    await call.settle()

    assert "abort" not in lease.names()
    assert lease.names()[-1] == "release"
    assert call.finals == []


# @spec ING-LIFE-028
async def test_cancellation_after_finish_settled_suppresses_only_the_final(
    riva, fake_factory
):
    call = riva()
    call.configure()
    await call.settle()
    lease = _lease(fake_factory)
    block = Block()
    lease.script("release", block)
    call.half_close()
    await block.entered.wait()

    call.cancel()
    await call.settle()
    block.release()
    await call.settle()

    assert lease.names() == ["flush", "finish", "release"]
    assert lease.state == "released"
    assert call.finals == []


# @spec ING-LIFE-036
async def test_cancellation_while_open_is_pending_calls_nothing_on_a_lease(
    riva, fake_factory, started
):
    block = Block()
    fake_factory.script("open", block)
    call = riva()
    call.configure()
    await block.entered.wait()

    call.cancel()
    await call.settle()
    block.release()
    await call.settle()

    assert lease_calls(fake_factory) == []
    assert len(started.registry.records) == 0


# @spec ING-LIFE-021
async def test_engine_expired_projects_idle_timeout(riva, fake_factory):
    call = riva()
    call.configure()
    await call.settle()
    lease = _lease(fake_factory)
    lease.script("feed", Raise(failed("expired")))
    call.audio(pcm16_bytes(pcm16_sine(0.1)))
    await call.wait()

    call.assert_status(grpc.StatusCode.ABORTED, "idle_timeout")
    assert lease.names() == ["feed", "abort", "release"]


# ------------------------------------------------------------ sends

PIECE = pcm16_bytes(pcm16_sine(0.1))
#: Virtual time past the old one-second wait of ``stop``: ``stop`` now
#: awaits what it settles with no fixed time limit (ING-EP-093).
STOP_WAIT_PAST = 5.0


class TestSendBound:
    @pytest.fixture
    def riva_options(self):
        return {"response_send_timeout": 5}

    # @spec ING-LIFE-061, ING-ADM-016, ING-LIFE-038
    async def test_a_stalled_write_ends_the_owner_as_a_transport_failure(
        self, riva, fake_factory, started
    ):
        call = riva()
        call.configure()
        await call.settle()
        lease = _lease(fake_factory)
        lease.script("feed", Return(["hello"]))
        call.context.block = True
        call.audio(PIECE)
        await call.until(lambda: call.context.stalled == 1, what="a write")

        try:
            await call.advance(4.9)
            assert lease.names() == ["feed"]
            assert not call.done
            await call.advance(0.2)
            await call.wait()

            # Cleaned up as for a disconnect; no status, nothing more written.
            assert lease.names() == ["feed", "abort", "release"]
            assert call.abandoned
            assert call.responses == [] and call.context.aborts == []
            assert call.context.code is None
            assert len(started.registry.records) == 0
            assert started.ingress.count == 0
        finally:
            call.context.unblocked.set()
        await call.settle()
        assert call.context.stalled == 1
        assert call.responses == []

    # @spec ING-LIFE-061
    async def test_a_stalled_status_is_abandoned_at_the_bound(
        self, riva, fake_factory, started
    ):
        call = riva()
        call.configure()
        await call.settle()
        lease = _lease(fake_factory)
        lease.script("feed", Raise(failed("expired")))
        call.context.block = True
        call.audio(PIECE)
        await call.until(lambda: call.context.stalled == 1, what="the status")
        assert lease.names() == ["feed", "abort", "release"]

        try:
            await call.advance(4.9)
            assert not call.done
            await call.advance(0.2)
            await call.wait()

            assert call.abandoned
            assert call.context.aborts == [] and call.context.code is None
            assert len(started.registry.records) == 0
            assert started.ingress.count == 0
        finally:
            call.context.unblocked.set()

    # @spec ING-LIFE-061, ING-ERR-015
    async def test_a_stalled_refusal_abandons_the_rpc(self, riva, started):
        started.ingress.close()  # as ``drain`` closes it
        refused = riva(start=False)
        refused.context.block = True
        refused.start()
        await refused.until(
            lambda: refused.context.stalled == 1, what="the refusal"
        )
        refusal = refused.context.parked[0]
        assert refusal.details.startswith("service_unavailable:")

        try:
            await refused.advance(4.9)
            assert not refused.done
            await refused.advance(0.2)
            await refused.wait()

            assert refused.abandoned and refused.context.aborts == []
            assert refused.requests.pulls == 0
            assert len(started.registry.records) == 0
        finally:
            refused.context.unblocked.set()


# @spec ING-LIFE-061, ING-LIFE-064, ING-ERR-062
async def test_interrupted_send_takes_shutdown_row_not_transport_failure(
    riva, fake_factory, started, app_state
):
    from support.app import stop_plugins

    call = riva()
    call.configure()
    await call.settle()
    lease = _lease(fake_factory)
    lease.script("feed", Return(["hello"]))
    call.context.block = True
    call.audio(PIECE)
    await call.until(lambda: call.context.stalled == 1, what="a write")
    (record,) = started.registry.records.values()
    owner = record.owner

    # The write is within its bound (60 s) when ``stop`` cancels the owner.
    stopper = asyncio.ensure_future(stop_plugins(app_state))
    try:
        await call.settle()
        await call.until(lambda: call.done, what="the handler's end")
    finally:
        call.context.unblocked.set()
        await call.settle()
    await asyncio.wait_for(stopper, 2)

    assert owner.cause is not None
    assert owner.cause.code == "service_unavailable", "a transport failure"
    # The shutdown abort row's status was offered, as its one attempt.
    assert call.context.stalled == 2
    assert isinstance(call.context.parked[1], RpcAbort)
    assert catalog_code(call.context.parked[1].details) == (
        "service_unavailable"
    )
    assert lease.names() == ["feed", "abort", "release"]
    assert len(started.registry.records) == 0


# @spec ING-LIFE-061, ING-LIFE-053
async def test_a_write_stalled_before_the_drain_ends_by_d(
    riva, fake_factory, started, drain_trigger
):
    call = riva()
    call.configure()
    await call.settle()
    lease = _lease(fake_factory)
    lease.script("feed", Return(["hello"]))
    call.context.block = True
    call.audio(PIECE)
    await call.until(lambda: call.context.stalled == 1, what="a write")

    _, deadline = await drain_trigger()
    assert not call.done
    assert lease.names() == ["feed"]
    # The write's own bound (60 s) is far off; D, set after it began, rules.
    await call.advance(deadline - call.clock.monotonic() + 0.01)
    await call.wait()

    assert call.abandoned and call.context.aborts == []
    assert lease.names() == ["feed", "abort", "release"]
    assert len(started.registry.records) == 0
    assert started.ingress.count == 0


# @spec ING-LIFE-061, ING-ERR-062
async def test_a_status_after_d_to_a_stalled_reader_does_not_wait(
    riva, fake_factory, started, drain_trigger
):
    call = riva()
    call.configure()
    await call.settle()
    lease = _lease(fake_factory)
    lease.script("flush", Block())
    _, deadline = await drain_trigger()
    await call.until(lambda: "flush" in lease.names(), what="flush")
    call.context.block = True

    await call.advance(deadline - call.clock.monotonic() + 0.01)
    await call.settle()

    # The abort row was offered once, at D, and not waited for.
    assert call.done
    assert call.abandoned
    assert call.context.stalled == 1 and call.context.aborts == []
    assert lease.names()[-3:] == ["flush", "abort", "release"]
    assert len(started.registry.records) == 0


# ------------------------------------------------------ cleanup ownership


# @spec ING-LIFE-063, ING-LIFE-064, ING-EP-106, ING-EP-093
async def test_a_second_cancellation_waits_for_cleanup(
    riva, fake_factory, started
):
    call = riva()
    call.configure()
    await call.settle()
    lease = _lease(fake_factory)
    block = Block()
    lease.script("abort", block)
    started.registry.cancel_owners()
    await block.entered.wait()

    # Later cancellations reach the surface while it waits for cleanup.
    try:
        for _ in range(2):
            started.registry.cancel_owners()
            await call.settle()

        # The record and the admission outlive the running cleanup.
        assert not call.done
        assert len(started.registry.records) == 1
        assert started.ingress.count == 1
    finally:
        block.release()
    await call.wait()
    call.assert_status(grpc.StatusCode.UNAVAILABLE, "service_unavailable")
    assert lease.names()[-2:] == ["abort", "release"]
    assert len(started.registry.records) == 0
    assert started.ingress.count == 0


# @spec ING-LIFE-063, ING-LIFE-035, ING-EP-106
async def test_a_continuation_holds_the_record_and_admission_after_the_answer(
    riva, fake_factory, started
):
    call = riva()
    call.configure()
    await call.settle()
    lease = _lease(fake_factory)
    lease.script("feed", Raise(failed("expired")))
    abort = Block(delay_cancel=True)
    lease.script("abort", abort)
    call.audio(PIECE)
    await abort.entered.wait()

    # Past the cleanup budget the stream is answered; the abort that
    # delays its cancellation stays with the record's continuation.
    try:
        await call.advance(started.config.session_cleanup_timeout + 0.1)
        await call.wait()
        call.assert_status(grpc.StatusCode.ABORTED, "idle_timeout")
        records = list(started.registry.records.values())
        assert len(records) == 1, "the record went before its continuation"
        assert records[0].continuation is not None
        assert not records[0].continuation.done()
        assert started.ingress.count == 1
    finally:
        abort.release()
    await call.settle()
    assert records[0].continuation.done()
    assert lease.names() == ["feed", "abort", "release"]
    assert len(started.registry.records) == 0
    assert started.ingress.count == 0


# @spec ING-LIFE-064, ING-EP-079, ING-CORE-032
@pytest.mark.parametrize("state", ["open", "flushed", "open_pending"])
async def test_settlement_cancel_takes_shutdown_row_and_cleans_by_lease_state(
    riva, fake_factory, started, app_state, state
):
    from support.app import stop_plugins

    call = riva()
    call.configure()
    opening = Block()
    if state == "open_pending":
        fake_factory.script("open", opening)
        await opening.entered.wait()
    else:
        await call.settle()
        if state == "flushed":
            fake_factory.leases[0].script("finish", Block())
            call.half_close()
            await call.until(
                lambda: "finish" in fake_factory.leases[0].names(),
                what="finish",
            )
    (record,) = started.registry.records.values()
    owner = record.owner

    # ``stop`` (the plugin torn down) cancels the owner in its settlement.
    stopper = asyncio.ensure_future(stop_plugins(app_state))
    try:
        await call.settle()
        await asyncio.wait_for(asyncio.shield(stopper), 2)
    finally:
        opening.release()
        await call.settle()

    # The selected cause and the cleanup, not the status the client gets.
    assert owner.cause is not None
    assert owner.cause.code == "service_unavailable"
    if state == "open":
        assert _lease(fake_factory).names() == ["abort", "release"]
    elif state == "flushed":
        assert _lease(fake_factory).names() == ["flush", "finish", "release"]
    else:
        assert lease_calls(fake_factory) == []
    for lease in fake_factory.leases:
        assert "teardown" not in lease.names()
        assert lease.teardown_calls == 0
    assert len(started.registry.records) == 0
    assert started.ingress.count == 0


# ------------------------------------- terminal sends, refusals and the drain


def _resisting(
    context, *, suppress: bool = False, on_park=None, on_cancel=None
) -> SimpleNamespace:
    """Make ``context``'s parked calls resist cancellation.

    A parked ``write`` or ``abort`` that is cancelled counts it, then either
    delays the cancellation until ``settled`` is set or, with ``suppress``,
    swallows it and goes on as if the client had read it. ``tasks`` are the
    tasks of the parked calls, in order. ``on_park`` is called as a call
    parks, ``on_cancel`` as a parked call is cancelled.
    """
    state = SimpleNamespace(settled=asyncio.Event(), cancellations=0, tasks=[])

    async def stall(item) -> None:
        if not context.block:
            return
        context.stalled += 1
        context.parked.append(item)
        state.tasks.append(asyncio.current_task())
        if on_park is not None:
            on_park()
        try:
            await context.unblocked.wait()
        except asyncio.CancelledError:
            state.cancellations += 1
            if on_cancel is not None:
                on_cancel()
            if suppress:
                return
            await state.settled.wait()
            raise

    context._stall = stall
    return state


# @spec ING-LIFE-061, ING-EP-106
async def test_a_final_stalled_before_the_drain_ends_by_d(
    riva, fake_factory, started, drain_trigger
):
    call = riva()
    call.configure()
    await call.settle()
    lease = _lease(fake_factory)
    call.context.block = True
    call.half_close()
    await call.until(lambda: call.context.stalled == 1, what="the final")
    assert lease.names() == ["flush", "finish", "release"]
    # The owner's cleanup has ended; the handler still holds the record.
    assert len(started.registry.records) == 1

    _, deadline = await drain_trigger()
    assert not call.done
    await call.advance(deadline - call.clock.monotonic() + 0.01)

    assert call.done, "the final's write outlived D"
    await call.wait()
    assert call.abandoned
    assert call.responses == [] and call.context.aborts == []
    assert len(started.registry.records) == 0
    assert started.ingress.count == 0


# @spec ING-LIFE-061, ING-EP-093, ING-EP-106, ING-LIFE-063
async def test_a_status_stalled_before_the_drain_ends_by_d(
    riva, fake_factory, started, drain_trigger
):
    call = riva()
    call.configure()
    await call.settle()
    lease = _lease(fake_factory)
    lease.script("feed", Raise(failed("expired")))
    call.context.block = True
    call.audio(PIECE)
    await call.until(lambda: call.context.stalled == 1, what="the status")
    assert lease.names() == ["feed", "abort", "release"]

    # Lease cleanup has ended; the record stays while the status is sent.
    (record,) = started.registry.records.values()
    assert not record.task.done()
    assert not started.registry.settled()

    _, deadline = await drain_trigger()
    assert not call.done
    await call.advance(deadline - call.clock.monotonic() + 0.01)

    assert call.done, "the status outlived D"
    await call.wait()
    assert call.abandoned and call.context.aborts == []
    assert call.context.code is None
    assert len(started.registry.records) == 0


# @spec ING-LIFE-061, ING-ERR-015
async def test_a_blocked_refusal_ends_by_an_earlier_d(
    riva, started, drain_trigger
):
    started.ingress.close()  # refused before any drain begins
    refused = riva(start=False)
    refused.context.block = True
    refused.start()
    await refused.until(lambda: refused.context.stalled == 1, what="refusal")
    assert len(started.registry.records) == 0

    # The refusal's own bound (60 s) is far off; a drain then sets D = +3 s.
    _, deadline = await drain_trigger()
    assert not refused.done
    await refused.advance(deadline - refused.clock.monotonic() + 0.01)

    assert refused.done, "the refusal outlived D"
    await refused.wait()
    assert refused.abandoned and refused.context.aborts == []
    assert refused.requests.pulls == 0


# @spec ING-LIFE-061
async def test_a_final_due_after_the_rpc_deadline_is_not_started(
    riva, fake_factory, started
):
    call = riva(time_remaining=20)
    call.configure()
    await call.settle()
    lease = _lease(fake_factory)
    clock = call.clock
    # ``release`` returns just after the RPC deadline; the final is then due.
    lease.after("release", lambda: clock.set(call.context.deadline + 1))

    await call.finish()

    assert lease.names() == ["flush", "finish", "release"]
    # Never started: no final and no status, as for a transport failure.
    assert call.abandoned
    assert call.responses == [] and call.context.aborts == []
    assert call.context.code is None
    assert len(started.registry.records) == 0


class TestUnsettledSend:
    @pytest.fixture
    def riva_options(self):
        return {"response_send_timeout": 5}

    # @spec ING-LIFE-061, ING-EP-106, ING-ADM-016, ING-LIFE-063
    async def test_a_write_delaying_its_cancellation_fails_at_its_bound(
        self, riva, fake_factory, started
    ):
        call = riva()
        call.configure()
        await call.settle()
        lease = _lease(fake_factory)
        lease.script("feed", Return(["hello"]))
        resist = _resisting(call.context)
        call.context.block = True
        call.audio(PIECE)
        await call.until(lambda: call.context.stalled == 1, what="a write")

        await call.advance(5.1)

        # Failed at its bound: the handler waited for no settlement, and
        # cleanup began at once with its own budget.
        assert resist.cancellations == 1
        assert call.done
        await call.wait()
        assert call.abandoned and call.context.aborts == []
        assert lease.names() == ["feed", "abort", "release"]
        (record,) = started.registry.records.values()
        (unsettled,) = record.unsettled_sends
        assert unsettled in record.children and not unsettled.done()
        # The owner has ended and there is no continuation: a record held
        # only by an unsettled send takes no ingress-cap count.
        assert started.ingress.count == 0

        # The record goes once the send settles.
        resist.settled.set()
        await call.settle()
        assert unsettled.done()
        assert len(started.registry.records) == 0
        assert started.ingress.count == 0
        assert call.responses == []

    # @spec ING-LIFE-061, ING-LIFE-063, ING-EP-079, ING-EP-093, ING-EP-106
    @pytest.mark.parametrize("when", ["at_the_bound", "after_the_bound"])
    async def test_a_cancellation_at_the_bound_leaves_the_send_to_stop(
        self, riva, fake_factory, started, app_state, when
    ):
        call = riva()
        call.configure()
        await call.settle()
        lease = _lease(fake_factory)
        lease.script("feed", Return(["hello"]))
        resist = _resisting(call.context)
        call.context.block = True
        call.audio(PIECE)
        await call.until(lambda: call.context.stalled == 1, what="a write")
        (record,) = started.registry.records.values()
        (write,) = resist.tasks

        if when == "at_the_bound":
            # The client cancels as the bound passes, before the handler
            # resumes; grpc.aio then cancels the handler.
            call.clock.set(call.clock.monotonic() + 5)
        else:
            await call.advance(5.1)
        call.context.cancel()
        call.task.cancel()
        await call.settle()

        # Whatever cancelled the handler, the write stays reachable.
        assert call.done
        assert not write.done()
        assert write in record.unsettled_sends
        assert list(started.registry.records.values()) == [record]
        assert started.ingress.count == 0
        assert lease.names() == ["feed", "abort", "release"]

        from support.app import stop_plugins

        # ``stop`` awaits the unsettled write, with no fixed time limit.
        stopper = asyncio.ensure_future(stop_plugins(app_state))
        try:
            await call.advance(STOP_WAIT_PAST)
            assert not stopper.done(), "stop returned with a send unsettled"
            assert not write.done()
        finally:
            resist.settled.set()
            await call.settle()
        assert stopper is not None
        await asyncio.wait_for(stopper, 2)
        assert write.done()
        assert len(started.registry.records) == 0
        assert started.ingress.count == 0
        assert call.responses == []

    # @spec ING-LIFE-063, ING-EP-106
    async def test_one_owner_per_stream_record_persists_through_continuation_and_sends(  # noqa: E501
        self, riva, fake_factory, started
    ):
        call = riva()
        call.configure()
        await call.settle()
        lease = _lease(fake_factory)
        lease.script("feed", Raise(failed("expired")))
        abort = Block(delay_cancel=True)
        lease.script("abort", abort)
        resist = _resisting(call.context)
        call.context.block = True
        call.audio(PIECE)
        await abort.entered.wait()
        (record,) = started.registry.records.values()
        try:
            # Past the cleanup bound the status is due; it stalls and
            # fails its bound, unsettled, while the continuation waits.
            await call.advance(started.config.session_cleanup_timeout + 0.1)
            await call.until(lambda: call.context.stalled == 1, what="status")
            await call.advance(5.1)
            assert call.done
            assert list(started.registry.records.values()) == [record]
            assert record.continuation is not None
            assert not record.continuation.done()
            (unsettled,) = record.unsettled_sends
            # The continuation holds the stream's one ingress-cap count.
            assert started.ingress.count == 1

            abort.release()
            await call.settle()
            assert record.continuation.done()
            assert lease.names() == ["feed", "abort", "release"]
            # Only the unsettled send holds the record now: the count was
            # released, once, when the owner and its continuation ended.
            assert list(started.registry.records.values()) == [record]
            assert not unsettled.done()
            assert started.ingress.count == 0
        finally:
            abort.release()
            resist.settled.set()
            await call.settle()
        assert len(started.registry.records) == 0
        assert started.ingress.count == 0

    # @spec ING-LIFE-061
    async def test_a_write_completing_after_its_bound_has_failed(
        self, riva, fake_factory, started
    ):
        call = riva()
        call.configure()
        await call.settle()
        lease = _lease(fake_factory)
        lease.script("feed", Return(["hello"]))
        call.context.block = True
        call.audio(PIECE)
        await call.until(lambda: call.context.stalled == 1, what="a write")

        # The bound passes and the write then completes, all before the
        # handler resumes.
        call.clock.set(call.clock.monotonic() + 5.1)
        call.context.unblocked.set()
        await call.settle()

        assert len(call.responses) == 1, "the write did complete, late"
        assert call.done, "the handler took the late write as delivered"
        await call.wait()
        assert call.abandoned and call.context.aborts == []
        assert lease.names() == ["feed", "abort", "release"]
        assert len(started.registry.records) == 0
        assert started.ingress.count == 0

    # @spec ING-LIFE-061, ING-LIFE-064, ING-EP-079, ING-EP-093
    async def test_after_stop_a_resisting_send_is_awaited_and_cleanup_aborts(
        self, riva, fake_factory, started, app_state
    ):
        call = riva()
        call.configure()
        await call.settle()
        lease = _lease(fake_factory)
        lease.script("feed", Return(["hello"]))
        resist = _resisting(call.context)
        call.context.block = True
        call.audio(PIECE)
        await call.until(lambda: call.context.stalled == 1, what="a write")
        (record,) = started.registry.records.values()
        owner = record.owner

        from support.app import stop_plugins

        stopper = asyncio.ensure_future(stop_plugins(app_state))
        try:
            await call.settle()

            # No clock time passed: every send bound counted as passed.
            assert call.done, "the handler waited after stop"
            await call.wait()
            assert call.abandoned and call.context.aborts == []
            # Interrupted by the plugin's cancellation, the write is no
            # transport failure: the shutdown abort row's status got its
            # one attempt, and the lease was cleaned up by its state.
            assert owner.cause is not None
            assert owner.cause.code == "service_unavailable"
            assert call.context.stalled == 2
            assert isinstance(call.context.parked[1], RpcAbort)
            assert lease.names() == ["feed", "abort", "release"]
            assert "teardown" not in lease.names()
            # ``stop`` awaits the sends that resist, with no time limit.
            await call.advance(STOP_WAIT_PAST)
            assert not stopper.done(), "stop returned with sends unsettled"
        finally:
            resist.settled.set()
            await call.settle()
        await asyncio.wait_for(stopper, 2)
        assert all(task.done() for task in resist.tasks)
        assert len(started.registry.records) == 0
        assert started.ingress.count == 0

    # @spec ING-EP-079, ING-EP-093, ING-EP-106, ING-LIFE-061, ING-ERR-015
    async def test_stop_awaits_continuations_unsettled_and_kept_sends(
        self, riva, fake_factory, started, app_state
    ):
        # A stream answered while its cleanup goes on: a continuation.
        held = riva()
        held.configure()
        await held.settle()
        held_lease = fake_factory.leases[0]
        held_lease.script("feed", Raise(failed("expired")))
        abort = Block(delay_cancel=True)
        held_lease.script("abort", abort)
        held.audio(PIECE)
        await abort.entered.wait()
        await held.advance(started.config.session_cleanup_timeout + 0.1)
        await held.wait()
        (held_record,) = started.registry.records.values()
        continuation = held_record.continuation
        assert continuation is not None and not continuation.done()

        # A stream whose write failed its bound and has not settled.
        sending = riva()
        sending.configure()
        await sending.settle()
        fake_factory.leases[1].script("feed", Return(["hello"]))
        resist = _resisting(sending.context)
        sending.context.block = True
        sending.audio(PIECE)
        await sending.until(lambda: sending.context.stalled == 1, what="write")
        await sending.advance(5.1)
        assert sending.done
        (unsettled,) = resist.tasks
        (sending_record,) = (
            r for r in started.registry.records.values() if r is not held_record
        )
        assert sending_record.unsettled_sends == {unsettled}
        assert resist.cancellations == 1

        # A refusal, which has no record: its handler awaits its send.
        started.ingress.close()
        refused = riva(start=False)
        refusal = _resisting(refused.context, suppress=True)
        refused.context.block = True
        refused.start()
        await refused.until(
            lambda: refused.context.stalled == 1, what="refusal"
        )
        assert refusal.cancellations == 0

        from support.app import stop_plugins

        stopper = asyncio.ensure_future(stop_plugins(app_state))
        try:
            await refused.settle()
            # The refusal's own bound is far off: it ended because ``stop``
            # made every send bound count as passed.
            assert refusal.cancellations == 1, "the refusal send went on"
            assert refused.done
            await refused.wait()
            assert refused.abandoned
            # The continuation and the unsettled send are awaited, never
            # cancelled by ``stop``, and ``stop`` has no fixed time limit.
            await refused.advance(STOP_WAIT_PAST)
            assert not continuation.cancelled()
            assert not continuation.done()
            assert not unsettled.done()
            assert not stopper.done(), "stop returned with work unsettled"

            abort.release()
            await refused.settle()
            assert continuation.done() and not continuation.cancelled()
            assert held_lease.names() == ["feed", "abort", "release"]
            assert not stopper.done(), "stop left the unsettled send"
        finally:
            abort.release()
            resist.settled.set()
            await refused.settle()
        await asyncio.wait_for(stopper, 2)
        assert unsettled.done()
        for lease in fake_factory.leases:
            assert "teardown" not in lease.names()
        assert len(started.registry.records) == 0
        assert started.ingress.count == 0

    # @spec ING-LIFE-061
    async def test_a_write_suppressing_its_cancellation_still_failed(
        self, riva, fake_factory, started
    ):
        call = riva()
        call.configure()
        await call.settle()
        lease = _lease(fake_factory)
        lease.script("feed", Return(["hello"]))
        resist = _resisting(call.context, suppress=True)
        call.context.block = True
        call.audio(PIECE)
        await call.until(lambda: call.context.stalled == 1, what="a write")

        await call.advance(5.1)
        await call.settle()

        # The write went on after its cancellation; it failed all the same.
        assert resist.cancellations == 1
        assert call.done, "the handler took the late write as delivered"
        await call.wait()
        assert call.abandoned and call.context.aborts == []
        assert lease.names() == ["feed", "abort", "release"]
        call.half_close()
        await call.settle()
        assert call.finals == []
        assert len(started.registry.records) == 0

    # @spec ING-LIFE-061, ING-LIFE-038, ING-LIFE-063, ING-EP-093, ING-EP-106
    async def test_a_cancellation_while_a_failed_write_is_abandoned(
        self, riva, fake_factory, started, app_state
    ):
        call = riva()
        call.configure()
        await call.settle()
        lease = _lease(fake_factory)
        lease.script("feed", Return(["hello"]))
        # The parked write's cancellation (its abandonment at the bound)
        # cancels the handler too, as a drain within the engine reserve
        # would at that moment; the client has not cancelled.
        resist = _resisting(
            call.context, on_cancel=started.registry.cancel_owners
        )
        call.context.block = True
        call.audio(PIECE)
        await call.until(lambda: call.context.stalled == 1, what="a write")
        (record,) = started.registry.records.values()
        (write,) = resist.tasks

        stopper = None
        try:
            await call.advance(5.1)

            assert resist.cancellations == 1
            assert not call.context.cancelled()
            # The failure stood: cleanup as for a disconnect, and the
            # handler ended with no status and nothing further written.
            assert call.done, "the handler waited on a further send"
            await call.wait()
            assert call.abandoned
            assert call.context.stalled == 1 and call.context.aborts == []
            assert lease.names() == ["feed", "abort", "release"]
            assert record.owner.cause.code == "internal"
            assert record.owner.cause.phase == "transport"
            # The unsettled write holds the record (with no ingress-cap
            # count) and stays reachable by ``stop``, which awaits it.
            assert write in record.unsettled_sends and not write.done()
            assert list(started.registry.records.values()) == [record]
            assert started.ingress.count == 0

            from support.app import stop_plugins

            stopper = asyncio.ensure_future(stop_plugins(app_state))
            await call.advance(STOP_WAIT_PAST)
            assert not stopper.done(), "stop returned with a send unsettled"
        finally:
            resist.settled.set()
            await call.settle()
        await asyncio.wait_for(stopper, 2)
        assert write.done()
        assert len(started.registry.records) == 0
        assert started.ingress.count == 0
        assert call.responses == []

    # @spec ING-LIFE-061, ING-ERR-062
    @pytest.mark.parametrize("when", ["first_step", "abandonment"])
    async def test_a_cancellation_during_a_failed_last_attempt_changes_nothing(
        self, riva, fake_factory, started, when
    ):
        call = riva(time_remaining=20)
        call.configure()
        await call.settle()
        lease = _lease(fake_factory)
        clock = call.clock
        lease.script("feed", Raise(failed("expired")))
        # Cleanup returns just after the RPC deadline: the status is then
        # due past its bound and gets its one attempt, which parks.
        lease.after("release", lambda: clock.set(call.context.deadline + 1))
        cancel = started.registry.cancel_owners
        resist = _resisting(
            call.context,
            on_park=cancel if when == "first_step" else None,
            on_cancel=cancel if when == "abandonment" else None,
        )
        call.context.block = True
        call.audio(PIECE)
        try:
            await call.until(lambda: call.done, what="the handler's end")

            # The attempt failed whatever cancelled the handler meanwhile:
            # it ended by the transport path, with no further status.
            assert resist.cancellations == 1
            await call.wait()
            assert call.abandoned
            assert call.context.stalled == 1 and call.context.aborts == []
            assert isinstance(call.context.parked[0], RpcAbort)
            assert lease.names() == ["feed", "abort", "release"]
        finally:
            resist.settled.set()
            await call.settle()
        assert len(started.registry.records) == 0
        assert started.ingress.count == 0
