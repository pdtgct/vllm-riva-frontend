"""The stream owner behind ``StreamingRecognize``: order, finals, disconnects.

Each test drives the servicer with ``support.riva.StreamingCall`` and
observes the wire result, the fake lease's record and the plugin's owner
registry.
"""

from __future__ import annotations

import grpc
import pytest
from support.audio import pcm16_bytes, pcm16_sine, wav_pcm16
from support.fake_session import Block, Raise, Return, failed

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

    # @spec ING-CORE-021, ING-CORE-010, ING-CORE-009
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


# @spec ING-LIFE-058
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


class TestSendBound:
    @pytest.fixture
    def riva_options(self):
        return {"response_send_timeout": 5}

    # @spec ING-LIFE-051, ING-ADM-016, ING-LIFE-038
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

    # @spec ING-LIFE-051
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

    # @spec ING-LIFE-051, ING-ERR-015
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


# @spec ING-LIFE-051, ING-LIFE-053
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


# @spec ING-LIFE-051, ING-ERR-059
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


# @spec ING-LIFE-058, ING-EP-060, ING-EP-059
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


# @spec ING-LIFE-058, ING-LIFE-035, ING-EP-060
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
