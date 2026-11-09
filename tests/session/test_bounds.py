"""The owner's time bounds, driven by the plugin's ``FakeClock``.

Defaults (vehicle leaf): ``preconfiguration_timeout`` 30 s,
``first_audio_timeout`` 10 s, ``session_idle_timeout`` 45 s,
``lease_call_timeout`` 45 s, ``session_finalization_timeout`` 30 s,
``session_cleanup_timeout`` 30 s, ``max_session_duration`` off.
"""

from __future__ import annotations

import grpc
import pytest
from support.audio import pcm16_bytes, pcm16_sine
from support.fake_session import Block, Raise, failed

from .conftest import lease_calls

PIECE = pcm16_bytes(pcm16_sine(0.1))


def _lease(fake_factory):
    assert fake_factory.leases, "no lease was opened"
    return fake_factory.leases[0]


async def _opened(riva, fake_factory):
    call = riva()
    call.configure()
    await call.settle()
    return call, _lease(fake_factory)


# --------------------------------------------------------- configuration


# @spec ING-LIFE-022, ING-LIFE-057
async def test_silent_preconfiguration_times_out_and_frees_the_slot(
    riva, fake_factory, started
):
    call = riva()
    await call.advance(29.9)
    assert not call.done

    await call.advance(0.2)
    await call.wait()
    call.assert_status(
        grpc.StatusCode.DEADLINE_EXCEEDED, "configuration_timeout"
    )
    assert [c for c in fake_factory.calls if c.name == "open"] == []
    assert len(started.registry.records) == 0
    assert started.ingress.count == 0


# @spec ING-LIFE-022, ING-LIFE-036
async def test_preconfiguration_expiry_cancels_a_pending_open(
    riva, fake_factory, started
):
    block = Block()
    fake_factory.script("open", block)
    call = riva()
    call.configure()
    await block.entered.wait()

    await call.advance(30.1)
    await call.wait()
    call.assert_status(
        grpc.StatusCode.DEADLINE_EXCEEDED, "configuration_timeout"
    )
    block.release()
    await call.settle()
    assert lease_calls(fake_factory) == []
    assert len(started.registry.records) == 0


# @spec ING-LIFE-022
async def test_configuration_timeout_is_distinct_from_idle_timeout(
    riva, fake_factory
):
    call, _ = await _opened(riva, fake_factory)
    # The preconfiguration bound ended at open; 30 s later the first-audio
    # bound (10 s) has fired as idle, never as configuration.
    await call.advance(30.1)
    await call.wait()
    call.assert_status(grpc.StatusCode.ABORTED, "idle_timeout")


# ---------------------------------------------------- first audio, idle


# @spec ING-LIFE-040, ING-LIFE-052
async def test_first_audio_bound_aborts_once_then_releases(riva, fake_factory):
    call, lease = await _opened(riva, fake_factory)
    await call.advance(9.9)
    assert not call.done

    await call.advance(0.2)
    await call.wait()
    call.assert_status(grpc.StatusCode.ABORTED, "idle_timeout")
    assert lease.names() == ["abort", "release"]


# @spec ING-LIFE-040, ING-LIFE-052
async def test_idle_bound_resets_only_on_accepted_audio(riva, fake_factory):
    call, lease = await _opened(riva, fake_factory)
    await call.advance(5)
    call.audio(PIECE)  # accepted at t=5: idle deadline t=50
    await call.advance(35)
    call.audio(b"")  # no audio
    call.audio(b"\x01")  # a byte residual: no whole sample, nothing accepted
    await call.advance(9.8)  # t=49.8
    assert not call.done

    await call.advance(0.3)  # t=50.1
    await call.wait()
    call.assert_status(grpc.StatusCode.ABORTED, "idle_timeout")
    assert lease.names().count("abort") == 1
    assert lease.names()[-2:] == ["abort", "release"]


# @spec ING-LIFE-040
async def test_accepted_audio_resets_the_idle_bound(riva, fake_factory):
    call, _ = await _opened(riva, fake_factory)
    call.audio(PIECE)  # t=0: deadline 45
    await call.advance(40)
    call.audio(PIECE)  # t=40: deadline 85
    await call.advance(44.8)  # t=84.8
    assert not call.done

    await call.advance(0.3)
    await call.wait()
    call.assert_status(grpc.StatusCode.ABORTED, "idle_timeout")


# @spec ING-LIFE-041
async def test_client_idle_timing_is_suspended_during_a_lease_call(
    riva, fake_factory
):
    call, lease = await _opened(riva, fake_factory)
    block = Block()
    lease.script("feed", block)
    call.audio(PIECE)
    await block.entered.wait()
    await call.advance(20)  # past first_audio (10), below lease_call (45)
    assert not call.done
    assert "abort" not in lease.names()

    block.release()
    await call.settle()
    await call.advance(44.8)
    assert not call.done

    await call.advance(0.4)
    await call.wait()
    call.assert_status(grpc.StatusCode.ABORTED, "idle_timeout")


# ------------------------------------------------------------ lease call


# @spec ING-LIFE-042, ING-LIFE-057
async def test_a_feed_over_lease_call_timeout_breaks_the_lease(
    riva, fake_factory
):
    call, lease = await _opened(riva, fake_factory)
    block = Block()
    lease.script("feed", block)
    call.audio(PIECE)
    await block.entered.wait()

    await call.advance(44.9)
    assert not call.done
    await call.advance(0.2)
    await call.wait()

    call.assert_status(grpc.StatusCode.UNAVAILABLE, "service_unavailable")
    assert lease.names() == ["feed", "abort", "release"]
    assert lease.failure_reason == "cancelled"


# ------------------------------------------------------------- duration


class TestDuration:
    @pytest.fixture
    def riva_options(self):
        return {"max_session_duration": 20}

    # @spec ING-LIFE-043
    async def test_duration_runs_from_open_without_reset(
        self, riva, fake_factory
    ):
        call, lease = await _opened(riva, fake_factory)
        for _ in range(3):
            call.audio(PIECE)
            await call.advance(5)  # t=15
        call.audio(PIECE)
        await call.advance(4.8)  # t=19.8
        assert not call.done

        await call.advance(0.3)
        await call.wait()
        call.assert_status(grpc.StatusCode.DEADLINE_EXCEEDED, "request_timeout")
        assert lease.names()[-2:] == ["abort", "release"]
        assert "finish" not in lease.names()


class TestFinalizationDisarms:
    @pytest.fixture
    def riva_options(self):
        return {"max_session_duration": 12}

    # @spec ING-LIFE-044
    async def test_accepted_finalization_disarms_first_audio_idle_and_duration(
        self, riva, fake_factory
    ):
        call, lease = await _opened(riva, fake_factory)
        block = Block()
        lease.script("flush", block)
        await call.advance(5)
        call.half_close()
        await block.entered.wait()
        await call.advance(10)  # t=15: past first_audio (10) and duration (12)
        assert not call.done
        assert "abort" not in lease.names()

        block.release()
        await call.wait()
        call.assert_ok()
        assert call.finals == ["final words"]


# @spec ING-LIFE-044, ING-LIFE-057
async def test_the_finalization_bound_applies_after_half_close(
    riva, fake_factory
):
    call, lease = await _opened(riva, fake_factory)
    block = Block()
    lease.script("flush", block)
    call.half_close()
    await block.entered.wait()
    await call.advance(29.9)
    assert not call.done

    await call.advance(0.2)
    await call.wait()
    call.assert_status(
        grpc.StatusCode.DEADLINE_EXCEEDED, "finalization_timeout"
    )
    assert lease.names() == ["flush", "abort", "release"]
    assert call.finals == []


# --------------------------------------------------------------- cleanup


# @spec ING-LIFE-037
async def test_an_exhausted_cleanup_budget_hands_the_session_to_the_factory(
    riva, fake_factory, started
):
    call, lease = await _opened(riva, fake_factory)
    lease.script("feed", Raise(failed("expired")))
    lease.script("abort", Block())
    release = Block()
    lease.script("release", release)
    call.audio(PIECE)
    await call.settle()
    assert lease.names() == ["feed", "abort"]

    await call.advance(29.9)
    assert lease.names() == ["feed", "abort"]

    await call.advance(0.2)
    await call.wait()
    assert lease.names() == ["feed", "abort", "release"]
    # The budget was spent, yet ``release`` began: it ran to its first
    # suspension before the owner cancelled it, so the cancellation handed
    # the session to the factory instead of discarding an unstarted call.
    assert release.entered.is_set(), "release was cancelled before it began"
    assert not release.released.is_set()
    assert lease.state == "aborted", "release completed instead"
    assert fake_factory.handed_over == [lease]
    call.assert_status(grpc.StatusCode.ABORTED, "idle_timeout")
    assert len(started.registry.records) == 0


# @spec ING-LIFE-037
async def test_the_cleanup_budget_is_shared_by_abort_and_release(
    riva, fake_factory
):
    call, lease = await _opened(riva, fake_factory)
    lease.script("feed", Raise(failed("expired")))
    abort = Block()
    lease.script("abort", abort)
    release = Block()
    lease.script("release", release)
    call.audio(PIECE)
    await abort.entered.wait()

    await call.advance(20)
    abort.release()
    await call.settle()
    assert lease.names() == ["feed", "abort", "release"]
    assert release.entered.is_set()
    await call.advance(9.8)  # t=29.8 of the one 30 s budget
    assert lease not in fake_factory.handed_over

    await call.advance(0.3)
    await call.wait()
    assert fake_factory.handed_over == [lease]
    assert lease.state == "aborted"
