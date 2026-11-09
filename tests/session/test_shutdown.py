"""Shutdown finalization: what a stream owner does when ``drain`` triggers.

``drain_trigger()`` starts ``drain`` with an 8 s budget (``D = now + 3``)
and advances virtual time just past the settle point (``D - 0.75``), where
the plugin raises the shutdown trigger for every remaining stream owner.
"""

from __future__ import annotations

import grpc
from support.audio import pcm16_bytes, pcm16_sine
from support.fake_session import Block, Raise, Return, failed
from support.realtime import COMPLETED, append_event, error_code

from .conftest import lease_calls

PIECE = pcm16_bytes(pcm16_sine(0.1))


def _lease(fake_factory):
    assert fake_factory.leases, "no lease was opened"
    return fake_factory.leases[0]


async def _streaming(riva, fake_factory):
    call = riva()
    call.configure()
    await call.settle()
    lease = _lease(fake_factory)
    call.audio(PIECE)
    await call.settle()
    return call, lease


# @spec ING-LIFE-054, ING-LIFE-029, ING-LIFE-017, ING-LIFE-034
async def test_an_idle_stream_is_finalized_with_a_final_and_ok(
    riva, fake_factory, drain_trigger
):
    call, lease = await _streaming(riva, fake_factory)
    lease.script("flush", Return("shutdown final"))
    block = Block()
    lease.script("release", block)
    pulls = call.requests.pulls

    await drain_trigger()
    await block.entered.wait()
    await call.settle()
    # The final waits for a successful release.
    assert call.finals == []

    call.audio(PIECE)  # unread after the trigger
    block.release()
    await call.wait()

    assert lease.names()[-3:] == ["flush", "finish", "release"]
    assert "abort" not in lease.names()
    assert call.finals == ["shutdown final"]
    call.assert_ok()
    assert call.requests.pulls == pulls
    assert len(call.requests.delivered) == 2


# @spec ING-LIFE-053, ING-LIFE-054
async def test_an_outstanding_feed_is_settled_not_cancelled(
    riva, fake_factory, drain_trigger
):
    call, lease = await _streaming(riva, fake_factory)
    block = Block()
    lease.script("feed", block)
    call.audio(PIECE)
    await block.entered.wait()

    await drain_trigger()
    assert lease.state == "open"
    assert "abort" not in lease.names()

    block.release(Return([]))
    await call.wait()
    assert lease.names()[-3:] == ["flush", "finish", "release"]
    assert call.finals == ["final words"]
    call.assert_ok()


# @spec ING-LIFE-053
async def test_a_failure_while_settling_takes_the_shutdown_abort_row(
    riva, fake_factory, drain_trigger
):
    call, lease = await _streaming(riva, fake_factory)
    block = Block()
    lease.script("feed", block)
    call.audio(PIECE)
    await block.entered.wait()

    await drain_trigger()
    # Outside shutdown, rejected_input would project internal.
    block.release(Raise(failed("rejected_input")))
    await call.wait()

    call.assert_status(grpc.StatusCode.UNAVAILABLE, "service_unavailable")
    assert lease.names()[-2:] == ["abort", "release"]
    assert "finish" not in lease.names()


# @spec ING-LIFE-053, ING-LIFE-057
async def test_a_settlement_wait_is_clipped_to_the_drain_deadline(
    riva, fake_factory, drain_trigger
):
    call, lease = await _streaming(riva, fake_factory)
    block = Block()
    lease.script("feed", block)
    call.audio(PIECE)
    await block.entered.wait()

    _, deadline = await drain_trigger()
    await call.advance(deadline - call.clock.monotonic() + 0.01)
    await call.wait()

    # The feed's own bound (45 s) has not expired; D has.
    call.assert_status(grpc.StatusCode.UNAVAILABLE, "service_unavailable")
    assert lease.names()[-3:] == ["feed", "abort", "release"]


# @spec ING-LIFE-048
async def test_a_failed_shutdown_flush_aborts_then_releases(
    riva, fake_factory, drain_trigger
):
    call, lease = await _streaming(riva, fake_factory)
    lease.script("flush", Raise(failed("abnormal_end")))

    await drain_trigger()
    await call.wait()

    call.assert_status(grpc.StatusCode.UNAVAILABLE, "service_unavailable")
    assert lease.names()[-3:] == ["flush", "abort", "release"]
    assert call.finals == []


# @spec ING-LIFE-048
async def test_a_failed_shutdown_finish_releases_only(
    riva, fake_factory, drain_trigger
):
    call, lease = await _streaming(riva, fake_factory)
    lease.script("finish", Raise(failed("abnormal_end")))

    await drain_trigger()
    await call.wait()

    call.assert_status(grpc.StatusCode.UNAVAILABLE, "service_unavailable")
    assert lease.names()[-3:] == ["flush", "finish", "release"]
    assert "abort" not in lease.names()
    assert call.finals == []


# @spec ING-LIFE-048
async def test_the_drain_deadline_arriving_first_aborts_the_finalization(
    riva, fake_factory, drain_trigger
):
    call, lease = await _streaming(riva, fake_factory)
    lease.script("flush", Block())

    _, deadline = await drain_trigger()
    assert not call.done
    await call.advance(deadline - call.clock.monotonic() + 0.01)
    await call.wait()

    call.assert_status(grpc.StatusCode.UNAVAILABLE, "service_unavailable")
    assert lease.names()[-3:] == ["flush", "abort", "release"]
    assert call.finals == []


# @spec ING-LIFE-031
async def test_an_owner_waiting_for_configuration_ends_with_the_shutdown_code(
    riva, fake_factory, drain_trigger
):
    call = riva()
    await call.requests.waiting.wait()

    await drain_trigger()
    await call.wait()

    call.assert_status(grpc.StatusCode.UNAVAILABLE, "service_unavailable")
    assert [c for c in fake_factory.calls if c.name == "open"] == []
    assert call.finals == []


# @spec ING-LIFE-032
async def test_an_open_returning_before_d_is_finalized(
    riva, fake_factory, drain_trigger
):
    fake_factory.final_text = ""
    block = Block()
    fake_factory.script("open", block)
    call = riva()
    call.configure()
    await block.entered.wait()

    await drain_trigger()  # just past the settle point, before ``D``
    block.release()
    await call.wait()

    # The pending ``open`` settled under its own bound: the lease it
    # returned is finalized like any open lease.
    lease = fake_factory.leases[0]
    assert [n for n in lease.names() if n != "feed"] == [
        "flush",
        "finish",
        "release",
    ]
    assert call.finals == [""]
    call.assert_ok()


# @spec ING-LIFE-032
async def test_an_open_still_pending_at_d_is_cancelled_with_the_shutdown_code(
    riva, fake_factory, drain_trigger
):
    block = Block()
    fake_factory.script("open", block)
    call = riva()
    call.configure()
    await block.entered.wait()

    _, deadline = await drain_trigger()
    await call.advance(deadline - call.clock.monotonic() + 0.01)
    await call.wait()

    # ``D`` cancelled ``open``: no lease exists and no lease call is made.
    assert fake_factory.leases == []
    assert lease_calls(fake_factory) == []
    call.assert_status(grpc.StatusCode.UNAVAILABLE, "service_unavailable")
    assert call.finals == []


# @spec ING-LIFE-055, ING-LIFE-034
async def test_a_running_client_finalization_continues_without_reflush(
    riva, fake_factory, drain_trigger
):
    call, lease = await _streaming(riva, fake_factory)
    block = Block()
    lease.script("flush", block)
    call.half_close()
    await block.entered.wait()
    fed_before = sum(lease.accepted)

    await drain_trigger()
    block.release(Return("client final"))
    await call.wait()

    assert lease.names().count("flush") == 1
    assert sum(lease.accepted) == fed_before
    assert lease.names()[-3:] == ["flush", "finish", "release"]
    assert call.finals == ["client final"]
    call.assert_ok()


# @spec ING-LIFE-055
async def test_a_client_finalization_wait_reaching_d_takes_the_shutdown_row(
    riva, fake_factory, drain_trigger
):
    call, lease = await _streaming(riva, fake_factory)
    lease.script("flush", Block())
    call.half_close()
    await call.settle()

    _, deadline = await drain_trigger()
    await call.advance(deadline - call.clock.monotonic() + 0.01)
    await call.wait()

    # The finalization bound (30 s) has not expired; D has.
    call.assert_status(grpc.StatusCode.UNAVAILABLE, "service_unavailable")
    assert lease.names()[-3:] == ["flush", "abort", "release"]


# @spec ING-LIFE-055, ING-LIFE-053
async def test_an_abnormal_cleanup_already_selected_keeps_its_cause(
    riva, fake_factory, drain_trigger
):
    call, lease = await _streaming(riva, fake_factory)
    abort = Block()
    lease.script("feed", Raise(failed("expired")))
    lease.script("abort", abort)
    call.audio(PIECE)
    await abort.entered.wait()

    await drain_trigger()
    abort.release()
    await call.wait()

    call.assert_status(grpc.StatusCode.ABORTED, "idle_timeout")
    assert lease.names()[-2:] == ["abort", "release"]
    assert "flush" not in lease.names()


# @spec ING-LIFE-055, ING-LIFE-034, ING-LIFE-029
async def test_realtime_trigger_cancels_the_receive_observer(
    realtime, first_update, fake_factory, drain_trigger
):
    session = realtime()
    await session.open_session(first_update())
    lease = _lease(fake_factory)
    block = Block()
    lease.script("flush", block)
    await session.send({"type": "input_audio_buffer.done"})
    await block.entered.wait()

    await drain_trigger()
    await session.send(append_event(PIECE))  # not read, not classified
    await session.settle()
    block.release(Return("realtime final"))
    await session.ws.wait_closed()
    await session.drain_events()

    assert "session_terminal" not in map(error_code, session.events)
    assert len(session.of_type(COMPLETED)) == 1
    assert lease.names().count("flush") == 1
    # The client's own finalization keeps its outcome: a successful one
    # closes 1000, not the shutdown finalization's 1001.
    assert session.close_code == 1000


# @spec ING-LIFE-029, ING-LIFE-054
async def test_realtime_shutdown_finalization_completes_then_closes_1001(
    realtime, first_update, fake_factory, drain_trigger
):
    session = realtime()
    await session.open_session(first_update())
    lease = _lease(fake_factory)
    await session.send(append_event(PIECE))
    await session.settle()

    await drain_trigger()
    await session.ws.wait_closed()
    await session.drain_events()

    assert lease.names()[-3:] == ["flush", "finish", "release"]
    assert session.types()[-2:] == [COMPLETED, "websocket.close"]
    assert session.close_code == 1001
