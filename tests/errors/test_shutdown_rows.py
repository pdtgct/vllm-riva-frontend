"""Shutdown rows of the phase table: finalized (``OK``, 1001) or aborted.

Core's ``drain_endpoint_plugins`` sets ``state.draining`` and calls the
plugin's ``drain(state, timeout)``; the tests do the same with the plugin's
``FakeClock``. With a 20 s timeout, ``D`` is 15 s away (``engine_reserve`` is
5 s) and the settle point 11.25 s (``D - min(5 s, 15 s / 4)``), where the
trigger is raised. Each stream sends audio first, so the first-audio bound
(10 s) is disarmed and the idle bound (45 s) is far away.
"""

from __future__ import annotations

import asyncio

import pytest
from support.fake_session import Block, Raise, failed

SURFACES = ("grpc", "realtime")
TIMEOUT = 20.0
SETTLE = 11.25
DEADLINE = 15.0


@pytest.fixture(params=SURFACES)
def surface(request) -> str:
    return request.param


def _lease(fake_factory):
    assert fake_factory.leases, "open was not called"
    return fake_factory.leases[-1]


async def _drain(app_state, plugin) -> asyncio.Task[None]:
    from support.app import drain_plugin

    task = asyncio.create_task(drain_plugin(plugin, app_state, TIMEOUT))
    await asyncio.sleep(0)
    return task


async def _finish_drain(task, clock) -> None:
    for _ in range(20):
        if task.done():
            break
        await clock.advance(1)
    await asyncio.wait_for(asyncio.shield(task), 10)


# @spec ING-ERR-027, ING-ERR-011
async def test_shutdown_finalization_ends_ok_after_the_final(
    surface, started, app_state, stream, clock, fake_factory, helpers
) -> None:
    fake_factory.final_text = "hello world"
    drv = stream(surface, started)
    await drv.configure()
    await drv.send_audio(helpers.speech_bytes())
    lease = _lease(fake_factory)
    await helpers.until(lambda: "feed" in lease.names())
    drain = await _drain(app_state, started)
    await clock.advance(SETTLE + 0.5)
    outcome = await drv.outcome()
    assert lease.names()[-3:] == ["flush", "finish", "release"]
    assert len(outcome.finals) == 1
    if surface == "grpc":
        assert outcome.status == "OK"
        final = outcome.finals[0].results[0]
        assert final.alternatives[0].transcript == "hello world"
        assert outcome.responses[-1] is outcome.finals[0]
    else:
        assert outcome.close == 1001
        assert outcome.events[-1]["type"] == helpers.COMPLETED_EVENT
        assert outcome.terminal is None
    await _finish_drain(drain, clock)


# @spec ING-ERR-059, ING-ERR-011
async def test_failure_after_the_trigger_is_the_shutdown_abort_row(
    surface, started, app_state, stream, clock, fake_factory, helpers
) -> None:
    drv = stream(surface, started)
    await drv.configure()
    await drv.send_audio(helpers.speech_bytes())
    lease = _lease(fake_factory)
    lease.script("flush", Raise(RuntimeError("flush broke")))
    drain = await _drain(app_state, started)
    await clock.advance(SETTLE + 0.5)
    outcome = await drv.outcome()
    # not the ``internal`` row the flush failure would select otherwise
    helpers.assert_row(
        outcome, "service_unavailable", "UNAVAILABLE", "error", 1012
    )
    assert outcome.finals == []
    assert lease.names()[-2:] == ["abort", "release"]
    await _finish_drain(drain, clock)


# @spec ING-ERR-059
async def test_engine_dead_after_the_trigger_is_the_shutdown_abort_row(
    surface, started, app_state, stream, clock, fake_factory, helpers
) -> None:
    drv = stream(surface, started)
    await drv.configure()
    await drv.send_audio(helpers.speech_bytes())
    _lease(fake_factory).script("flush", Raise(failed("engine_dead")))
    drain = await _drain(app_state, started)
    await clock.advance(SETTLE + 0.5)
    outcome = await drv.outcome()
    helpers.assert_row(
        outcome, "service_unavailable", "UNAVAILABLE", "error", 1012
    )
    await _finish_drain(drain, clock)


# @spec ING-ERR-059, ING-ERR-056
async def test_drain_deadline_before_shutdown_finalization_is_the_abort_row(
    surface, started, app_state, stream, clock, fake_factory, helpers
) -> None:
    drv = stream(surface, started)
    await drv.configure()
    await drv.send_audio(helpers.speech_bytes())
    lease = _lease(fake_factory)
    lease.script("flush", Block())
    drain = await _drain(app_state, started)
    await clock.advance(SETTLE + 0.5)
    await helpers.until(lambda: "flush" in lease.names())
    # the shutdown finalization's bounds are later than D: the wait is
    # clipped to D and expires as D, not as finalization_timeout
    await clock.advance(DEADLINE - SETTLE)
    outcome = await drv.outcome()
    helpers.assert_row(
        outcome, "service_unavailable", "UNAVAILABLE", "error", 1012
    )
    assert outcome.code != "finalization_timeout"
    assert "finish" not in lease.names()
    await _finish_drain(drain, clock)


# @spec ING-ERR-059
async def test_drain_deadline_during_a_running_client_finalization(
    surface, started, app_state, stream, clock, fake_factory, helpers
) -> None:
    drv = stream(surface, started)
    await drv.configure()
    await drv.send_audio(helpers.speech_bytes())
    lease = _lease(fake_factory)
    lease.script("flush", Block())
    drain = await _drain(app_state, started)
    await clock.advance(SETTLE - 6)
    await drv.finalize()  # client-selected finalization starts first
    await helpers.until(lambda: "flush" in lease.names())
    await clock.advance(6.5)  # the trigger finds it running
    await clock.advance(DEADLINE - SETTLE)  # D before its own 30 s bound
    outcome = await drv.outcome()
    helpers.assert_row(
        outcome, "service_unavailable", "UNAVAILABLE", "error", 1012
    )
    await _finish_drain(drain, clock)


# @spec ING-ERR-059, ING-ERR-005
async def test_cause_selected_before_the_trigger_is_kept(
    started, app_state, stream, clock, fake_factory, helpers
) -> None:
    drv = stream("grpc", started)
    await drv.configure()
    lease = _lease(fake_factory)
    lease.script("abort", Block())  # idle-timeout cleanup still running
    drain = await _drain(app_state, started)
    await clock.advance(10.5)  # first-audio bound: idle_timeout selected
    await helpers.until(lambda: "abort" in lease.names())
    await clock.advance(DEADLINE - 10.5 + 1)
    outcome = await drv.outcome()
    assert outcome.code == "idle_timeout"
    assert outcome.status == "ABORTED"
    await _finish_drain(drain, clock)
