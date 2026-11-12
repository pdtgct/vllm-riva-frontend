"""The owner's time bounds, driven by the plugin's ``FakeClock``.

Defaults (vehicle leaf): ``preconfiguration_timeout`` 30 s,
``first_audio_timeout`` 10 s, ``session_idle_timeout`` 45 s,
``lease_call_timeout`` 45 s, ``session_finalization_timeout`` 30 s,
``session_cleanup_timeout`` 30 s, ``max_session_duration`` off.
"""

from __future__ import annotations

import asyncio
import contextlib

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


# @spec ING-LIFE-022, ING-LIFE-062
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


# @spec ING-LIFE-042, ING-LIFE-062
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


# @spec ING-LIFE-044, ING-LIFE-062
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


# @spec ING-LIFE-060
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


# @spec ING-LIFE-059
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


# ------------------------------------------ cleanup and the drain deadline

CLEANUP = 30.0


async def _drain(started, app_state, timeout: float):
    """Start the plugin's ``drain`` as core calls it; return ``(task, D)``."""
    from support.app import drain_plugin

    from vllm_riva_frontend.lifecycle import engine_reserve

    deadline = started.clock.monotonic() + timeout - engine_reserve(timeout)
    task = asyncio.ensure_future(drain_plugin(started, app_state, timeout))
    await started.clock.settle()
    return task, deadline


async def _ended(task) -> None:
    """Let a plugin call end; cancel it if it is still running."""
    done, _ = await asyncio.wait({task}, timeout=2)
    if not done:
        task.cancel()
        await asyncio.wait({task}, timeout=2)
    with contextlib.suppress(BaseException):
        task.result()


def _fail_with_release_held(call, lease) -> Block:
    """Fail the lease's next ``feed``; keep ``release`` suspended."""
    lease.script("feed", Raise(failed("expired")))
    release = Block()
    lease.script("release", release)
    call.audio(PIECE)
    return release


# @spec ING-LIFE-059
@pytest.mark.parametrize("when", ["graceful", "draining_only", "after_d"])
async def test_cleanup_bound_fixed_at_start_d_during_graceful_else_cleanup_timeout(  # noqa: E501
    riva, fake_factory, started, app_state, drain_trigger, when
):
    call, lease = await _opened(riva, fake_factory)
    drain = None
    if when == "graceful":
        # ``D`` is 90 s away: the cleanup bound is the time remaining to it,
        # not capped by ``session_cleanup_timeout`` (30 s).
        drain, deadline = await _drain(started, app_state, 100.0)
        release = _fail_with_release_held(call, lease)
        await release.entered.wait()
        start = call.clock.monotonic()
        expires = deadline
    elif when == "draining_only":
        # Core set ``state.draining`` and no ``drain`` call runs.
        app_state.draining = True
        release = _fail_with_release_held(call, lease)
        await release.entered.wait()
        start = call.clock.monotonic()
        expires = start + CLEANUP
    else:
        # Cleanup begins at ``D``, when the outstanding feed's wait ends.
        lease.script("feed", Block())
        release = Block()
        lease.script("release", release)
        call.audio(PIECE)
        drain, deadline = await drain_trigger()
        await call.advance(deadline - call.clock.monotonic() + 0.01)
        await release.entered.wait()
        start = call.clock.monotonic()
        expires = start + CLEANUP
    try:
        assert lease.names()[-2:] == ["abort", "release"]
        await call.advance(min(CLEANUP, expires - start) - 0.2)
        assert lease not in fake_factory.handed_over, "the bound ended early"
        if expires - start > CLEANUP:
            await call.advance(expires - call.clock.monotonic() - 0.2)
            assert lease not in fake_factory.handed_over, (
                "the bound was capped by session_cleanup_timeout"
            )

        await call.advance(0.4)
        await call.wait()
        assert fake_factory.handed_over == [lease], "the bound did not expire"
        assert len(started.registry.records) == 0
    finally:
        release.release()
        if drain is not None and when == "graceful":
            await _ended(drain)


# @spec ING-LIFE-060
@pytest.mark.parametrize("where", ["serving", "stop"])
@pytest.mark.parametrize("outstanding", ["abort", "release"])
async def test_expiry_cancels_release_or_abort_then_runs_and_cancels_release_task(  # noqa: E501
    riva, fake_factory, started, app_state, where, outstanding
):
    from support.app import stop_plugins

    call, lease = await _opened(riva, fake_factory)
    engine_abort = Block()
    if outstanding == "abort":
        # ``abort`` waits for the generation task's engine abort.
        lease.script("engine_abort", engine_abort)
    release = Block()
    lease.script("release", release)
    stopper = None
    if where == "serving":
        call.audio(PIECE)  # starts the idle bound
        lease.script("feed", Raise(failed("expired")))
        call.audio(PIECE)
    else:
        stopper = asyncio.ensure_future(stop_plugins(app_state))
    await call.until(lambda: outstanding in lease.names(), what=outstanding)
    await call.settle()
    try:
        await call.advance(CLEANUP - 0.2)
        assert lease not in fake_factory.handed_over
        assert not release.released.is_set()

        await call.advance(0.4)
        await call.wait()

        # ``release`` began after the outstanding call settled, ran to its
        # first suspension and was cancelled there and awaited: the session
        # is with the factory's cleanup before the owner counts as ended.
        assert lease.names()[-2:] == ["abort", "release"]
        assert release.entered.is_set(), "release was cancelled unstarted"
        assert fake_factory.handed_over == [lease]
        assert lease.state == "aborted", "release completed instead"
        assert "teardown" not in lease.names()
        assert len(started.registry.records) == 0
    finally:
        engine_abort.release()
        release.release()
        if stopper is not None:
            await _ended(stopper)


# @spec ING-LIFE-062
@pytest.mark.parametrize("drain", ["draining_only", "graceful"])
async def test_waits_clip_to_d_only_during_graceful_drain(
    riva, fake_factory, started, app_state, drain
):
    call, lease = await _opened(riva, fake_factory)
    lease.script("feed", Block())
    call.audio(PIECE)
    await call.until(lambda: "feed" in lease.names(), what="feed")
    task = None
    try:
        if drain == "draining_only":
            # No ``drain`` call: the 45 s ``lease_call_timeout`` holds.
            app_state.draining = True
            await call.advance(44.8)
            assert not call.done, "the bound was clipped without a drain"
            await call.advance(0.4)
            await call.wait()
            call.assert_status(
                grpc.StatusCode.UNAVAILABLE, "service_unavailable"
            )
        else:
            # During the drain's graceful work the wait ends at ``D``.
            task, deadline = await _drain(started, app_state, 20.0)
            await call.advance(deadline - call.clock.monotonic() - 0.1)
            assert not call.done
            await call.advance(0.2)
            await call.wait()
            call.assert_status(
                grpc.StatusCode.UNAVAILABLE, "service_unavailable"
            )
            assert call.clock.monotonic() < deadline + 1
        assert lease.names() == ["feed", "abort", "release"]
    finally:
        if task is not None:
            await _ended(task)


# ------------------------------------------------- the shared send bound


def _send_plugin(clock, notices):
    """A plugin stand-in for ``surfaces.bounded_send``: no drain running.

    ``drain_announced`` hands out the futures in ``notices`` in order, then
    pending ones.
    """
    from types import SimpleNamespace

    def drain_announced():
        if notices:
            return notices.pop(0)
        return asyncio.get_running_loop().create_future()

    return SimpleNamespace(
        clock=clock,
        config=SimpleNamespace(response_send_timeout=60),
        drain_announced=drain_announced,
        drain_deadline=None,
        send_bounds_passed=False,
        torn_down=False,
        registry=None,
    )


# @spec ING-LIFE-061, ING-EP-110
async def test_a_send_completing_with_the_drain_notice_is_delivered(clock):
    """A send that completed in time is delivered, whatever woke the caller.

    The drain notice and the send complete in the same loop pass, the
    notice first, so the caller resumes before the send's done callbacks
    have run; the send completed within its bound and is delivered.
    """
    from vllm_riva_frontend import surfaces

    loop = asyncio.get_running_loop()
    notice = loop.create_future()
    plugin = _send_plugin(clock, [notice])
    written = loop.create_future()

    async def send():
        await written
        return "sent"

    sending = asyncio.ensure_future(
        surfaces.bounded_send(plugin, send(), record=None)
    )
    await clock.settle()
    assert not sending.done()
    notice.set_result(None)
    written.set_result(None)
    assert await asyncio.wait_for(sending, 2) == "sent"


# @spec ING-EP-106, ING-EP-095
async def test_a_refusal_send_starts_no_task_but_the_send(clock):
    """A refusal that creates no owner starts its send and nothing else.

    The bound is a timer of the clock, not a task: while the send waits,
    the only new task is the send itself, and it still ends at its bound.
    """
    from vllm_riva_frontend import surfaces

    plugin = _send_plugin(clock, [])
    stalled = asyncio.Event()

    async def send():
        await stalled.wait()

    before = asyncio.all_tasks()
    sending = asyncio.ensure_future(
        surfaces.bounded_send(plugin, send(), record=None, terminal=True)
    )
    await clock.settle()
    new = asyncio.all_tasks() - before - {sending}
    assert len(new) == 1, f"tasks besides the send: {new}"
    (send_task,) = new
    await clock.advance(plugin.config.response_send_timeout + 1)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(sending, 2)
    await clock.settle()
    assert send_task.done()
    assert clock.sleepers == 0, "the bound's timer outlived the wait"


# @spec ING-EP-069, ING-LIFE-061
def test_shared_helpers_take_no_vestigial_parameters():
    """The send bound reads the plugin alone; discovery always gets ``discard``.

    ``bounded_send`` takes no ``owner`` and ``settle_shielded`` no
    ``plugin``; ``discover_session_factory`` has no default ``discard``, so
    every caller decides how a discarded factory is settled.
    """
    import inspect

    from vllm_riva_frontend import lease, surfaces

    assert "owner" not in inspect.signature(surfaces.bounded_send).parameters
    assert "plugin" not in (
        inspect.signature(surfaces.settle_shielded).parameters
    )
    discard = inspect.signature(lease.discover_session_factory).parameters[
        "discard"
    ]
    assert discard.default is inspect.Parameter.empty
    assert not hasattr(lease, "_close_factory")


# @spec ING-EP-106, ING-EP-100
async def test_the_real_clock_timer_is_a_loop_timer_not_a_task():
    """``RealClock.timer`` resolves at its deadline with no task running it.

    A deadline already passed resolves at once; cancelling the future
    cancels the loop's timer.
    """
    from vllm_riva_frontend.clock import RealClock

    clock = RealClock()
    before = asyncio.all_tasks()
    passed = clock.timer(clock.monotonic() - 1)
    assert passed.done()
    soon = clock.timer(clock.monotonic() + 0.01)
    later = clock.timer(clock.monotonic() + 3600)
    assert asyncio.all_tasks() == before, "a timer started a task"
    await asyncio.wait_for(soon, 2)
    assert not later.done()
    later.cancel()
    await asyncio.sleep(0)
    assert later.cancelled()


# @spec ING-EP-095, ING-EP-110
async def test_a_cancelled_terminal_refusal_attempt_reraises_after_settling(
    clock,
):
    """A refusal's handler cancelled during its terminal attempt re-raises.

    Every send bound has passed, so the refusal gets its one terminal
    attempt; the handler is cancelled during that attempt's turn while the
    send is outstanding, and the registry is closed. The handler cancels
    the send, awaits it through its stubborn cleanup and, once it has
    ended, re-raises the cancellation, not ``TimeoutError``.
    """
    from types import SimpleNamespace

    from vllm_riva_frontend import surfaces

    plugin = _send_plugin(clock, [])
    plugin.send_bounds_passed = True
    plugin.registry = SimpleNamespace(adopt_send=lambda task: False)
    entered = asyncio.Event()
    release = asyncio.Event()
    cancellations: list[bool] = []

    async def send():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellations.append(True)
            await release.wait()
            raise

    sending = asyncio.ensure_future(
        surfaces.bounded_send(plugin, send(), record=None, terminal=True)
    )
    await asyncio.sleep(0)  # the attempt has started; the handler waits
    assert not sending.done()
    sending.cancel()
    await clock.settle()
    assert entered.is_set()
    assert cancellations == [True], "the handler cancelled its send"
    assert not sending.done(), "the handler awaits the send it cancelled"
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(sending, 2)
    assert sending.cancelled()
