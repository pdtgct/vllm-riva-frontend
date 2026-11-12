"""The seam conformance suite: what the stream owner relies on in a factory.

Every test takes the ``seam`` fixture: an adapter that provides the factory,
the clock its ``drain`` deadline is measured on and the hooks of
``support.seam`` (inject a retained fault, keep a lease call, the next
``open``, a cleanup's retirement wait or a generation task's engine abort
suspended, end a generation task with an exception, and observe the
factory's tasks, hand-overs and cancellations). It is parametrized:

* ``fake``: ``support.seam.FakeSeam`` over ``FakeSessionFactory`` (the
  ``fake_seam`` fixture), on the plugin's ``FakeClock``; the frontend's CPU
  lane.
* ``real``: the model plugin's real factory, marked ``real_factory`` and
  skipped unless the model plugin is installed. The model plugin's
  repository provides its adapter by overriding ``real_seam`` (or ``seam``)
  with one of the same shape, on ``time.monotonic()``, in its seam
  conformance run on the RFC 1 + RFC 2 build.

The suite checks the seam the session design states (``SESS-OWN-003`` to
``SESS-OWN-006``, ``SESS-ABORT-002``, ``SESS-REL-003``), the call order the
owner relies on, the error attributes, each factory obligation (Factory
obligations 1 to 8, ``ING-CORE-023`` to ``ING-CORE-029`` and
``ING-CORE-034``) and the tests ``ING-CORE-030`` lists.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
from typing import Any

import numpy as np
import pytest
from support.clock import settle
from support.fake_session import Raise, failed

#: Real-time allowance around ``drain``'s deadline for loop scheduling.
EARLY = 0.05
LATE = 0.5
#: The longest real time a wait the suite expects to end may take.
LIMIT = 5.0


@pytest.fixture
def real_seam():
    """The model plugin's adapter; its conformance run overrides this."""
    pytest.skip(
        "the model plugin's seam conformance run provides its real_seam"
    )


@pytest.fixture(
    params=["fake", pytest.param("real", marks=pytest.mark.real_factory)]
)
def seam(request):
    if request.param == "fake":
        return request.getfixturevalue("fake_seam")
    return request.getfixturevalue("real_seam")


# ------------------------------------------------------------------ helpers


def _cadence() -> str:
    from vllm_riva_frontend.lease import STREAM_CADENCE

    return STREAM_CADENCE


def _locale(factory: Any) -> str:
    return factory.default_locale or sorted(factory.locales)[0]


async def _open(factory: Any) -> Any:
    lease = await factory.open(cadence=_cadence(), locale=_locale(factory))
    # The generation task takes its first step.
    await settle()
    return lease


def _attrs(exc: BaseException) -> tuple[str | None, str | None]:
    from vllm_riva_frontend.errors import seam_error

    return seam_error(exc)


async def _elapse(seam: Any, seconds: float) -> None:
    """Let ``seconds`` pass on the seam's clock, virtual or real."""
    advance = getattr(seam.clock, "advance", None)
    if advance is not None:
        await advance(seconds)
    else:
        await asyncio.sleep(seconds)
    await settle()


async def _cancelled(task: asyncio.Future[Any]) -> None:
    """Cancel ``task`` and await it; it must end cancelled."""
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, LIMIT)


async def _cancel_abort_in_handler(seam: Any, lease: Any) -> Any:
    """Cancel an ``abort`` while the generation task awaits the engine.

    Returns the ``Block`` that keeps the generation task's cancellation
    handler awaiting the engine's abort.
    """
    engine_abort = seam.slow_engine_abort(lease)
    abort = asyncio.create_task(lease.abort())
    await asyncio.wait_for(engine_abort.entered.wait(), LIMIT)
    await _cancelled(abort)
    generation = seam.generation(lease)
    assert generation is not None and not generation.done()
    return engine_abort


async def _cancel_release(seam: Any, lease: Any) -> None:
    """Run ``release`` as a task, let it suspend, cancel and await it."""
    block = seam.slow(lease, "release")
    release = asyncio.create_task(lease.release())
    await asyncio.wait_for(block.entered.wait(), LIMIT)
    await _cancelled(release)
    assert seam.handed_over(lease)


async def _hand_over(seam: Any) -> Any:
    """An aborted lease whose ``release`` was cancelled after it began."""
    lease = await _open(seam.factory)
    await lease.abort()
    await _cancel_release(seam, lease)
    return lease


async def _break(seam: Any, lease: Any) -> None:
    """Fail an open lease by cancelling a suspended ``feed``."""
    block = seam.slow(lease, "feed")
    samples = np.zeros(1600, dtype=np.float32)
    feed = asyncio.create_task(lease.feed(samples, on_accepted=lambda n: None))
    await asyncio.wait_for(block.entered.wait(), LIMIT)
    await _cancelled(feed)


async def _factory_settlement(seam: Any) -> tuple[list[BaseException], list]:
    """The vehicle's factory settlement steps (``ING-EP-096``).

    ``drain(math.inf)``, ``teardown_all()``, an await of every returned
    task, ``drain(now)``. Returns the faults both drains returned and the
    tasks still running when ``teardown_all`` returned.
    """
    factory = seam.factory
    faults = list(await asyncio.wait_for(factory.drain(math.inf), LIMIT))
    returned = factory.teardown_all()
    still_running = [task for task in returned if not task.done()]
    if returned:
        await asyncio.wait_for(asyncio.wait(returned), LIMIT)
    faults.extend(await factory.drain(seam.clock.monotonic()))
    return faults, still_running


# ------------------------------------------------------- protocol and order


# @spec ING-CORE-022, ING-CORE-033
async def test_factory_and_lease_satisfy_the_structural_protocol(seam):
    from vllm_riva_frontend.lease import SessionFactory, SessionLease

    factory = seam.factory
    assert _cadence() in factory.served_cadences
    assert isinstance(factory.locales, frozenset)
    for member in ("open", "drain", "teardown_all"):
        assert callable(getattr(factory, member))
    lease = await _open(factory)
    for member in (
        "feed",
        "update_locale",
        "flush",
        "finish",
        "abort",
        "release",
    ):
        assert callable(getattr(lease, member))
    if getattr(SessionFactory, "_is_runtime_protocol", False):
        assert isinstance(factory, SessionFactory)
    if getattr(SessionLease, "_is_runtime_protocol", False):
        assert isinstance(lease, SessionLease)
    await lease.abort()
    await lease.release()


# @spec ING-CORE-022, ING-CORE-026
async def test_drain_takes_an_absolute_deadline_and_returns_faults_once(seam):
    factory = seam.factory
    fault = RuntimeError("cleanup failed")
    seam.inject_fault(fault)
    deadline = seam.clock.monotonic() + 5.0

    first = await factory.drain(deadline)
    second = await factory.drain(deadline)

    assert first == [fault]
    assert second == []


# @spec ING-CORE-022
async def test_normal_call_order_feed_flush_finish_release(seam):
    lease = await _open(seam.factory)
    credits: list[int] = []
    samples = np.zeros(1600, dtype=np.float32)

    transcripts = await lease.feed(samples, on_accepted=credits.append)
    assert isinstance(transcripts, list)
    assert credits == [1600]
    assert await lease.update_locale(_locale(seam.factory)) is None
    final = await lease.flush()
    assert isinstance(final, str)
    await lease.finish()
    await lease.release()


# @spec ING-CORE-022
async def test_abnormal_call_order_abort_release_and_flushed_rejects_abort(
    seam,
):
    lease = await _open(seam.factory)
    await lease.abort()
    await lease.release()

    flushed = await _open(seam.factory)
    await flushed.flush()
    with pytest.raises(Exception) as info:
        await flushed.abort()
    assert _attrs(info.value)[0] == "failed"
    await flushed.release()


# @spec ING-CORE-022
async def test_a_release_after_a_cancelled_abort_keeps_a_tracked_task(seam):
    """``SESS-ABORT-002``: tracked until the generation task ends."""
    lease = await _open(seam.factory)
    engine_abort = await _cancel_abort_in_handler(seam, lease)

    await lease.release()

    assert seam.owns(lease), "no task tracks the running generation task"
    engine_abort.release()
    assert await asyncio.wait_for(seam.factory.drain(math.inf), LIMIT) == []
    assert not seam.owns(lease)
    assert seam.generation(lease).done()


# @spec ING-CORE-022
async def test_a_cancelled_release_leaves_the_session_to_the_factory(seam):
    """``SESS-REL-003``, ``SESS-OWN-003``."""
    lease = await _open(seam.factory)
    await lease.abort()
    cleanup = seam.slow_cleanup()

    await _cancel_release(seam, lease)

    assert seam.handed_over(lease)
    assert seam.owns(lease)
    cleanup.release()
    assert await asyncio.wait_for(seam.factory.drain(math.inf), LIMIT) == []
    assert not seam.owns(lease)


# @spec ING-CORE-022
async def test_teardown_all_closes_and_returns_every_task_it_owns(seam):
    """``SESS-OWN-005`` and ``SESS-OWN-006``."""
    factory = seam.factory
    # A session whose ``abort`` was cancelled: its generation task runs.
    aborted = await _open(factory)
    await _cancel_abort_in_handler(seam, aborted)
    # A session whose ``release`` was cancelled: its cleanup task runs.
    handed = await _open(factory)
    await handed.abort()
    seam.slow_cleanup()
    await _cancel_release(seam, handed)
    # A session still open.
    live = await _open(factory)
    running = seam.running_tasks()
    assert seam.generation(aborted) in running

    tasks = factory.teardown_all()

    assert isinstance(tasks, set)
    assert running <= tasks, "teardown_all left a task out"
    await asyncio.wait_for(asyncio.wait(tasks), LIMIT)
    assert seam.running_tasks() == set()
    with pytest.raises(Exception) as refused:
        await factory.open(cadence=_cadence(), locale=_locale(factory))
    assert _attrs(refused.value)[0] == "failed"
    samples = np.zeros(1600, dtype=np.float32)
    with pytest.raises(Exception) as fed:
        await live.feed(samples, on_accepted=lambda n: None)
    assert _attrs(fed.value)[0] == "failed"
    with pytest.raises(Exception) as aborting:
        await live.abort()
    assert _attrs(aborting.value)[0] == "failed"
    live.teardown()  # the one lease call a closed factory accepts


# @spec ING-CORE-022
async def test_error_attributes_are_read_structurally_including_unknown(seam):
    from vllm_riva_frontend.errors import project_seam_error

    factory = seam.factory
    with pytest.raises(Exception) as refused:
        await factory.open(cadence=_cadence(), locale="no-such-locale")
    assert _attrs(refused.value) == ("refused", "locale")

    class Unknown(Exception):
        session_error = "failed"
        reason = "a_reason_this_frontend_does_not_know"

    assert _attrs(Unknown()) == (
        "failed",
        "a_reason_this_frontend_does_not_know",
    )
    assert project_seam_error(Unknown()).code == "internal"
    assert _attrs(ValueError("bare")) == (None, None)
    assert project_seam_error(ValueError("bare")).code == "internal"


# --------------------------------------------------- factory obligations


# @spec ING-CORE-023
async def test_drain_inf_awaits_cleanup_and_holding_tasks_cancelling_none(
    seam,
):
    factory = seam.factory
    retirement = seam.slow_cleanup()
    handed = await _hand_over(seam)
    held = await _open(factory)
    engine_abort = await _cancel_abort_in_handler(seam, held)
    await held.release()
    assert seam.owns(handed) and seam.owns(held)

    drain = asyncio.create_task(factory.drain(math.inf))
    await _elapse(seam, 0.2)
    assert not drain.done(), "drain(math.inf) returned with tasks running"
    retirement.release()
    await settle()
    assert not drain.done(), "drain(math.inf) left the holding task"
    engine_abort.release()

    assert await asyncio.wait_for(drain, LIMIT) == []
    assert not seam.owns(handed) and not seam.owns(held)
    assert seam.foreign_cancellations() == []
    assert seam.late_cancellations() == []


# @spec ING-CORE-023
async def test_finite_drain_returns_at_deadline_cancelling_none(seam):
    retirement = seam.slow_cleanup()
    lease = await _hand_over(seam)
    assert seam.owns(lease)

    budget = 0.2
    deadline = seam.clock.monotonic() + budget
    drain = asyncio.create_task(seam.factory.drain(deadline))
    await settle()
    assert not drain.done(), "drain gave up before its deadline"
    await _elapse(seam, budget + 0.05)
    faults = await asyncio.wait_for(drain, LIMIT)
    returned = seam.clock.monotonic()

    assert returned >= deadline - EARLY, "drain gave up before its deadline"
    assert returned <= deadline + LATE, "drain overran its absolute deadline"
    assert faults == []
    assert seam.owns(lease), "the deadline cancelled the cleanup"
    # The current time as the deadline returns at once.
    now = asyncio.create_task(seam.factory.drain(seam.clock.monotonic()))
    assert await asyncio.wait_for(now, LIMIT) == []
    assert seam.owns(lease)
    assert seam.foreign_cancellations() == []

    retirement.release()
    assert await asyncio.wait_for(seam.factory.drain(math.inf), LIMIT) == []
    assert not seam.owns(lease)


# @spec ING-CORE-024
@pytest.mark.parametrize("kind", ["cleanup", "holding"])
async def test_cleanup_and_holding_tasks_end_after_generation_task(seam, kind):
    lease = await _open(seam.factory)
    engine_abort = await _cancel_abort_in_handler(seam, lease)
    if kind == "cleanup":
        await _cancel_release(seam, lease)
    else:
        await lease.release()
    generation = seam.generation(lease)
    await settle()
    assert seam.owns(lease), f"the {kind} task ended before generation"

    engine_abort.release()
    for _ in range(1_000):
        if not seam.owns(lease):
            break
        await asyncio.sleep(0)
    # At the first turn the session is no longer held, generation is over.
    assert not seam.owns(lease)
    assert generation.done(), f"the {kind} task ended first"
    assert seam.late_cancellations() == []
    assert seam.foreign_cancellations() == []
    assert await seam.factory.drain(seam.clock.monotonic()) == []


# @spec ING-CORE-024
async def test_teardown_all_returns_the_cleanup_and_holding_tasks_it_cancels(
    seam,
):
    handed = await _open(seam.factory)
    await _cancel_abort_in_handler(seam, handed)
    await _cancel_release(seam, handed)
    held = await _open(seam.factory)
    await _cancel_abort_in_handler(seam, held)
    await held.release()
    running = seam.running_tasks()
    assert seam.owns(handed) and seam.owns(held)

    returned = seam.factory.teardown_all()

    assert running <= returned
    await asyncio.wait_for(asyncio.wait(returned), LIMIT)
    assert not seam.owns(handed) and not seam.owns(held)
    # The last-resort cancellations are the vehicle's to report, not faults.
    assert await seam.factory.drain(seam.clock.monotonic()) == []


# @spec ING-CORE-025
async def test_retained_fault_kinds_and_exclusions(seam):
    factory = seam.factory
    # Excluded: a refusal ``open`` raises (an ordinary capacity refusal).
    seam.refuse_next_open("capacity")
    with pytest.raises(Exception) as refused:
        await factory.open(cadence=_cadence(), locale=_locale(factory))
    assert _attrs(refused.value) == ("refused", "capacity")
    # Excluded: a generation failure the owner sees as a lease reason.
    surfaced = await _open(factory)
    seam.fail_generation(surfaced, failed("engine_dead"))
    await settle()
    samples = np.zeros(1600, dtype=np.float32)
    with pytest.raises(Exception) as fed:
        await surfaced.feed(samples, on_accepted=lambda n: None)
    assert _attrs(fed.value) == ("failed", "engine_dead")
    await surfaced.abort()
    await surfaced.release()
    assert await factory.drain(seam.clock.monotonic()) == []

    # Retained: a cleanup task's failed retirement wait.
    dead = RuntimeError("retirement wait: engine dead")
    seam.slow_cleanup().release(Raise(dead))
    await _hand_over(seam)
    # Retained: a generation exception after the session was handed over.
    broken = await _open(factory)
    await _break(seam, broken)
    await _cancel_release(seam, broken)
    late = RuntimeError("generation failed after the hand-over")
    seam.fail_generation(broken, late)
    await settle()

    faults = await asyncio.wait_for(factory.drain(math.inf), LIMIT)
    assert {id(f) for f in faults} == {id(dead), id(late)}
    assert len(faults) == 2
    assert not any(isinstance(f, asyncio.CancelledError) for f in faults)

    # Excluded: a task ``teardown_all`` cancels as a last resort.
    seam.slow_cleanup()
    await _hand_over(seam)
    returned = factory.teardown_all()
    await asyncio.wait_for(asyncio.wait(returned), LIMIT)
    assert await factory.drain(seam.clock.monotonic()) == []


# @spec ING-CORE-026
async def test_each_fault_returned_once_across_callers(seam):
    factory = seam.factory
    first_fault = RuntimeError("first")
    second_fault = RuntimeError("second")
    seam.inject_fault(first_fault)
    seam.inject_fault(second_fault)
    retirement = seam.slow_cleanup()
    await _hand_over(seam)

    callers = [asyncio.create_task(factory.drain(math.inf)) for _ in range(2)]
    await settle()
    retirement.release()
    results = await asyncio.wait_for(asyncio.gather(*callers), LIMIT)

    returned = [id(fault) for result in results for fault in result]
    assert sorted(returned) == sorted([id(first_fault), id(second_fault)])
    assert len(set(returned)) == len(returned), "a fault reached two callers"
    assert await factory.drain(seam.clock.monotonic()) == []


# @spec ING-CORE-027
async def test_no_cancellation_reaches_a_generation_task_in_its_handler(seam):
    factory = seam.factory
    # A cancelled ``abort`` and then ``release``: a holding task.
    held = await _open(factory)
    held_abort = await _cancel_abort_in_handler(seam, held)
    await held.release()
    # A cancelled ``abort`` and then a cancelled ``release``: a cleanup task.
    handed = await _open(factory)
    handed_abort = await _cancel_abort_in_handler(seam, handed)
    await _cancel_release(seam, handed)
    # A cancelled ``open`` after its generation task started.
    block = seam.slow_open()
    opening = asyncio.create_task(
        factory.open(cadence=_cadence(), locale=_locale(factory))
    )
    await asyncio.wait_for(block.entered.wait(), LIMIT)
    await settle()
    await _cancelled(opening)
    await settle()

    held_abort.release()
    handed_abort.release()
    assert await asyncio.wait_for(factory.drain(math.inf), LIMIT) == []

    assert seam.late_cancellations() == []
    assert seam.foreign_cancellations() == []


# @spec ING-CORE-028
@pytest.mark.parametrize("at", ["first_step", "retirement_wait"])
async def test_release_cancelled_at_any_suspension_hands_over(seam, at):
    lease = await _open(seam.factory)
    await lease.abort()
    block = seam.slow(lease, "release")
    release = asyncio.create_task(lease.release())
    if at == "first_step":
        # The call runs to its first suspension and no further.
        await asyncio.sleep(0)
    else:
        await asyncio.wait_for(block.entered.wait(), LIMIT)
        await settle()
    assert not release.done()

    await _cancelled(release)

    assert seam.handed_over(lease)
    assert await asyncio.wait_for(seam.factory.drain(math.inf), LIMIT) == []
    assert not seam.owns(lease)


# @spec ING-CORE-029
async def test_closed_factory_starts_no_task(seam):
    factory = seam.factory
    lease = await _open(factory)
    await lease.abort()
    block = seam.slow(lease, "release")
    release = asyncio.create_task(lease.release())
    await asyncio.wait_for(block.entered.wait(), LIMIT)

    returned = factory.teardown_all()
    if returned:
        await asyncio.wait_for(asyncio.wait(returned), LIMIT)
    # A hand-over after ``teardown_all`` starts no cleanup task.
    release.cancel()
    with contextlib.suppress(BaseException):
        await asyncio.wait_for(release, LIMIT)
    await settle()
    assert seam.running_tasks() == set()
    assert not seam.owns(lease)
    # Nor does an ``open`` after it.
    with pytest.raises(Exception) as info:
        await factory.open(cadence=_cadence(), locale=_locale(factory))
    assert _attrs(info.value)[0] == "failed"
    await settle()
    assert seam.running_tasks() == set()


# @spec ING-CORE-034
async def test_repeated_settlement_returns_at_once(seam):
    factory = seam.factory
    seam.slow_cleanup()  # never released: cancelled as a last resort
    await _hand_over(seam)
    returned = factory.teardown_all()
    await asyncio.wait_for(asyncio.wait(returned), LIMIT)

    # Whatever the deadline, every later ``drain`` returns at once.
    assert await asyncio.wait_for(factory.drain(math.inf), 1.0) == []
    far = seam.clock.monotonic() + 3600
    assert await asyncio.wait_for(factory.drain(far), 1.0) == []
    again = factory.teardown_all()  # raises nothing
    assert all(task.done() for task in again)
    # Only the retained faults not yet returned, once.
    fault = RuntimeError("not yet returned")
    seam.inject_fault(fault)
    assert await asyncio.wait_for(factory.drain(math.inf), 1.0) == [fault]
    assert await asyncio.wait_for(factory.drain(math.inf), 1.0) == []


# ------------------------------------------------- the ING-CORE-030 tests


# @spec ING-CORE-030
async def test_an_ordinary_refusal_and_generation_failure_leave_no_fault(
    seam,
):
    factory = seam.factory
    seam.refuse_next_open("capacity")
    with pytest.raises(Exception) as refused:
        await factory.open(cadence=_cadence(), locale=_locale(factory))
    assert _attrs(refused.value) == ("refused", "capacity")
    assert seam.running_tasks() == set(), "a refusal started a task"
    lease = await _open(factory)
    seam.fail_generation(lease, failed("engine_dead"))
    await settle()
    samples = np.zeros(1600, dtype=np.float32)
    with pytest.raises(Exception) as fed:
        await lease.feed(samples, on_accepted=lambda n: None)
    assert _attrs(fed.value) == ("failed", "engine_dead")
    await lease.abort()
    await lease.release()

    assert await asyncio.wait_for(factory.drain(math.inf), LIMIT) == []


# @spec ING-CORE-030
async def test_a_handler_raising_after_the_hand_over_yields_one_fault(seam):
    lease = await _open(seam.factory)
    raised = RuntimeError("engine abort failed after the hand-over")
    handler = seam.fail_engine_abort(lease, raised)
    abort = asyncio.create_task(lease.abort())
    await asyncio.wait_for(handler.entered.wait(), LIMIT)
    await _cancelled(abort)
    await _cancel_release(seam, lease)

    handler.release()
    faults = await asyncio.wait_for(seam.factory.drain(math.inf), LIMIT)

    assert faults == [raised]
    assert not seam.owns(lease), "the cleanup task has not dropped it"
    assert await seam.factory.drain(math.inf) == []
    assert await seam.factory.drain(seam.clock.monotonic()) == []


# @spec ING-CORE-030
async def test_a_handler_raising_under_an_uncancelled_abort_yields_one_fault(
    seam,
):
    lease = await _open(seam.factory)
    raised = RuntimeError("engine abort failed under abort")
    handler = seam.fail_engine_abort(lease, raised)
    abort = asyncio.create_task(lease.abort())
    await asyncio.wait_for(handler.entered.wait(), LIMIT)

    handler.release()
    assert await asyncio.wait_for(abort, LIMIT) is None  # returns normally

    await lease.release()
    assert await seam.factory.drain(seam.clock.monotonic()) == [raised]
    assert await seam.factory.drain(seam.clock.monotonic()) == []
    assert await seam.factory.drain(math.inf) == []


# @spec ING-CORE-030
async def test_a_cancelled_open_after_its_generation_started_is_covered(seam):
    factory = seam.factory
    engine_abort = seam.slow_abandoned_engine_abort()
    block = seam.slow_open()
    opening = asyncio.create_task(
        factory.open(cadence=_cadence(), locale=_locale(factory))
    )
    await asyncio.wait_for(block.entered.wait(), LIMIT)
    await settle()  # the generation task has taken its first step

    await _cancelled(opening)
    session = seam.abandoned()
    assert session is not None, "the cancelled open gave up no session"
    generation = seam.generation(session)
    assert generation is not None
    try:
        await settle()
        # The intermediate state: the generation task, cancelled, awaits the
        # engine's abort; a holding or cleanup task covers the session, and
        # ``drain(math.inf)`` waits on that covering task.
        assert seam.engine_calls(session) == ["generate", "abort"]
        assert not generation.done(), "the engine abort is still pending"
        assert seam.owns(session), "no holding or cleanup task covers it"
        drain = asyncio.create_task(factory.drain(math.inf))
        await settle()
        assert not drain.done(), "drain(math.inf) did not wait on the cover"
        assert seam.late_cancellations() == []
    finally:
        engine_abort.release()
    assert await asyncio.wait_for(drain, LIMIT) == []
    assert generation.done() and not seam.owns(session)

    faults, still_running = await _factory_settlement(seam)
    assert faults == [] and still_running == []
    assert seam.running_tasks() == set()
    assert seam.late_cancellations() == []
    assert seam.foreign_cancellations() == []


# @spec ING-CORE-030
async def test_a_cancelled_open_before_the_first_step_never_runs(seam):
    factory = seam.factory
    opener: dict[str, asyncio.Task[Any]] = {}

    def cancel_opener() -> None:
        opener["task"].cancel()

    seam.slow_open(before_generation=True, on_enter=cancel_opener)
    opener["task"] = asyncio.create_task(
        factory.open(cadence=_cadence(), locale=_locale(factory))
    )
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(opener["task"], LIMIT)
    session = seam.abandoned()
    assert session is not None, "the cancelled open gave up no session"
    generation = seam.generation(session)

    await asyncio.sleep(0)  # one turn of the event loop

    assert generation is None or generation.done(), "it outlived one turn"
    assert seam.engine_calls(session) == [], "the generation task ran"
    assert not seam.owns(session)
    assert seam.running_tasks() == set()
    assert seam.late_cancellations() == []
    assert seam.foreign_cancellations() == []
    assert await factory.drain(seam.clock.monotonic()) == []


# @spec ING-CORE-030, ING-LIFE-060
async def test_the_release_hand_over_runs_through_the_owners_cancellation(
    seam,
):
    """The owner's own mechanism, not a fake hook, cancels ``release``."""
    from vllm_riva_frontend.admission import OwnerKind
    from vllm_riva_frontend.config import parse_config
    from vllm_riva_frontend.errors import catalog
    from vllm_riva_frontend.lease import Cause, StreamOwner
    from vllm_riva_frontend.lifecycle import OwnerRegistry

    budget = 0.5
    opened: list[Any] = []

    async def opener(**kwargs: Any) -> Any:
        lease = await seam.factory.open(**kwargs)
        opened.append(lease)
        return lease

    registry = OwnerRegistry()
    record = registry.register(OwnerKind.stream, asyncio.current_task())
    owner = StreamOwner(
        lease_opener=opener,
        surface="riva",
        config=parse_config(
            {"riva_frontend": {"session_cleanup_timeout": budget}}
        ),
        clock=seam.clock,
        registry=registry,
        record=record,
        catalog=catalog(),
    )
    await owner.open(_cadence(), _locale(seam.factory))
    (lease,) = opened
    await settle()
    release = seam.slow(lease, "release")

    failing = asyncio.create_task(
        owner.fail(Cause(code="internal", phase="transport"))
    )
    await asyncio.wait_for(release.entered.wait(), LIMIT)
    assert not failing.done()
    await _elapse(seam, budget + 0.1)
    await asyncio.wait_for(failing, LIMIT)

    # The owner cancelled its ``release`` task after its first step and
    # awaited it: the session is with the factory's cleanup.
    assert seam.handed_over(lease)
    assert not release.released.is_set()
    assert len(registry.records) == 0
    faults, still_running = await _factory_settlement(seam)
    assert faults == [] and still_running == []
    assert seam.late_cancellations() == []
    assert seam.foreign_cancellations() == []
