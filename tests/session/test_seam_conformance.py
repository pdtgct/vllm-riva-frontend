"""The seam conformance suite: what the stream owner relies on in a factory.

The suite runs here against ``FakeSessionFactory``. The model plugin's
repository runs the same functions against its real factory by overriding the
``seam`` fixture with an adapter that provides the factory, the clock its
``drain`` deadline is measured on (``time.monotonic()`` for the real factory)
and the hooks the suite needs (inject a cleanup fault, make ``abort``,
``release`` or the factory's cleanup of a handed-over session slow, tell
whether a session was handed to the factory's cleanup and whether that
cleanup still owns it).
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from support.clock import Clock, RealClock
from support.fake_session import Block, FakeLease, FakeSessionFactory

#: Real-time allowance around ``drain``'s deadline for loop scheduling.
EARLY = 0.05
LATE = 0.5


@dataclass
class FakeSeam:
    """The suite's hooks over ``FakeSessionFactory``."""

    factory: FakeSessionFactory
    #: The clock of ``factory.drain``'s absolute deadline.
    clock: Clock

    def inject_fault(self, fault: BaseException) -> None:
        self.factory.faults.append(fault)

    def slow(self, lease: FakeLease, name: str) -> Block:
        block = Block()
        lease.script(name, block)
        return block

    def slow_cleanup(self) -> Block:
        """Keep the next handed-over session's cleanup running."""
        block = Block()
        self.factory.script("cleanup", block)
        return block

    def handed_over(self, lease: FakeLease) -> bool:
        return lease in self.factory.handed_over

    def owns(self, lease: FakeLease) -> bool:
        return self.factory.owns(lease)


@pytest.fixture(params=["fake"])
def seam(request):
    clock = RealClock()
    return FakeSeam(FakeSessionFactory(final_text="final", clock=clock), clock)


def _cadence() -> str:
    from vllm_riva_frontend.lease import STREAM_CADENCE

    return STREAM_CADENCE


def _locale(factory: Any) -> str:
    return factory.default_locale or sorted(factory.locales)[0]


async def _open(factory: Any) -> Any:
    return await factory.open(cadence=_cadence(), locale=_locale(factory))


def _attrs(exc: BaseException) -> tuple[str | None, str | None]:
    from vllm_riva_frontend.errors import seam_error

    return seam_error(exc)


# @spec ING-CORE-016
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
        "teardown",
    ):
        assert callable(getattr(lease, member))
    if getattr(SessionFactory, "_is_runtime_protocol", False):
        assert isinstance(factory, SessionFactory)
    if getattr(SessionLease, "_is_runtime_protocol", False):
        assert isinstance(lease, SessionLease)
    await lease.abort()
    await lease.release()


# @spec ING-CORE-016
async def test_drain_takes_an_absolute_deadline_and_returns_faults_once(seam):
    factory = seam.factory
    fault = RuntimeError("cleanup failed")
    seam.inject_fault(fault)
    deadline = seam.clock.monotonic() + 5.0

    first = await factory.drain(deadline)
    second = await factory.drain(deadline)

    assert first == [fault]
    assert second == []


async def _hand_over(seam: Any) -> Any:
    """A lease whose ``release`` was cancelled after it began."""
    lease = await _open(seam.factory)
    await lease.abort()
    block = seam.slow(lease, "release")
    release = asyncio.create_task(lease.release())
    await block.entered.wait()
    release.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await release
    assert seam.handed_over(lease)
    return lease


# @spec ING-CORE-016
async def test_drain_returns_by_its_deadline_while_cleanup_still_runs(seam):
    cleanup = seam.slow_cleanup()
    lease = await _hand_over(seam)
    assert seam.owns(lease)

    budget = 0.2
    deadline = seam.clock.monotonic() + budget
    faults = await asyncio.wait_for(seam.factory.drain(deadline), budget + 5)
    returned = seam.clock.monotonic()

    assert returned >= deadline - EARLY, "drain gave up before its deadline"
    assert returned <= deadline + LATE, "drain overran its absolute deadline"
    assert faults == []
    assert seam.owns(lease), "the running cleanup stays with the factory"

    cleanup.release()
    await asyncio.wait_for(
        seam.factory.drain(seam.clock.monotonic() + 5.0), 10.0
    )
    assert not seam.owns(lease)


# @spec ING-CORE-016
async def test_teardown_all_is_terminal(seam):
    factory = seam.factory
    tasks = factory.teardown_all()
    assert isinstance(tasks, set)

    with pytest.raises(Exception) as info:
        await _open(factory)
    assert _attrs(info.value)[0] == "failed"


# @spec ING-CORE-016
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


# @spec ING-CORE-016
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


# @spec ING-CORE-016, ING-LIFE-037
async def test_release_is_accepted_after_a_cancelled_abort(seam):
    lease = await _open(seam.factory)
    block = seam.slow(lease, "abort")
    abort = asyncio.create_task(lease.abort())
    await block.entered.wait()
    abort.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await abort

    await lease.release()


# @spec ING-CORE-016, ING-LIFE-037
async def test_a_cancelled_release_leaves_the_session_to_the_factory(seam):
    lease = await _open(seam.factory)
    await lease.abort()
    block = seam.slow(lease, "release")
    release = asyncio.create_task(lease.release())
    await block.entered.wait()
    release.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await release

    assert seam.handed_over(lease)


# @spec ING-CORE-016
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
