"""Direct RFC-1 SessionFactory/SessionLease tests-first contract."""

import asyncio
import inspect

import pytest

from vllm_riva_frontend.lease import (
    DirectLeaseOwner,
    SessionFactory,
    SessionLease,
)


class RecordingLease:
    """Small async fake used to describe the Phase-6 owner call order."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def feed(self, samples: object, *, on_accepted: object) -> list[str]:
        del samples, on_accepted
        self.calls.append("feed")
        return []

    async def update_locale(self, locale: str) -> object:
        self.calls.append(f"locale:{locale}")
        return None

    async def flush(self) -> str:
        self.calls.append("flush")
        return "tail"

    async def finish(self) -> None:
        self.calls.append("finish")

    async def abort(self) -> None:
        self.calls.append("abort")

    async def release(self) -> None:
        self.calls.append("release")


class RecordingFactory:
    def __init__(self, lease: RecordingLease) -> None:
        self.lease = lease
        self.opened: list[tuple[str, str]] = []

    async def open(self, *, cadence: str, locale: str) -> RecordingLease:
        self.opened.append((cadence, locale))
        return self.lease


class FailingLease(RecordingLease):
    """Lease fake that fails normal terminal work but records cleanup order."""

    async def finish(self) -> None:
        self.calls.append("finish")
        raise RuntimeError("engine failure")


class ReleaseFailLease(RecordingLease):
    async def release(self) -> None:
        self.calls.append("release")
        raise RuntimeError("release failure")


class AbortFailLease(RecordingLease):
    async def abort(self) -> None:
        self.calls.append("abort")
        raise RuntimeError("abort failure")


class AbortAndReleaseFailLease(AbortFailLease):
    async def release(self) -> None:
        self.calls.append("release")
        raise RuntimeError("release failure")


class SettlingFactory:
    """Factory whose open settles only after owner cancellation."""

    def __init__(self, lease: RecordingLease) -> None:
        self.lease = lease
        self.started = asyncio.Event()
        self.settle = asyncio.Event()

    async def open(self, *, cadence: str, locale: str) -> RecordingLease:
        del cadence, locale
        self.started.set()
        await self.settle.wait()
        return self.lease


# @spec ING-CORE-001, ING-CORE-002, ING-VEH-004
def test_direct_boundary_exposes_only_the_factory_and_lease_operations() -> (
    None
):
    assert set(SessionFactory.__dict__) >= {"open"}
    assert {
        "feed",
        "update_locale",
        "flush",
        "finish",
        "abort",
        "release",
    } <= set(SessionLease.__dict__)


# @spec ING-FE-005, ING-VEH-004
def test_feed_uses_the_exact_synchronous_on_accepted_callback_contract() -> (
    None
):
    parameter = inspect.signature(SessionLease.feed).parameters["on_accepted"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


# @spec ING-LIFE-010, ING-LIFE-012
def test_normal_owner_flushes_finishes_then_releases() -> None:
    lease = RecordingLease()
    factory = RecordingFactory(lease)
    owner = DirectLeaseOwner(factory, cleanup_timeout=1.0)

    async def exercise() -> str | None:
        await owner.open(cadence="1120ms", locale="en-US")
        return await owner.complete()

    result = asyncio.run(exercise())
    assert result == "tail"
    assert factory.opened == [("1120ms", "en-US")]
    assert lease.calls == ["flush", "finish", "release"]


class StructuredTerminalLease(RecordingLease):
    """Lease shaped like the 0.25 bounded-stream host line.

    ``flush`` returns a structured terminal result carrying
    ``complete_text`` plus a completion acknowledgement this frontend
    does not consume.
    """

    class _Terminal:
        complete_text = " final tail"
        completion = object()
        emit_segment_event = False

    async def flush(self) -> object:
        self.calls.append("flush")
        return self._Terminal()


class MalformedTerminalLease(RecordingLease):
    """Lease violating the host contract with an unrecognizable shape."""

    async def flush(self) -> object:
        self.calls.append("flush")
        return 41


# @spec ING-LIFE-010
def test_complete_normalizes_the_structured_terminal_of_the_v025_host() -> (
    None
):
    lease = StructuredTerminalLease()
    owner = DirectLeaseOwner(RecordingFactory(lease), cleanup_timeout=1.0)

    async def exercise() -> str | None:
        await owner.open(cadence="1120ms", locale="en-US")
        return await owner.complete()

    result = asyncio.run(exercise())
    assert result == " final tail"
    assert lease.calls == ["flush", "finish", "release"]


# @spec ING-LIFE-010
def test_complete_rejects_a_malformed_terminal_shape_with_abort_cleanup() -> (
    None
):
    lease = MalformedTerminalLease()
    owner = DirectLeaseOwner(RecordingFactory(lease), cleanup_timeout=1.0)

    async def exercise() -> None:
        await owner.open(cadence="1120ms", locale="en-US")
        with pytest.raises(TypeError, match="neither a transcript string"):
            await owner.complete()
        # The shape violation selected the abnormal terminal; a retry
        # must not re-enter flush/finish on the same lease.
        assert await owner.complete() is None

    asyncio.run(exercise())
    assert lease.calls == ["flush", "abort", "release"]


# @spec ING-LIFE-004, ING-LIFE-010, ING-LIFE-013
def test_abnormal_owner_aborts_then_releases_without_finish() -> None:
    lease = RecordingLease()
    factory = RecordingFactory(lease)
    owner = DirectLeaseOwner(factory, cleanup_timeout=1.0)

    async def exercise() -> None:
        await owner.open(cadence="560ms", locale="auto")
        await owner.cancel()

    asyncio.run(exercise())
    assert factory.opened == [("560ms", "auto")]
    assert lease.calls == ["abort", "release"]


# @spec ING-VEH-014
def test_owner_has_no_provider_mode_or_batcher_constructor_argument() -> None:
    assert tuple(inspect.signature(DirectLeaseOwner).parameters) == (
        "factory",
        "cleanup_timeout",
        "fault_reporter",
    )


# @spec ING-CORE-008, ING-FE-006, ING-VEH-007, ING-VEH-008
def test_configured_lease_has_no_admission_or_chunk_geometry() -> None:
    signature = inspect.signature(SessionFactory.open)
    assert set(signature.parameters) == {"self", "cadence", "locale"}
    assert "chunk_samples" not in str(signature)
    assert "admitted" not in str(signature)


# @spec ING-LIFE-010, ING-LIFE-013, ING-LIFE-014
def test_normal_failure_aborts_releases_and_keeps_primary_error() -> None:
    lease = FailingLease()
    factory = RecordingFactory(lease)
    owner = DirectLeaseOwner(factory, cleanup_timeout=1.0)

    async def exercise() -> None:
        await owner.open(cadence="1120ms", locale="auto")
        await owner.complete()

    with pytest.raises(RuntimeError, match="engine failure"):
        asyncio.run(exercise())
    assert lease.calls == ["flush", "finish", "abort", "release"]


# @spec ING-LIFE-010, ING-LIFE-013, ING-LIFE-014
def test_failed_normal_terminal_cannot_retry_into_a_second_terminal_path() -> (
    None
):
    lease = FailingLease()
    owner = DirectLeaseOwner(RecordingFactory(lease), cleanup_timeout=1.0)

    async def exercise() -> None:
        await owner.open(cadence="1120ms", locale="auto")
        with pytest.raises(RuntimeError, match="engine failure"):
            await owner.complete()
        assert await owner.complete() is None
        await owner.cancel()

    asyncio.run(exercise())
    assert lease.calls == ["flush", "finish", "abort", "release"]


# @spec ING-LIFE-012, ING-LIFE-013
def test_open_feed_and_duplicate_terminal_are_serialized_and_idempotent() -> (
    None
):
    lease = RecordingLease()
    owner = DirectLeaseOwner(RecordingFactory(lease), cleanup_timeout=1.0)
    accepted: list[int] = []

    async def exercise() -> None:
        await owner.open(cadence="1120ms", locale="auto")
        assert await owner.feed([0.0], on_accepted=accepted.append) == []
        assert await owner.complete() == "tail"
        assert await owner.complete() == "tail"
        await owner.cancel()

    asyncio.run(exercise())
    assert lease.calls == ["feed", "flush", "finish", "release"]


# @spec ING-LIFE-010, ING-LIFE-014
def test_release_failure_after_finish_never_allows_an_illegal_abort() -> None:
    lease = ReleaseFailLease()
    owner = DirectLeaseOwner(RecordingFactory(lease), cleanup_timeout=1.0)

    async def exercise() -> None:
        await owner.open(cadence="1120ms", locale="auto")
        with pytest.raises(RuntimeError, match="release failure"):
            await owner.complete()
        await owner.cancel()
        with pytest.raises(RuntimeError, match="release failure"):
            await owner.complete()

    asyncio.run(exercise())
    assert lease.calls == ["flush", "finish", "release"]


# @spec ING-LIFE-014, ING-VEH-017
def test_cleanup_failure_reports_unhealthy_once() -> None:
    lease = ReleaseFailLease()
    faults: list[BaseException] = []
    owner = DirectLeaseOwner(
        RecordingFactory(lease),
        cleanup_timeout=1.0,
        fault_reporter=faults.append,
    )

    async def exercise() -> None:
        await owner.open(cadence="1120ms", locale="auto")
        with pytest.raises(RuntimeError, match="release failure"):
            await owner.complete()

    asyncio.run(exercise())
    assert [str(fault) for fault in faults] == ["release failure"]
    assert lease.calls == ["flush", "finish", "release"]


# @spec ING-LIFE-012, ING-LIFE-013
def test_cancelled_open_settles_then_aborts_and_releases_returned_lease() -> (
    None
):
    lease = RecordingLease()
    factory = SettlingFactory(lease)
    owner = DirectLeaseOwner(factory, cleanup_timeout=1.0)

    async def exercise() -> None:
        opening = asyncio.create_task(
            owner.open(cadence="1120ms", locale="auto")
        )
        await factory.started.wait()
        opening.cancel()
        await asyncio.sleep(0)
        factory.settle.set()
        with pytest.raises(asyncio.CancelledError):
            await opening

    asyncio.run(exercise())
    assert lease.calls == ["abort", "release"]


# @spec ING-LIFE-012, ING-LIFE-014, ING-VEH-017
def test_cancelled_open_timeout_reports_fault_before_late_lease_arrives() -> (
    None
):
    lease = RecordingLease()
    factory = SettlingFactory(lease)
    faults: list[BaseException] = []
    owner = DirectLeaseOwner(
        factory,
        cleanup_timeout=0.01,
        fault_reporter=faults.append,
    )

    async def exercise() -> None:
        opening = asyncio.create_task(
            owner.open(cadence="1120ms", locale="auto")
        )
        await factory.started.wait()
        opening.cancel()
        with pytest.raises(asyncio.TimeoutError):
            await opening
        assert len(faults) == 1
        assert isinstance(faults[0], asyncio.TimeoutError)
        factory.settle.set()
        await asyncio.sleep(0.02)

    asyncio.run(exercise())
    assert len(faults) == 1
    assert lease.calls == ["abort", "release"]


# @spec ING-LIFE-012, ING-LIFE-013
def test_owner_rejects_invalid_order_duplicate_open_and_terminal_work() -> None:
    with pytest.raises(ValueError, match="cleanup_timeout"):
        DirectLeaseOwner(RecordingFactory(RecordingLease()), cleanup_timeout=0)

    lease = RecordingLease()
    owner = DirectLeaseOwner(RecordingFactory(lease), cleanup_timeout=1.0)

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="open must succeed"):
            await owner.feed([], on_accepted=lambda _: None)
        with pytest.raises(RuntimeError, match="open must succeed"):
            await owner.update_locale("auto")
        await owner.open(cadence="1120ms", locale="auto")
        with pytest.raises(RuntimeError, match="exactly one"):
            await owner.open(cadence="1120ms", locale="auto")
        await owner.update_locale("en-US")
        await owner.cancel()
        with pytest.raises(RuntimeError, match="session terminal"):
            await owner.feed([], on_accepted=lambda _: None)
        with pytest.raises(RuntimeError, match="session terminal"):
            await owner.update_locale("auto")
        await owner.cancel()
        assert await owner.complete() is None

    asyncio.run(exercise())
    assert lease.calls == ["locale:en-US", "abort", "release"]


# @spec ING-LIFE-004, ING-LIFE-013, ING-LIFE-014
@pytest.mark.parametrize(
    ("lease_type", "message"),
    [
        (AbortFailLease, "abort failure"),
        (AbortAndReleaseFailLease, "abort failure"),
    ],
)
def test_abnormal_cleanup_retains_abort_as_the_primary_failure(
    lease_type: type[RecordingLease], message: str
) -> None:
    lease = lease_type()
    owner = DirectLeaseOwner(RecordingFactory(lease), cleanup_timeout=1.0)

    async def exercise() -> None:
        await owner.open(cadence="1120ms", locale="auto")
        await owner.cancel()

    with pytest.raises(RuntimeError, match=message):
        asyncio.run(exercise())
    assert lease.calls == ["abort", "release"]


# @spec ING-LIFE-012, ING-LIFE-013
def test_factory_open_failure_marks_owner_terminal() -> None:
    class FailingFactory:
        async def open(self, *, cadence: str, locale: str) -> RecordingLease:
            del cadence, locale
            raise RuntimeError("open failure")

    owner = DirectLeaseOwner(FailingFactory(), cleanup_timeout=1.0)

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="open failure"):
            await owner.open(cadence="1120ms", locale="auto")
        assert await owner.complete() is None
        await owner.cancel()

    asyncio.run(exercise())
