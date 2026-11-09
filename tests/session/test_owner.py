"""The stream owner, driven directly: states, finalization and cleanup order.

``StreamOwner`` is built from the layout's constructor arguments over the
shared ``FakeSessionFactory``; the fake lease (``strict=True``) raises on
overlapping calls and on calls its state does not allow, so a wrong order
shows up as a failure of the call itself.
"""

from __future__ import annotations

import asyncio
import contextlib

import numpy as np
import pytest
from support.clock import settle
from support.fake_session import Block, Raise, Return, failed

SAMPLES = np.zeros(1600, dtype=np.float32)


async def _open(owner, locale: str = "en-US"):
    return await owner.open(cadence="560ms", locale=locale)


def _lease(fake_factory):
    assert len(fake_factory.leases) == 1
    return fake_factory.leases[0]


# @spec ING-CORE-011, ING-LIFE-058
async def test_owner_registers_then_enters_configured_when_open_returns(
    make_owner, owner_parts, fake_factory
):
    owner = make_owner()
    assert owner.state == "awaiting_configuration"
    assert len(owner_parts["registry"].records) == 1

    await _open(owner)

    assert owner.state == "configured"
    assert owner.lease_state == "open"
    assert owner.calls == []
    assert fake_factory.names("lease:0") == []


# @spec ING-LIFE-017, ING-LIFE-025, ING-LIFE-058
async def test_finalize_flushes_finishes_releases_and_returns_flush_text(
    make_owner, owner_parts, fake_factory
):
    owner = make_owner()
    await _open(owner)
    lease = _lease(fake_factory)
    lease.script("feed", Return(["partial"]))
    await owner.feed(SAMPLES)
    lease.script("flush", Return("the final transcript"))

    text = await owner.finalize()

    assert text == "the final transcript"
    assert lease.names() == ["feed", "flush", "finish", "release"]
    assert owner.calls == lease.names()
    assert lease.names().count("finish") == 1
    assert owner.lease_state == "released"
    assert owner.state == "terminal"
    assert owner.cause is None
    assert len(owner_parts["registry"].records) == 0


# @spec ING-LIFE-003, ING-LIFE-017
async def test_zero_audio_finalization_still_flushes(make_owner, fake_factory):
    owner = make_owner()
    await _open(owner)
    lease = _lease(fake_factory)
    lease.script("flush", Return(""))

    assert await owner.finalize() == ""
    names = [n for n in lease.names() if n != "feed"]
    assert names == ["flush", "finish", "release"]
    assert sum(lease.accepted) == 0


# @spec ING-LIFE-017, ING-LIFE-026
@pytest.mark.parametrize("returned", [None, b"bytes", ["list"], 7])
async def test_a_non_string_flush_result_is_a_failed_flush(
    make_owner, fake_factory, returned
):
    from vllm_riva_frontend.errors import FrontendError

    owner = make_owner()
    await _open(owner)
    lease = _lease(fake_factory)
    lease.script("flush", Return(returned))

    with pytest.raises(FrontendError) as info:
        await owner.finalize()

    # The call returned, so the lease is flushed: it accepts neither
    # ``finish`` (the flush failed) nor ``abort`` (ING-LIFE-027), and the
    # strict fake would raise on either. Cleanup is ``release`` only.
    assert info.value.code == "internal"
    assert lease.names() == ["flush", "release"]
    assert owner.cause is not None and owner.cause.code == "internal"
    assert owner.lease_state == "released"


# @spec ING-LIFE-026, ING-LIFE-056
async def test_a_failed_flush_takes_abort_then_release_and_never_finish(
    make_owner, fake_factory
):
    from vllm_riva_frontend.errors import FrontendError

    owner = make_owner()
    await _open(owner)
    lease = _lease(fake_factory)
    lease.script("flush", Raise(failed("abnormal_end")))

    with pytest.raises(FrontendError):
        await owner.finalize()

    assert lease.names() == ["flush", "abort", "release"]
    assert owner.cause.code == "internal"
    assert owner.lease_state == "released"


# @spec ING-LIFE-027, ING-LIFE-020
async def test_finish_failure_after_flush_goes_to_release_only(
    make_owner, fake_factory
):
    from vllm_riva_frontend.errors import FrontendError

    owner = make_owner()
    await _open(owner)
    lease = _lease(fake_factory)
    lease.script("finish", Raise(failed("abnormal_end")))

    with pytest.raises(FrontendError) as info:
        await owner.finalize()

    assert lease.names() == ["flush", "finish", "release"]
    assert "abort" not in lease.names()
    assert info.value.code == "internal"


# @spec ING-LIFE-027
async def test_cancelled_finish_after_flush_goes_to_release_only(
    make_owner, fake_factory
):
    owner = make_owner()
    await _open(owner)
    lease = _lease(fake_factory)
    block = Block()
    lease.script("finish", block)

    task = asyncio.create_task(owner.finalize())
    await block.entered.wait()
    task.cancel()
    with contextlib.suppress(BaseException):
        await task

    assert lease.names() == ["flush", "finish", "release"]
    assert "abort" not in lease.names()


# @spec ING-LIFE-020
async def test_release_failure_on_the_normal_path_is_no_final(
    make_owner, fake_factory
):
    from vllm_riva_frontend.errors import FrontendError

    owner = make_owner()
    await _open(owner)
    lease = _lease(fake_factory)
    lease.script("release", Raise(failed("state")))

    with pytest.raises(FrontendError) as info:
        await owner.finalize()

    assert info.value.code == "internal"
    assert owner.lease_state != "released"


# @spec ING-LIFE-026, ING-LIFE-056
async def test_an_abnormal_cause_from_open_takes_abort_then_release(
    make_owner, fake_factory, cause
):
    owner = make_owner()
    await _open(owner)
    lease = _lease(fake_factory)
    await owner.feed(SAMPLES)

    await owner.fail(cause("idle_timeout", "idle"))
    await owner.fail(cause("internal", "engine"))

    assert lease.names() == ["feed", "abort", "release"]
    assert owner.cause.code == "idle_timeout"
    assert owner.state == "terminal"


# @spec ING-LIFE-026, ING-LIFE-027
async def test_an_abnormal_cause_after_flush_takes_release_only(
    make_owner, fake_factory, cause
):
    owner = make_owner()
    await _open(owner)
    lease = _lease(fake_factory)
    block = Block()
    lease.script("finish", block)
    task = asyncio.create_task(owner.finalize())
    await block.entered.wait()

    await owner.fail(cause("internal", "transport"))
    with contextlib.suppress(BaseException):
        await task

    assert "abort" not in lease.names()
    assert lease.names()[-1] == "release"


# @spec ING-LIFE-035, ING-LIFE-058
async def test_an_abnormal_cause_cancels_and_settles_the_call_before_cleanup(
    make_owner, fake_factory, cause
):
    owner = make_owner()
    await _open(owner)
    lease = _lease(fake_factory)
    block = Block()
    lease.script("feed", block)
    feed = asyncio.create_task(owner.feed(SAMPLES))
    await block.entered.wait()

    await owner.fail(cause("idle_timeout", "idle"))
    with contextlib.suppress(BaseException):
        await feed

    # The strict fake raises on overlap, so a released lease proves the
    # abort started only after the cancelled feed settled.
    assert lease.names() == ["feed", "abort", "release"]
    assert lease.state == "released"
    assert feed.done()


# @spec ING-LIFE-036
async def test_a_cause_while_open_is_pending_cancels_open_and_calls_nothing(
    make_owner, fake_factory, cause
):
    from vllm_riva_frontend.errors import FrontendError

    owner = make_owner()
    block = Block()
    fake_factory.script("open", block)
    opening = asyncio.create_task(_open(owner))
    await block.entered.wait()

    await owner.fail(cause("configuration_timeout", "preconfiguration"))
    with contextlib.suppress(BaseException):
        await opening
    block.release()
    await asyncio.sleep(0)

    # The interrupted ``open`` raises the selected cause in its caller.
    assert isinstance(opening.exception(), FrontendError)
    assert opening.exception().code == "configuration_timeout"
    assert [
        c for c in fake_factory.calls if c.target.startswith("lease:")
    ] == []
    assert fake_factory.leases == []
    assert owner.cause.code == "configuration_timeout"


# @spec ING-LIFE-028
async def test_a_failure_after_finish_settles_does_not_turn_into_abort(
    make_owner, fake_factory, cause
):
    from vllm_riva_frontend.errors import FrontendError

    owner = make_owner()
    await _open(owner)
    lease = _lease(fake_factory)
    block = Block()
    lease.script("release", block)
    finalizing = asyncio.create_task(owner.finalize())
    await block.entered.wait()

    # ``fail`` returns after cleanup, and the cleanup here is the running
    # ``release``: it neither cancels that call nor returns before it ends.
    failing = asyncio.create_task(owner.fail(cause("internal", "transport")))
    try:
        await settle()
        assert not failing.done(), "fail returned before release ended"
        assert not finalizing.done()
    finally:
        block.release()
    done, _ = await asyncio.wait({failing, finalizing}, timeout=2)
    assert done == {failing, finalizing}, "fail or finalize did not return"

    assert failing.exception() is None
    assert lease.names() == ["flush", "finish", "release"]
    assert lease.state == "released"
    assert fake_factory.handed_over == []
    assert owner.lease_state == "released"
    # The cause is recorded and suppresses the final: ``finalize`` raises it
    # instead of returning the transcript.
    assert owner.cause is not None and owner.cause.code == "internal"
    assert isinstance(finalizing.exception(), FrontendError)
    assert finalizing.exception().code == "internal"
