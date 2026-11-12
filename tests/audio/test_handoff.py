"""What a stream hands the lease: whole-sample pieces, bounded, drained.

The Riva servicer is driven directly; ``FakeLease.fed`` records every piece
``feed`` received and ``FakeLease.accepted`` the counts it acknowledged
through ``on_accepted``. ``FakeLease.peak_outstanding`` is the most samples
the lease ever held handed over and unacknowledged: the pre-submit bound seen
from the lease's side. The frontend's own view of that credit is
``StreamOwner.outstanding_samples``, read on an owner built directly (the
shared ``make_owner`` fixture).
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import grpc
import numpy as np
import pytest
from scipy.signal import resample_poly
from support.audio import pcm16_bytes, pcm16_sine
from support.clock import settle, until
from support.fake_session import Block

from .conftest import open_stream

PACKAGE = Path(__file__).parents[2] / "src" / "vllm_riva_frontend"
LIMIT = 1_000


def _lease(fake_factory):
    assert fake_factory.leases, "open was not called"
    return fake_factory.leases[-1]


# @spec ING-FE-009
async def test_stream_hands_whole_sample_16k_float32_pieces(
    started, fake_factory
) -> None:
    samples = pcm16_sine(0.077)  # 1232 samples: no cadence multiple
    data = pcm16_bytes(samples)
    stream = await open_stream(started)
    stream.audio(data[:1001])  # a sample straddles the two messages
    stream.audio(data[1001:])
    await stream.finish()
    lease = _lease(fake_factory)
    for piece in lease.fed:
        piece = np.asarray(piece)
        assert piece.dtype == np.float32
        assert piece.ndim == 1
    np.testing.assert_array_equal(
        lease.fed_samples,
        samples.astype(np.float32) / np.float32(32768.0),
    )
    names = lease.names()
    assert names[-3:] == ["flush", "finish", "release"]
    assert lease.accepted == [np.asarray(p).size for p in lease.fed]


# @spec ING-FE-009, ING-FE-010
async def test_finalization_is_a_signal_after_the_drained_tail(
    started, fake_factory
) -> None:
    samples = pcm16_sine(0.1, rate=8_000)
    stream = await open_stream(started, sample_rate_hertz=8_000)
    stream.audio(pcm16_bytes(samples))
    await stream.finish()
    lease = _lease(fake_factory)
    names = lease.names()
    flush = names.index("flush")
    assert "feed" not in names[flush:]
    fed = lease.fed_samples
    # the resampler's lookahead reached the lease before ``flush``
    assert fed.size == 2 * samples.size
    expected = resample_poly(samples.astype(np.float64) / 32768.0, 2, 1)
    np.testing.assert_allclose(fed, expected, atol=1e-6)
    # ING-FE-012: a live stream is not requantized
    assert np.any(np.rint(fed * 32768.0) / 32768.0 != fed)


# @spec ING-FE-010
async def test_a_trailing_partial_sample_is_discarded(
    started, fake_factory
) -> None:
    samples = pcm16_sine(0.01)
    stream = await open_stream(started)
    stream.audio(pcm16_bytes(samples) + b"\x7f")
    await stream.finish()
    lease = _lease(fake_factory)
    assert lease.fed_samples.size == samples.size
    assert lease.names()[-3:] == ["flush", "finish", "release"]


# @spec ING-FE-011, ING-FE-007
@pytest.mark.parametrize("riva_options", [{"pre_submit_max_samples": LIMIT}])
@pytest.mark.parametrize("rate", [16_000, 8_000])
async def test_a_large_message_is_fed_in_slices_of_at_most_the_bound(
    rate, started, fake_factory
) -> None:
    count = 3_500 if rate == 16_000 else 1_750  # 3500 samples at 16 kHz
    samples = pcm16_sine(count / rate, rate=rate)
    stream = await open_stream(started, sample_rate_hertz=rate)
    stream.audio(pcm16_bytes(samples))
    await stream.finish()
    lease = _lease(fake_factory)
    sizes = [np.asarray(p).size for p in lease.fed]
    assert len(sizes) >= 4
    assert max(sizes) <= LIMIT
    assert sum(sizes) == 3_500
    assert lease.accepted == sizes
    assert lease.peak_outstanding <= LIMIT


# @spec ING-FE-007, ING-FE-011
@pytest.mark.parametrize("riva_options", [{"pre_submit_max_samples": LIMIT}])
async def test_reads_pause_while_a_feed_is_outstanding(
    started, fake_factory
) -> None:
    stream = await open_stream(started)
    lease = _lease(fake_factory)
    block = Block()
    lease.script("feed", block)
    stream.audio(pcm16_bytes(pcm16_sine(0.05)))  # 800 samples
    stream.audio(pcm16_bytes(pcm16_sine(0.05)))
    stream.audio(pcm16_bytes(pcm16_sine(0.05)))
    await until(block.entered.is_set, what="feed")
    await asyncio.sleep(0.02)
    # while ``feed`` waits on the engine the owner cannot hand over more
    # audio, so the servicer does not read the next message: the
    # configuration and the first audio message only
    assert len(stream.requests.delivered) == 2
    assert lease.names().count("feed") == 1
    block.release()
    await stream.finish()
    assert len(stream.requests.delivered) == 4
    assert sum(lease.accepted) == 3 * 800


# @spec ING-FE-007
async def test_credit_is_acknowledged_once_per_feed(
    started, fake_factory
) -> None:
    stream = await open_stream(started)
    for _ in range(5):
        stream.audio(pcm16_bytes(pcm16_sine(0.02)))
    await stream.finish()
    lease = _lease(fake_factory)
    feeds = [c for c in lease.calls if c.name == "feed"]
    assert len(lease.accepted) == len(feeds)
    assert lease.accepted == [np.asarray(c.args[0]).size for c in feeds]
    for call in feeds:
        assert callable(call.kwargs["on_accepted"])


# @spec ING-FE-007
async def test_credit_returns_at_on_accepted_while_feed_is_pending(
    make_owner, fake_factory
) -> None:
    # The model's ``feed`` calls ``on_accepted`` before it returns: the
    # fake keeps ``feed`` pending until the test fires the callback, and
    # then holds it on ``hold``. The owner's ``outstanding_samples`` must
    # fall at the callback, not when ``feed`` returns.
    owner = make_owner()
    await owner.open(cadence="560ms", locale="en-US")
    lease = _lease(fake_factory)
    lease.defer_acceptance = True
    hold = Block()
    lease.script("feed", hold)
    assert owner.outstanding_samples == 0

    feed = asyncio.ensure_future(owner.feed(np.zeros(800, np.float32)))
    await until(lambda: lease.outstanding == 800, what="feed called")
    await settle()
    assert lease.running == "feed" and not feed.done()
    assert owner.outstanding_samples == 800, "no credit taken for the feed"

    assert lease.acknowledge() == 1  # ``on_accepted(800)``, feed pending
    assert owner.outstanding_samples == 0, "credit not released by on_accepted"
    await settle()
    assert hold.entered.is_set() and not feed.done()
    assert lease.running == "feed"
    assert owner.outstanding_samples == 0

    hold.release()
    await asyncio.wait_for(feed, 2)
    assert owner.outstanding_samples == 0
    assert lease.accepted == [800]
    assert owner.calls == ["feed"]


def _constants(path: Path) -> set:
    tree = ast.parse(path.read_text())
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, (str, int))
        and not isinstance(node.value, bool)
    }


# @spec ING-FE-009, ING-FE-007
def test_no_frontend_code_computes_or_reads_a_cadence_chunk_size() -> None:
    import vllm_riva_frontend.lease as lease_module

    assert lease_module.STREAM_CADENCE == "560ms"
    # 560 ms and 1120 ms at 16 kHz, and the cadence strings outside the one
    # module that names the cadence the surfaces open with
    forbidden = {8_960, 17_920, "1120ms", "560ms"}
    offenders = {
        path.name: sorted(map(str, _constants(path) & forbidden))
        for path in sorted(PACKAGE.glob("*.py"))
        if path.name != "lease.py" and _constants(path) & forbidden
    }
    assert offenders == {}


# @spec ING-FE-015
@pytest.mark.parametrize("reason", ["input_capacity", "output_capacity"])
async def test_session_capacity_failure_is_buffer_overflow(
    reason, started, fake_factory
) -> None:
    from support.fake_session import Raise, failed

    stream = await open_stream(started)
    lease = _lease(fake_factory)
    lease.script("feed", Raise(failed(reason)))
    stream.audio(pcm16_bytes(pcm16_sine(0.05)))
    await stream.wait()
    stream.assert_status(grpc.StatusCode.RESOURCE_EXHAUSTED, "buffer_overflow")
    layer = "input" if reason == "input_capacity" else "output"
    assert layer in stream.details
    assert lease.names()[-2:] == ["abort", "release"]
