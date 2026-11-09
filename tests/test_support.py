"""Tests of the shared test support itself.

They import only ``tests/support`` and core, never the frontend package, so
they pass before the port and keep passing while it is rewritten.
"""

from __future__ import annotations

import asyncio
import io
import json
import struct
from argparse import Namespace

import grpc
import numpy as np
import pytest
from fastapi import FastAPI, Request, UploadFile, WebSocket
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from starlette.datastructures import State
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse
from support import audio, riva
from support import realtime as realtime_support
from support.app import (
    build_app,
    drain_plugin,
    drain_plugins,
    find_route,
    http_client,
    init_state,
    middleware_stack,
    plugins_ready,
    serve_plugin,
    start_plugins,
    stop_plugins,
    supported_tasks,
)
from support.asgi import AsgiWebSocket, BlockingSendWebSocket, asgi_http
from support.clock import FakeClock, RealClock, settle
from support.fake_engine import FakeEngineClient
from support.fake_session import (
    Block,
    FakeSessionFactory,
    Raise,
    Return,
    SeamError,
    failed,
    fake_entry_points,
    refused,
)
from support.fake_transcription import (
    FakeServingModels,
    FakeTranscriptionServing,
    error_response,
    event_stream,
    sse,
)
from support.grpc_harness import (
    FakeServicerContext,
    RequestStream,
    RpcAbort,
    aio_channel,
    free_port,
    serve,
)
from support.http import (
    BOUNDARY,
    HttpExchange,
    assert_core_error,
    chunked,
    content_type,
    error_of,
    file_request,
    multipart_body,
    until,
)
from support.state import RecordingShutdown, make_args, make_state

# ------------------------------------------------------------------ clock


async def test_fake_clock_wakes_sleepers_in_deadline_order(clock: FakeClock):
    order: list[str] = []

    async def sleeper(name: str, seconds: float) -> None:
        await clock.sleep(seconds)
        order.append(name)

    tasks = [
        asyncio.create_task(sleeper("late", 3.0)),
        asyncio.create_task(sleeper("early", 1.0)),
    ]
    await clock.settle()
    assert clock.sleepers == 2
    assert clock.next_wake == pytest.approx(1001.0)
    await clock.advance(1.0)
    assert order == ["early"]
    await clock.advance(2.0)
    assert order == ["early", "late"]
    assert clock.monotonic() == pytest.approx(1003.0)
    await asyncio.gather(*tasks)


async def test_until_polls_in_real_time_and_never_moves_the_clock(
    clock: FakeClock,
):
    from support import clock as clock_module

    flag: list[bool] = []
    asyncio.get_running_loop().call_soon(flag.append, True)
    await clock_module.until(lambda: flag, what="the callback")
    assert clock.monotonic() == pytest.approx(1000.0)
    with pytest.raises(AssertionError, match="never"):
        await clock_module.until(lambda: False, timeout=0.01, what="never")
    ran: list[int] = []

    async def worker() -> None:
        for step in range(3):
            ran.append(step)
            await asyncio.sleep(0)

    task = asyncio.create_task(worker())
    await clock_module.settle(10)
    assert task.done() and ran == [0, 1, 2]
    assert until is clock_module.until  # ``support.http`` re-exports it


async def test_fake_clock_wait_for_returns_before_the_deadline(clock):
    async def quick() -> str:
        await clock.sleep(0.5)
        return "done"

    waiter = asyncio.create_task(clock.wait_for(quick(), clock.monotonic() + 2))
    await clock.advance(0.5)
    assert await waiter == "done"
    assert clock.sleepers == 0


async def test_fake_clock_wait_for_times_out_and_cancels_the_awaitable(clock):
    cancelled = asyncio.Event()

    async def slow() -> None:
        try:
            await clock.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    waiter = asyncio.create_task(clock.wait_for(slow(), clock.monotonic() + 1))
    await clock.advance(1.0)
    with pytest.raises(TimeoutError):
        await waiter
    assert cancelled.is_set()


async def test_real_clock_wait_for_uses_wall_time():
    clock = RealClock()
    with pytest.raises(TimeoutError):
        await clock.wait_for(asyncio.sleep(5), clock.monotonic() + 0.01)
    assert await clock.wait_for(asyncio.sleep(0, "x"), clock.monotonic() + 1)


# ``FakeClock`` and ``RealClock`` give ``wait_for`` the same semantics, so a
# rule that holds under the fake (for example "start ``release``, then cancel
# it") holds only if the code under test makes it hold.


@pytest.fixture(params=["fake", "real"])
def any_clock(request):
    return FakeClock() if request.param == "fake" else RealClock()


async def test_wait_for_with_an_expired_deadline_never_starts_a_coroutine(
    any_clock,
):
    started: list[bool] = []

    async def work() -> str:
        started.append(True)
        return "ran"

    with pytest.raises(TimeoutError):
        await any_clock.wait_for(work(), any_clock.monotonic() - 1)
    await settle(5)
    assert started == []


async def test_wait_for_with_an_expired_deadline_cancels_a_pending_future(
    any_clock,
):
    future = asyncio.get_running_loop().create_future()
    with pytest.raises(TimeoutError):
        await any_clock.wait_for(future, any_clock.monotonic() - 1)
    assert future.cancelled()


async def test_wait_for_returns_an_already_completed_future_at_any_deadline(
    any_clock,
):
    for offset in (-1.0, 0.0, 5.0):
        future = asyncio.get_running_loop().create_future()
        future.set_result("done")
        deadline = any_clock.monotonic() + offset
        assert await any_clock.wait_for(future, deadline) == "done"


async def test_cancelling_the_waiter_cancels_the_awaitable_on_both_clocks(
    any_clock,
):
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow() -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    waiter = asyncio.create_task(
        any_clock.wait_for(slow(), any_clock.monotonic() + 60)
    )
    await asyncio.wait_for(entered.wait(), 1)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert cancelled.is_set()
    if isinstance(any_clock, FakeClock):
        assert any_clock.sleepers == 0


async def test_fake_clock_wait_for_keeps_its_absolute_deadline(clock):
    waiter = asyncio.create_task(
        clock.wait_for(asyncio.Event().wait(), clock.monotonic() + 2)
    )
    await clock.settle()
    assert clock.next_wake == pytest.approx(1002.0)
    clock.set(1001.0)
    await clock.advance(0.9)
    assert not waiter.done()
    await clock.advance(0.2)
    with pytest.raises(TimeoutError):
        await waiter


async def test_advance_reaches_a_wait_registered_after_many_loop_turns(clock):
    woke: list[float] = []

    async def late_sleeper() -> None:
        for _ in range(500):  # far more turns than ``advance`` settles
            await asyncio.sleep(0)
        await clock.wait_for(asyncio.Event().wait(), clock.monotonic() + 1)

    async def run() -> None:
        try:
            await late_sleeper()
        except TimeoutError:
            woke.append(clock.monotonic())

    task = asyncio.create_task(run())
    await clock.advance(1.5, settle=1)
    assert woke == [pytest.approx(1001.5)]
    await task


async def test_settle_runs_the_loop_until_it_is_idle():
    steps: list[int] = []

    async def chain() -> None:
        for step in range(300):
            steps.append(step)
            await asyncio.sleep(0)

    task = asyncio.create_task(chain())
    await settle(1)
    assert task.done() and len(steps) == 300


# --------------------------------------------------------------- session


async def test_fake_factory_open_returns_a_recording_lease(fake_factory):
    lease = await fake_factory.open(cadence="560ms", locale="en-US")
    assert fake_factory.leases == [lease]
    assert fake_factory.names() == ["open"]
    assert fake_factory.calls[0].kwargs == {
        "cadence": "560ms",
        "locale": "en-US",
    }
    assert lease.state == "open"
    assert lease.locale == "en-US"


async def test_fake_factory_refuses_unserved_cadence_and_locale(fake_factory):
    with pytest.raises(SeamError) as cadence:
        await fake_factory.open(cadence="80ms", locale="en-US")
    assert (cadence.value.session_error, cadence.value.reason) == (
        "refused",
        "cadence",
    )
    with pytest.raises(SeamError) as locale:
        await fake_factory.open(cadence="560ms", locale="xx-XX")
    assert locale.value.reason == "locale"
    assert fake_factory.leases == []


async def test_fake_lease_feed_credits_once_before_returning(fake_factory):
    lease = await fake_factory.open(cadence="560ms", locale="auto")
    credits: list[int] = []
    samples = np.zeros(1600, dtype=np.float32)
    assert await lease.feed(samples, on_accepted=credits.append) == []
    assert credits == [1600]
    assert lease.accepted == [1600]
    lease.script("feed", Return(["hello"]))
    assert await lease.feed(samples, on_accepted=credits.append) == ["hello"]
    assert credits == [1600, 1600]
    assert lease.names() == ["feed", "feed"]


async def test_fake_lease_counts_outstanding_credit_and_can_defer_it(
    fake_factory,
):
    lease = await fake_factory.open(cadence="560ms", locale="auto")
    credits: list[int] = []
    await lease.feed(np.zeros(300, np.float32), on_accepted=credits.append)
    assert (lease.outstanding, lease.peak_outstanding) == (0, 300)
    assert lease.running is None

    # Deferred acceptance happens only while ``feed`` is still pending: the
    # callback fires inside the call, and ``feed`` returns after it.
    lease.defer_acceptance = True
    hold = Block()
    lease.script("feed", hold)
    feed = asyncio.ensure_future(
        lease.feed(np.zeros(200, np.float32), on_accepted=credits.append)
    )
    await settle()
    assert credits == [300] and lease.outstanding == 200
    assert lease.running == "feed" and not feed.done()
    assert not hold.entered.is_set(), "the behaviour ran before acceptance"
    assert lease.acknowledge() == 1
    assert credits == [300, 200] and lease.outstanding == 0
    assert lease.running == "feed" and not feed.done()
    await settle()
    assert hold.entered.is_set() and not feed.done()
    hold.release()
    assert await feed == []
    assert lease.running is None
    assert lease.accepted == [300, 200]
    assert lease.peak_outstanding == 300
    assert lease.acknowledge() == 0

    # A ``feed`` cancelled before its acknowledgement accepted nothing and
    # breaks the lease, as a cancelled ``feed`` breaks the model's.
    other = await fake_factory.open(cadence="560ms", locale="auto")
    other.defer_acceptance = True
    pending = asyncio.ensure_future(
        other.feed(np.zeros(100, np.float32), on_accepted=credits.append)
    )
    await settle()
    assert other.outstanding == 100
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert other.outstanding == 0 and other.accepted == []
    assert (other.state, other.failure_reason) == ("failed", "cancelled")
    assert other.acknowledge() == 0
    assert credits == [300, 200]


async def test_fake_lease_after_hook_runs_when_the_call_succeeds(
    fake_factory,
):
    lease = await fake_factory.open(cadence="560ms", locale="auto")
    seen: list[str] = []
    lease.after("release", lambda: seen.append(lease.state))
    lease.script("abort", Raise(failed("state")))
    with pytest.raises(SeamError):
        await lease.abort()
    await lease.release()
    assert seen == ["released"]
    await lease.release()
    assert seen == ["released"]  # the hook runs once


async def test_fake_lease_fed_samples_concatenates_every_feed(fake_factory):
    lease = await fake_factory.open(cadence="560ms", locale="auto")
    assert lease.fed_samples.dtype == np.float32
    assert lease.fed_samples.size == 0
    first = np.arange(3, dtype=np.float32)
    second = np.arange(3, 5, dtype=np.float32)
    await lease.feed(first, on_accepted=lambda n: None)
    await lease.feed(second, on_accepted=lambda n: None)
    np.testing.assert_array_equal(lease.fed_samples, np.arange(5))


async def test_fake_lease_block_parks_until_released(fake_factory):
    lease = await fake_factory.open(cadence="560ms", locale="auto")
    block = Block()
    lease.script("flush", block)
    task = asyncio.create_task(lease.flush())
    await asyncio.wait_for(block.entered.wait(), 1)
    assert not task.done()
    block.release(Return("final text"))
    assert await task == "final text"
    assert lease.state == "flushed"


async def test_fake_lease_enforces_the_state_table(fake_factory):
    lease = await fake_factory.open(cadence="560ms", locale="auto")
    with pytest.raises(SeamError) as info:
        await lease.finish()
    assert (info.value.session_error, info.value.reason) == ("failed", "state")
    assert await lease.flush() == ""
    with pytest.raises(SeamError):
        await lease.abort()
    await lease.finish()
    await lease.release()
    assert lease.state == "released"
    assert fake_factory.released == [lease]
    await lease.release()  # idempotent after release
    assert lease.names() == [
        "finish",
        "flush",
        "abort",
        "finish",
        "release",
        "release",
    ]


async def test_fake_lease_failure_and_cancellation_break_the_lease(
    fake_factory,
):
    lease = await fake_factory.open(cadence="560ms", locale="auto")
    lease.script("feed", Raise(failed("expired")))
    with pytest.raises(SeamError):
        await lease.feed(np.zeros(1, np.float32), on_accepted=lambda n: None)
    assert (lease.state, lease.failure_reason) == ("failed", "expired")
    with pytest.raises(SeamError) as info:
        await lease.flush()
    assert info.value.reason == "expired"
    await lease.abort()
    assert lease.state == "aborted"

    other = await fake_factory.open(cadence="560ms", locale="auto")
    block = Block()
    other.script("flush", block)
    task = asyncio.create_task(other.flush())
    await block.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (other.state, other.failure_reason) == ("failed", "cancelled")


async def test_fake_lease_rejects_overlapping_calls(fake_factory):
    lease = await fake_factory.open(cadence="560ms", locale="auto")
    block = Block()
    lease.script("feed", block)
    task = asyncio.create_task(
        lease.feed(np.zeros(1, np.float32), on_accepted=lambda n: None)
    )
    await block.entered.wait()
    with pytest.raises(RuntimeError):
        await lease.update_locale("de-DE")
    lease.teardown()  # allowed at any time
    block.release()
    await task
    assert lease.teardown_calls == 1


async def test_fake_factory_drain_returns_faults_once(fake_factory):
    fault = RuntimeError("orphan cleanup failed")
    fake_factory.faults.append(fault)
    assert await fake_factory.drain(1234.5) == [fault]
    assert await fake_factory.drain(1235.5) == []
    assert fake_factory.drains == [1234.5, 1235.5]


async def test_fake_factory_teardown_all_is_terminal(fake_factory):
    lease = await fake_factory.open(cadence="560ms", locale="auto")
    parked = asyncio.create_task(asyncio.sleep(100))
    fake_factory.tasks.add(parked)
    assert fake_factory.teardown_all() == {parked}
    assert fake_factory.closed and fake_factory.teardown_calls == 1
    with pytest.raises(asyncio.CancelledError):
        await parked
    with pytest.raises(SeamError) as info:
        await fake_factory.open(cadence="560ms", locale="auto")
    assert (info.value.session_error, info.value.reason) == (
        "failed",
        "cancelled",
    )
    with pytest.raises(SeamError):
        await lease.flush()
    lease.teardown()
    assert lease.state == "torn_down"


async def test_fake_lease_cancelled_release_is_handed_over(fake_factory):
    lease = await fake_factory.open(cadence="560ms", locale="auto")
    await lease.abort()
    block = Block()
    lease.script("release", block)
    task = asyncio.create_task(lease.release())
    await block.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fake_factory.handed_over == [lease]
    assert lease.state == "aborted"


async def test_fake_factory_drain_waits_for_cleanup_until_its_deadline(clock):
    factory = FakeSessionFactory(clock=clock)
    lease = await factory.open(cadence="560ms", locale="auto")
    await lease.abort()
    cleanup = Block()
    factory.script("cleanup", cleanup)
    release_block = Block()
    lease.script("release", release_block)
    release = asyncio.create_task(lease.release())
    await release_block.entered.wait()
    release.cancel()
    with pytest.raises(asyncio.CancelledError):
        await release
    assert factory.handed_over == [lease] and factory.owns(lease)

    drain = asyncio.create_task(factory.drain(clock.monotonic() + 2))
    await clock.advance(1.9)
    assert not drain.done()
    await clock.advance(0.2)
    assert await drain == []
    assert factory.owns(lease), "a running cleanup stays with the factory"

    cleanup.release()
    assert await factory.drain(clock.monotonic() + 2) == []
    assert not factory.owns(lease)


async def _hand_over_with(factory, behaviour):
    """Open a lease, abort it and cancel its ``release`` after it began."""
    lease = await factory.open(cadence="560ms", locale="auto")
    await lease.abort()
    factory.script("cleanup", behaviour)
    block = Block()
    lease.script("release", block)
    release = asyncio.create_task(lease.release())
    await block.entered.wait()
    release.cancel()
    with pytest.raises(asyncio.CancelledError):
        await release
    assert lease in factory.handed_over
    return lease


async def test_fake_factory_drain_returns_each_cleanup_fault_once(clock):
    factory = FakeSessionFactory(clock=clock)
    early = RuntimeError("cleanup failed at once")
    await _hand_over_with(factory, Raise(early))
    await settle()
    (task,) = factory.cleanup.values()
    assert task.done() and task.exception() is early
    assert await factory.drain(clock.monotonic() + 1) == [early]
    assert await factory.drain(clock.monotonic() + 1) == []

    # A cleanup still running at a drain's deadline fails later: the next
    # drain returns its fault, and only that one.
    late = RuntimeError("cleanup failed after the deadline")
    cleanup = Block(then=Raise(late))
    lease = await _hand_over_with(factory, cleanup)
    drain = asyncio.create_task(factory.drain(clock.monotonic() + 1))
    await clock.advance(1.1)
    assert await drain == []
    assert factory.owns(lease)
    cleanup.release()
    await settle()
    assert not factory.owns(lease)
    assert await factory.drain(clock.monotonic() + 1) == [late]
    assert await factory.drain(clock.monotonic() + 1) == []

    # A cancelled cleanup is not a fault.
    await _hand_over_with(factory, Block())
    factory.teardown_all()
    await settle()
    assert await factory.drain(clock.monotonic() + 1) == []


async def test_fake_factory_teardown_all_cancels_running_cleanup():
    factory = FakeSessionFactory()
    lease = await factory.open(cadence="560ms", locale="auto")
    await lease.abort()
    factory.script("cleanup", Block())
    block = Block()
    lease.script("release", block)
    release = asyncio.create_task(lease.release())
    await block.entered.wait()
    release.cancel()
    with pytest.raises(asyncio.CancelledError):
        await release
    (task,) = factory.teardown_all()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not factory.owns(lease)


def test_seam_error_helpers_carry_both_attributes():
    exc = refused("capacity")
    assert (exc.session_error, exc.reason) == ("refused", "capacity")
    assert failed("engine_dead").reason == "engine_dead"
    assert getattr(RuntimeError(), "session_error", None) is None


def test_fake_entry_points_mimic_importlib_metadata():
    factory = FakeSessionFactory()
    declining = FakeSessionFactory()
    points = fake_entry_points(
        one=factory.constructor, two=lambda engine, model_config: None
    )
    found = points(group="vllm_riva_frontend.session_factories")
    assert [p.name for p in found] == ["one", "two"]
    assert found[0].load()("engine", "model_config") is factory
    assert factory.constructed_with == ("engine", "model_config")
    assert found[1].load()("engine", "model_config") is None
    assert points(group="vllm.endpoint_plugins") == []
    assert declining.constructor(None, None) is declining


# ---------------------------------------------------------------- engine


async def test_fake_engine_uses_core_attribute_names():
    engine = FakeEngineClient(
        riva_frontend={"grpc_bind": "127.0.0.1:50051"},
        stream_admission={"never_fed_timeout": 15},
        max_num_active_seqs=8,
    )
    config = engine.vllm_config
    assert config.cache_config.num_gpu_blocks == 17
    assert config.scheduler_config.max_num_seqs == 16
    assert config.scheduler_config.max_num_active_seqs == 8
    assert config.parallel_config.data_parallel_size == 1
    assert config.shutdown_timeout == 60.0
    assert config.additional_config["riva_frontend"]["grpc_bind"].endswith(
        "50051"
    )
    assert config.additional_config["stream_admission"] == {
        "never_fed_timeout": 15
    }
    assert engine.model_config is config.model_config
    assert engine.model_config.served_model_name == engine.model_config.model
    assert not engine.errored
    await engine.check_health()
    await engine.abort("transcribe-1")
    await engine.abort(["a", "b"])
    assert engine.aborted == ["transcribe-1", "a", "b"]
    assert await engine.get_supported_tasks() == ("transcription", "realtime")


async def test_fake_engine_death_is_observable():
    from vllm.v1.engine.exceptions import EngineDeadError

    engine = FakeEngineClient()
    engine.die()
    assert engine.errored
    assert isinstance(engine.dead_error, EngineDeadError)
    with pytest.raises(EngineDeadError):
        await engine.check_health()
    engine.set_capacity(num_gpu_blocks=None)
    assert engine.vllm_config.cache_config.num_gpu_blocks is None


def test_stream_deadline_defaults_match_core():
    from vllm.v1.core.stream_admission import StreamAdmissionConfig

    defaults = StreamAdmissionConfig()
    assert (
        defaults.never_fed_timeout,
        defaults.parked_idle_timeout,
        defaults.draining_timeout,
    ) == (15.0, 60.0, 40.0)


# ---------------------------------------------------------- transcription


def _request(data: bytes = b"RIFF") -> object:
    from vllm.entrypoints.speech_to_text.transcription.protocol import (
        TranscriptionRequest,
    )

    return TranscriptionRequest(
        file=UploadFile(io.BytesIO(data), filename="clip.wav", size=len(data))
    )


def _raw_request(headers: list[tuple[bytes, bytes]]) -> Request:
    return Request({"type": "http", "headers": headers})


async def test_fake_transcription_derives_core_request_id():
    models = FakeServingModels("served-name")
    service = FakeTranscriptionServing(models, default_text="hello")
    raw = _raw_request([(b"x-request-id", b"abc-123")])
    response = await service.create_transcription(b"audio", _request(), raw)
    assert response.text == "hello"
    assert service.request_ids == ["transcribe-abc-123"]
    assert raw.state.request_metadata.request_id == "transcribe-abc-123"
    assert service.calls[0].audio_data == b"audio"
    assert service.calls[0].headers == {"x-request-id": "abc-123"}

    await service.create_transcription(b"", _request())
    assert service.request_ids[1].startswith("transcribe-")
    assert service.request_ids[1] != "transcribe-"


async def test_fake_transcription_scripts_errors_and_event_streams():
    service = FakeTranscriptionServing()
    service.script(
        Return(error_response("bad", 400, param="language")),
        Return(event_stream(sse("a"), sse("b"))),
        Raise(ValueError("Invalid or unsupported audio file.")),
    )
    error = await service.create_transcription(b"", _request())
    assert (error.error.code, error.error.param) == (400, "language")
    stream = await service.create_transcription(b"", _request())
    assert [event async for event in stream] == ["data: a\n\n", "data: b\n\n"]
    assert sse({"n": 0}) == 'data: {"n": 0}\n\n'
    assert stream.pulls == 3
    with pytest.raises(ValueError):
        await service.create_transcription(b"", _request())
    service.shutdown()
    assert service.shutdown_calls == 1
    assert service.models.model_name() == "nvidia/nemotron-asr-streaming-test"
    assert service.models.model_name_calls == 1
    assert service.model_cls.supported_languages["en"] == "english"


async def test_core_upload_helper_reads_an_in_memory_upload():
    from vllm.entrypoints.speech_to_text.base.utils import (
        read_upload_with_limit,
    )
    from vllm.exceptions import VLLMValidationError

    payload = b"\x00" * 1000
    upload = UploadFile(io.BytesIO(payload), filename="a.wav", size=1000)
    assert await read_upload_with_limit(upload, max_size_mb=0.001) == payload
    too_big = UploadFile(io.BytesIO(payload), filename="a.wav", size=1000)
    with pytest.raises(VLLMValidationError) as info:
        await read_upload_with_limit(too_big, max_size_mb=0.0001)
    assert info.value.parameter == "audio_filesize_mb"


# ------------------------------------------------------------------ state


def test_make_state_fills_what_core_fills():
    engine = FakeEngineClient()
    state = make_state(engine)
    assert state.engine_client is engine
    assert state.vllm_config is engine.vllm_config
    assert state.openai_serving_models.model_name() == engine.model_config.model
    assert isinstance(
        state.openai_serving_transcription, FakeTranscriptionServing
    )
    assert state.server_load_metrics == 0
    assert state.enable_server_load_tracking is False
    assert not hasattr(state, "draining")
    assert isinstance(state.request_shutdown, RecordingShutdown)
    state.request_shutdown()
    assert (
        state.request_shutdown.calls == 1 and state.request_shutdown.requested
    )

    stream_only = make_state(engine, tasks=("realtime",))
    assert not hasattr(stream_only, "openai_serving_transcription")
    args = make_args(api_server_count=2)
    assert args.api_server_count == 2 and args.api_key is None


# -------------------------------------------------------------------- app


class _RecordingPlugin:
    """A minimal endpoint plugin that records the lifecycle core drives."""

    name = "recording"
    required_tasks = ("realtime",)
    owns_non_http_listener = True

    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []
        self.ready_value = True

    def attach_router(self, app: FastAPI) -> None:
        self.events.append(("attach_router", app))

        @app.get("/v1/health/ready")
        async def ready_route() -> JSONResponse:
            return JSONResponse({"status": "ready"})

        @app.post("/v1/audio/transcriptions")
        async def override(raw_request: Request) -> JSONResponse:
            return JSONResponse({"override": True})

    async def init_state(self, engine_client, state: State, args) -> None:
        self.events.append(("init_state", (engine_client, state, args)))

    async def start(self, state: State) -> None:
        self.events.append(("start", state))

    def ready(self, state: State) -> bool:
        return self.ready_value

    async def drain(self, state: State, timeout: float) -> None:
        self.events.append(("drain", (state.draining, timeout)))

    async def stop(self, state: State) -> None:
        self.events.append(("stop", state))


def test_build_app_has_core_routes_by_task():
    app = build_app()
    assert supported_tasks(app) == ("transcription", "realtime")
    assert find_route(app, "GET", "/health") is not None
    assert find_route(app, "POST", "/v1/audio/transcriptions") is not None
    assert find_route(app, "POST", "/v1/audio/translations") is not None
    assert any(route.path == "/v1/realtime" for route in app.routes)

    streams = build_app(supported_tasks=("realtime",))
    assert find_route(streams, "POST", "/v1/audio/transcriptions") is None
    assert any(route.path == "/v1/realtime" for route in streams.routes)


def test_build_app_attaches_plugins_through_core_override():
    plugin = _RecordingPlugin()
    app = build_app([plugin], api_key="secret")
    assert plugin.events[0][0] == "attach_router"
    assert app.state.endpoint_plugins == [plugin]
    assert find_route(app, "GET", "/v1/health/ready") is not None
    route = find_route(app, "POST", "/v1/audio/transcriptions")
    assert route is not None and route.endpoint.__name__ == "override"
    translations = find_route(app, "POST", "/v1/audio/translations")
    assert translations is not None
    assert translations.endpoint.__name__ != "override"
    names = [cls.__name__ for cls in middleware_stack(app)]
    assert names == [
        "WebSocketMetricsMiddleware",
        "AuthenticationMiddleware",
        "CORSMiddleware",
    ]


async def test_init_state_runs_core_phase_b_and_health():
    plugin = _RecordingPlugin()
    app = build_app([plugin], api_key="secret")
    engine = FakeEngineClient()
    state = await init_state(app, engine)
    assert state is app.state
    assert state.draining is False
    kinds = [event[0] for event in plugin.events]
    assert kinds == ["attach_router", "init_state"]
    _, (client, seen_state, args) = plugin.events[1]
    assert client is engine and seen_state is state
    assert args is app.state.args
    async with http_client(app) as client_http:
        assert (await client_http.get("/health")).status_code == 200
        assert (await client_http.get("/v1/health/ready")).status_code == 401
    async with http_client(app, api_key="secret") as authed:
        response = await authed.get("/v1/health/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}
        assert (await authed.get("/health")).status_code == 200


async def test_init_state_refuses_listener_plugins_without_one_api_server():
    plugin = _RecordingPlugin()
    app = build_app([plugin], args=make_args(api_server_count=2))
    with pytest.raises(ValueError, match="api-server-count"):
        await init_state(app, FakeEngineClient())


async def test_lifecycle_helpers_drive_core_functions():
    plugin = _RecordingPlugin()
    app = build_app([plugin])
    state = await init_state(app, FakeEngineClient())
    await start_plugins(state)
    assert state.started_endpoint_plugins == [plugin]
    assert plugins_ready(state)
    plugin.ready_value = False
    assert not plugins_ready(state)
    async with http_client(app) as client:
        assert (await client.get("/health")).status_code == 503
    plugin.ready_value = True
    await drain_plugins(state, 5.0)
    assert state.draining is True
    drained = [event for event in plugin.events if event[0] == "drain"]
    assert len(drained) == 1
    draining_seen, timeout = drained[0][1]
    assert draining_seen is True and 0 < timeout <= 5.0
    async with http_client(app) as client:
        assert (await client.get("/health")).status_code == 503
    await stop_plugins(state)
    assert [event[0] for event in plugin.events][-1] == "stop"
    assert state.started_endpoint_plugins == []


async def test_core_transcription_route_reaches_the_fake_service():
    app = build_app()
    engine = FakeEngineClient()
    service = FakeTranscriptionServing(default_text="transcript")
    await init_state(app, engine, transcription=service)
    async with http_client(app) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("clip.wav", b"RIFF-bytes", "audio/wav")},
            data={"model": engine.model_config.model, "language": "en"},
            headers={"X-Request-Id": "fixed-id"},
        )
    assert response.status_code == 200
    assert response.json()["text"] == "transcript"
    assert service.calls[0].audio_data == b"RIFF-bytes"
    assert service.calls[0].request.language == "en"
    assert service.request_ids == ["transcribe-fixed-id"]


async def test_realtime_stand_in_is_reachable_over_asgi_websocket():
    app = build_app()
    await init_state(app, FakeEngineClient())
    socket = AsgiWebSocket(app, "/v1/realtime?intent=other")
    assert await socket.connect()
    assert await socket.receive_json() == {"type": "core.realtime"}
    closing = await socket.receive()
    assert closing["type"] == "websocket.close"
    assert socket.close_code == 1000
    await socket.wait_closed()
    assert app.state.core_realtime_scopes[0]["query_string"] == b"intent=other"


async def test_asgi_websocket_counts_consumed_messages_and_pushes_sync():
    taken: list[str] = []

    async def app(scope, receive, send) -> None:
        assert (await receive())["type"] == "websocket.connect"
        await send({"type": "websocket.accept"})
        taken.append((await receive())["text"])
        await send({"type": "websocket.close", "code": 1000})

    socket = AsgiWebSocket(app, "/ws")
    assert await socket.connect()
    assert socket.consumed == 1
    socket.push_json({"n": 1})
    socket.push_json({"n": 2})
    await socket.wait_closed()
    assert taken == ['{"n": 1}']
    assert socket.consumed == 2, "the app read one message, not both"
    assert len(socket.sent) == 2


async def test_asgi_websocket_reports_denied_handshakes():
    app = build_app(api_key="secret")
    await init_state(app, FakeEngineClient())
    socket = AsgiWebSocket(app, "/v1/realtime")
    assert not await socket.connect()
    assert socket.denied is not None and socket.denied["status"] == 401


async def test_asgi_http_delivers_chunks_and_disconnects():
    app = FastAPI()
    seen: dict[str, object] = {}

    @app.post("/echo")
    async def echo(request: Request) -> JSONResponse:
        chunks = [chunk async for chunk in request.stream() if chunk]
        seen["chunks"] = chunks
        return JSONResponse({"length": sum(len(c) for c in chunks)})

    @app.post("/watch")
    async def watch(request: Request) -> JSONResponse:
        seen["disconnected"] = await request.is_disconnected()
        try:
            async for _ in request.stream():
                pass
        except ClientDisconnect:
            seen["after"] = "client_disconnect"
            return JSONResponse({"disconnected": True})
        return JSONResponse({"disconnected": False})

    response = await asgi_http(
        app, "POST", "/echo", body=[b"ab", b"cde"], headers=[("x-a", "1")]
    )
    assert response.status == 200 and response.completed
    assert response.json() == {"length": 5}
    assert seen["chunks"] == [b"ab", b"cde"]

    response = await asgi_http(
        app, "POST", "/watch", body=[b"ab", b"cd"], disconnect_after=1
    )
    assert seen["disconnected"] is False
    assert seen["after"] == "client_disconnect"
    assert response.json() == {"disconnected": True}


# ------------------------------------------------------------------- gRPC


async def test_fake_servicer_context_counts_down_to_its_absolute_deadline(
    clock,
):
    context = FakeServicerContext(time_remaining=20.0, clock=clock)
    assert context.deadline == pytest.approx(1020.0)
    assert context.time_remaining() == pytest.approx(20.0)
    await clock.advance(8.0)
    assert context.time_remaining() == pytest.approx(12.0)
    await clock.advance(13.0)
    assert context.time_remaining() == 0.0
    context.set_time_remaining(None)
    assert context.time_remaining() is None and context.deadline is None
    assert FakeServicerContext().time_remaining() is None
    real = FakeServicerContext(time_remaining=60.0)
    assert 59.0 < real.time_remaining() <= 60.0
    call = riva.StreamingCall(_FakeServicer(), clock, time_remaining=3.0)
    assert call.context.clock is clock
    assert call.context.deadline == pytest.approx(clock.monotonic() + 3.0)
    call.half_close()
    await call.wait()


async def test_fake_servicer_context_abort_and_cancel():
    context = FakeServicerContext(time_remaining=2.5)
    assert 0.0 < context.time_remaining() <= 2.5
    with pytest.raises(RpcAbort) as info:
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "code: detail")
    assert info.value.code is grpc.StatusCode.INVALID_ARGUMENT
    assert context.aborts[0].details == "code: detail"
    assert context.code is grpc.StatusCode.INVALID_ARGUMENT

    fired: list[object] = []
    context.add_done_callback(fired.append)
    assert not context.cancelled()
    context.cancel()
    context.cancel()
    assert context.cancelled() and fired == [context]
    context.add_done_callback(fired.append)
    assert fired == [context, context]


async def test_request_stream_is_test_driven():
    stream = RequestStream(["config"])
    iterator = stream.__aiter__()
    assert await iterator.__anext__() == "config"
    pull = asyncio.create_task(iterator.__anext__())
    await asyncio.wait_for(stream.waiting.wait(), 1)
    stream.push("audio")
    assert await pull == "audio"
    assert not stream.waiting.is_set()
    stream.close()
    with pytest.raises(StopAsyncIteration):
        await iterator.__anext__()
    assert stream.pulls == 3 and stream.delivered == ["config", "audio"]
    failing = RequestStream()
    failing.fail(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        await failing.__aiter__().__anext__()


async def test_real_grpc_server_on_a_free_port():
    servicer = health.aio.HealthServicer()

    def register(server: grpc.aio.Server) -> None:
        health_pb2_grpc.add_HealthServicer_to_server(servicer, server)

    port = free_port()
    assert isinstance(port, int) and port > 0
    server, address = await serve(register, port=port)
    try:
        assert address == f"127.0.0.1:{port}"
        await servicer.set("", health_pb2.HealthCheckResponse.SERVING)
        async with aio_channel(address) as channel:
            stub = health_pb2_grpc.HealthStub(channel)
            reply = await stub.Check(
                health_pb2.HealthCheckRequest(service=""), timeout=5
            )
        assert reply.status == health_pb2.HealthCheckResponse.SERVING
    finally:
        await server.stop(0)


# ------------------------------------------------------------------ audio


def test_g711_anchor_values_follow_itu_tables():
    assert audio.mulaw_decode(bytes([0xFF]))[0] == 0
    assert audio.mulaw_decode(bytes([0x00]))[0] == -32124
    assert audio.mulaw_decode(bytes([0x80]))[0] == 32124
    assert audio.alaw_decode(bytes([0xD5]))[0] == 8
    assert audio.alaw_decode(bytes([0x55]))[0] == -8
    assert len(set(audio.alaw_table().tolist())) == 256
    assert len(set(audio.mulaw_table().tolist())) == 255  # two zeros


def test_g711_codes_round_trip_through_decode_and_encode():
    codes = np.arange(256, dtype=np.uint8)
    alaw = audio.alaw_encode(audio.alaw_decode(codes.tobytes()))
    assert np.array_equal(alaw, codes)
    mulaw = audio.mulaw_encode(audio.mulaw_decode(codes.tobytes()))
    keep = codes != 0x7F  # negative zero encodes as positive zero
    assert np.array_equal(mulaw[keep], codes[keep])
    samples = audio.pcm16_sine(0.01, rate=8000)
    decoded = audio.mulaw_decode(audio.g711_bytes(samples, "mulaw"))
    assert np.max(np.abs(decoded.astype(int) - samples.astype(int))) < 1024
    decoded = audio.alaw_decode(audio.g711_bytes(samples, "alaw"))
    assert np.max(np.abs(decoded.astype(int) - samples.astype(int))) < 1024


def test_pcm16_fixtures_are_deterministic_and_bounded():
    tone = audio.pcm16_sine(0.5)
    assert tone.dtype == np.int16 and tone.shape == (8000,)
    assert np.array_equal(tone, audio.pcm16_sine(0.5))
    assert np.max(np.abs(tone.astype(int))) <= 16384
    ramp = audio.pcm16_ramp(4, start=32766)
    assert ramp.tolist() == [32766, 32767, 32767, 32767]
    raw = audio.pcm16_bytes(ramp)
    assert raw == struct.pack("<4h", 32766, 32767, 32767, 32767)
    assert np.array_equal(audio.pcm16_from_bytes(raw), ramp)
    floats = audio.float_from_pcm16(np.array([-32768, 16384], np.int16))
    assert floats.dtype == np.float32 and floats.tolist() == [-1.0, 0.5]
    assert audio.silence_pcm16(0.001, rate=8000).shape == (8,)


def test_wav_bytes_writes_a_canonical_header_and_sentinels():
    samples = audio.pcm16_ramp(3)
    wav = audio.wav_pcm16(samples, rate=8000)
    assert wav[:4] == b"RIFF" and wav[8:16] == b"WAVEfmt "
    code, channels, rate, byte_rate, align, bits = struct.unpack(
        "<HHIIHH", wav[20:36]
    )
    assert (code, channels, rate, byte_rate, align, bits) == (
        1,
        1,
        8000,
        16000,
        2,
        16,
    )
    assert wav[36:40] == b"data"
    assert struct.unpack("<I", wav[40:44])[0] == 6
    assert wav[44:] == audio.pcm16_bytes(samples)
    assert struct.unpack("<I", wav[4:8])[0] == len(wav) - 8

    unknown = audio.wav_bytes(
        b"\x00\x01",
        encoding="mulaw",
        rate=8000,
        data_length=audio.DATA_LENGTH_UNKNOWN,
        extra_chunks=[(b"LIST", b"abc")],
    )
    offset = audio.wav_header_size([(b"LIST", b"abc")])
    assert offset == 44 + 8 + 4
    assert unknown[offset - 8 : offset - 4] == b"data"
    assert struct.unpack("<I", unknown[offset - 4 : offset])[0] == 0xFFFFFFFF
    assert struct.unpack("<H", unknown[20:22])[0] == 7
    assert struct.unpack("<H", unknown[34:36])[0] == 8
    assert unknown[36:40] == b"LIST"


def test_make_args_is_a_namespace_core_middleware_accepts():
    args = make_args(api_key=["k"], enable_request_id_headers=True)
    assert isinstance(args, Namespace)
    app = build_app(args=args, supported_tasks=("realtime",))
    names = [cls.__name__ for cls in middleware_stack(app)]
    assert names == [
        "WebSocketMetricsMiddleware",
        "XRequestIdMiddleware",
        "AuthenticationMiddleware",
        "CORSMiddleware",
    ]


# ---------------------------------------------------------- Riva driver


class _FakeServicer:
    """A stand-in servicer in the pinned handler forms."""

    def __init__(self, *, abort_after: int | None = None) -> None:
        self.abort_after = abort_after
        self.seen: list[object] = []

    async def StreamingRecognize(self, requests, context):  # noqa: N802
        index = 0
        async for request in requests:
            self.seen.append(request)
            if request.HasField("streaming_config"):
                continue
            if self.abort_after is not None and index >= self.abort_after:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "invalid_audio: bad (fields: audio_content)",
                )
            index += 1
            yield riva.rasr.StreamingRecognizeResponse(
                results=[
                    riva.rasr.StreamingRecognitionResult(
                        alternatives=[
                            riva.rasr.SpeechRecognitionAlternative(
                                transcript=f"interim {index}"
                            )
                        ],
                        is_final=False,
                    )
                ]
            )
        yield riva.rasr.StreamingRecognizeResponse(
            results=[
                riva.rasr.StreamingRecognitionResult(
                    alternatives=[
                        riva.rasr.SpeechRecognitionAlternative(
                            transcript="final"
                        )
                    ],
                    is_final=True,
                )
            ]
        )

    async def Recognize(self, request, context):  # noqa: N802
        if not request.audio:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "invalid_audio: empty"
            )
        return riva.rasr.RecognizeResponse(
            results=[
                riva.rasr.SpeechRecognitionResult(
                    alternatives=[
                        riva.rasr.SpeechRecognitionAlternative(
                            transcript=request.config.language_code
                        )
                    ]
                )
            ]
        )


def test_riva_message_builders_use_proto_field_names():
    config = riva.recognition_config(
        encoding="MULAW", sample_rate_hertz=8_000, language_code="de-DE"
    )
    assert config.encoding == riva.MULAW
    assert config.sample_rate_hertz == 8_000
    assert config.language_code == "de-DE"
    assert config.audio_channel_count == 1 and config.max_alternatives == 1
    request = riva.config_request(
        interim_results=False, runtime_config={"k": "v"}, encoding=riva.ALAW
    )
    assert request.streaming_config.config.encoding == riva.ALAW
    assert request.streaming_config.interim_results is False
    assert dict(request.runtime_config) == {"k": "v"}
    assert riva.audio_request(b"ab").audio_content == b"ab"
    unary = riva.recognize_request(b"xy", language_code="en-US")
    assert unary.audio == b"xy" and unary.config.language_code == "en-US"
    assert riva.catalog_code("busy: authority: load_shed") == "busy"
    assert riva.catalog_code("") is None


async def test_streaming_call_drives_the_async_generator_handler(clock):
    servicer = _FakeServicer()
    call = riva.StreamingCall(servicer, clock)
    call.configure(language_code="en-US")
    call.audio(b"\x00\x01")
    call.audio(b"\x00\x02")
    await call.until(lambda: len(call.responses) == 2, what="two interims")
    assert not call.done
    assert await call.finish() is call
    assert call.requests.closed
    assert call.interims == ["interim 1", "interim 2"]
    assert call.finals == ["final"]
    assert len(call.final_responses) == 1
    assert call.status is grpc.StatusCode.OK and call.code is None
    call.assert_ok()
    assert servicer.seen[0].HasField("streaming_config")


async def test_streaming_call_records_aborts_and_handler_errors(clock):
    call = riva.StreamingCall(_FakeServicer(abort_after=0), clock)
    call.configure()
    call.audio(b"\x00")
    await call.wait()
    assert call.status is grpc.StatusCode.INVALID_ARGUMENT
    assert call.code == "invalid_audio"
    call.assert_status(
        grpc.StatusCode.INVALID_ARGUMENT, "invalid_audio", "audio_content"
    )
    call.assert_rejected(grpc.StatusCode.INVALID_ARGUMENT, "invalid_audio")
    with pytest.raises(AssertionError):
        call.assert_ok()

    class _Broken:
        async def StreamingRecognize(self, requests, context):  # noqa: N802
            raise RuntimeError("bug")
            yield  # pragma: no cover

    broken = riva.StreamingCall(_Broken(), clock)
    with pytest.raises(AssertionError, match="handler raised"):
        await broken.wait()
    assert isinstance(broken.error, RuntimeError)


async def test_streaming_call_cancel_fires_the_context_callbacks(clock):
    call = riva.StreamingCall(_FakeServicer(), clock, time_remaining=3.0)
    fired: list[object] = []
    call.context.add_done_callback(fired.append)
    call.cancel()
    assert call.context.cancelled() and fired == [call.context]
    call.half_close()
    await call.wait()
    await call.settle(2)
    await call.advance(1.0)
    assert clock.monotonic() == pytest.approx(1001.0)


async def test_recognize_classifies_the_unary_outcome():
    servicer = _FakeServicer()
    outcome = await riva.recognize(
        servicer, riva.recognize_request(b"a", language_code="de-DE")
    )
    assert outcome.abort is None and outcome.error is None
    assert outcome.status is grpc.StatusCode.OK
    assert outcome.transcript == "de-DE"
    refused = await riva.recognize(servicer, riva.recognize_request(b""))
    assert refused.response is None and refused.transcript is None
    assert refused.status is grpc.StatusCode.INVALID_ARGUMENT
    assert refused.code == "invalid_audio"
    assert refused.context is not None and refused.context.aborts


# ------------------------------------------------------- Realtime driver


def _realtime_app() -> FastAPI:
    """A tiny app speaking the pinned Realtime event shapes."""
    app = FastAPI()

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_json(
            {"type": "conversation.created", "event_id": "e1"}
        )
        while True:
            event = await websocket.receive_json()
            kind = event["type"]
            if kind == "transcription_session.update":
                await websocket.send_json(
                    {
                        "type": "transcription_session.updated",
                        "event_id": "e2",
                        "session": event["session"],
                    }
                )
            elif kind == "input_audio_buffer.append":
                await websocket.send_json(
                    {
                        "type": realtime_support.DELTA,
                        "event_id": "e3",
                        "item_id": "i",
                        "content_index": 0,
                        "delta": event["audio"],
                    }
                )
            elif kind == "input_audio_buffer.commit":
                await websocket.send_json(
                    {"type": "input_audio_buffer.committed", "event_id": "e4"}
                )
            elif kind == "input_audio_buffer.done":
                await websocket.send_json(
                    {
                        "type": realtime_support.COMPLETED,
                        "event_id": "e5",
                        "transcript": "done",
                        "is_last_result": True,
                    }
                )
                await websocket.close(code=1000)
                return
            else:
                await websocket.send_json(
                    {
                        "type": "error",
                        "event_id": "e6",
                        "error": {
                            "code": "busy",
                            "message": "busy: cap",
                            "param": "a,b",
                            "authority": "load_shed",
                        },
                    }
                )
                await websocket.close(code=1013)
                return

    return app


def test_realtime_builders_and_error_readers():
    session = realtime_support.default_session(language=None)
    assert session["input_audio_transcription"]["language"] == ""
    assert session["input_audio_format"] == "pcm16"
    cell = realtime_support.session_cell("g711_ulaw", 8_000)
    assert cell["input_audio_params"] == {
        "sample_rate_hz": 8_000,
        "num_channels": 1,
    }
    assert realtime_support.update_event(session)["session"] is session
    assert realtime_support.append_event(b"\x00")["audio"] == "AA=="
    event = {
        "type": "error",
        "error": {"code": "busy", "param": "a, b", "authority": "provider"},
    }
    assert realtime_support.error_code(event) == "busy"
    assert realtime_support.error_fields(event) == ["a", "b"]
    assert realtime_support.error_authority(event) == "provider"
    assert realtime_support.error_code({"type": "x"}) is None
    assert realtime_support.error_fields({"type": "x", "error": {}}) == []


async def test_realtime_client_drives_a_session_to_its_close(clock):
    client = realtime_support.RealtimeClient(_realtime_app(), clock)
    updated = await client.open_session()
    assert updated["session"]["modalities"] == ["text"]
    await client.append(b"\x01")
    delta = await client.until_type(realtime_support.DELTA)
    assert delta["delta"] == "AQ=="
    await client.commit()
    await client.done_event()
    before = await client.assert_closes(1000)
    assert [e["type"] for e in before] == [
        realtime_support.COMMITTED,
        realtime_support.COMPLETED,
    ]
    assert client.close_code == 1000
    assert client.of_type(realtime_support.COMPLETED)[0]["is_last_result"]
    assert client.types()[0] == realtime_support.CREATED
    assert client.terminal is None
    await client.wait_closed()
    assert client.done


async def test_realtime_client_reads_the_pinned_error_event():
    client = realtime_support.RealtimeClient(_realtime_app())
    await client.open_session()
    await client.send({"type": "not.an.event"})
    events = await client.outcome()
    assert client.close_code == 1013
    terminal = client.terminal
    assert terminal is not None and terminal is events[-2]
    assert realtime_support.error_code(terminal) == "busy"
    assert realtime_support.error_fields(terminal) == ["a", "b"]
    assert realtime_support.error_authority(terminal) == "load_shed"
    assert realtime_support.is_close(events[-1])
    assert await client.drain_events() == []


async def test_blocking_send_websocket_stalls_the_apps_sends():
    app = _realtime_app()
    ws = BlockingSendWebSocket(app, realtime_support.REALTIME_URL)
    client = realtime_support.RealtimeClient(app, ws=ws)
    assert await client.connect()
    assert (await client.next())["type"] == realtime_support.CREATED
    ws.block = True
    await client.update(realtime_support.default_session())
    for _ in range(20):
        await asyncio.sleep(0)
    assert ws.stalled == 1
    (parked,) = ws.parked
    assert parked["type"] == "websocket.send"
    assert json.loads(parked["text"])["type"] == realtime_support.UPDATED
    with pytest.raises(TimeoutError):
        await client.next(timeout=0.05)
    ws.unblocked.set()
    assert (await client.next())["type"] == realtime_support.UPDATED


# ----------------------------------------------------------- HttpExchange


def _exchange_app() -> FastAPI:
    app = FastAPI()

    @app.post("/sink")
    async def sink(request: Request) -> JSONResponse:
        total = 0
        try:
            async for chunk in request.stream():
                total += len(chunk)
        except ClientDisconnect:
            return JSONResponse({"disconnected": True})
        return JSONResponse({"length": total})

    @app.post("/two")
    async def two(request: Request):  # type: ignore[no-untyped-def]
        from starlette.responses import StreamingResponse

        async def body():  # type: ignore[no-untyped-def]
            yield b"one"
            yield b"two"

        return StreamingResponse(body())

    return app


def test_multipart_body_builder_writes_parts_in_order():
    body = multipart_body(
        [("model", "m")], file=b"abc", filename="x.wav", file_first=False
    )
    assert body.startswith(b"--" + BOUNDARY + b"\r\n")
    assert body.index(b'name="model"') < body.index(b'name="file"')
    assert body.endswith(b"--" + BOUNDARY + b"--\r\n")
    assert b'filename="x.wav"' in body and b"\r\n\r\nabc\r\n" in body
    assert b'name="file"' not in multipart_body([("a", "b")], file=None)
    boundary = BOUNDARY.decode()
    assert content_type() == f"multipart/form-data; boundary={boundary}"
    assert chunked(b"abcde", 2) == [b"ab", b"cd", b"e"]
    assert chunked(b"", 2) == [b""]


async def test_http_exchange_delivers_chunks_stalls_and_disconnects():
    app = _exchange_app()
    plain = HttpExchange(app, [], [b"ab", b"cde"], path="/sink")
    await plain.finish()
    assert plain.status == 200 and plain.completed
    assert plain.json() == {"length": 5}
    assert plain.chunks_delivered == 2 and plain.error is None

    stalled = HttpExchange(
        app, [], [b"ab", b"cd"], path="/sink", stall_before=1
    ).start()
    await until(lambda: stalled.chunks_delivered == 1, what="first chunk")
    assert not stalled.done
    stalled.resume()
    await stalled.finish()
    assert stalled.json() == {"length": 4}

    dropped = HttpExchange(
        app, [], [b"ab", b"cd"], path="/sink", stall_before=1
    ).start()
    await until(lambda: dropped.chunks_delivered == 1, what="first chunk")
    dropped.disconnect()
    dropped.resume()
    await dropped.finish()
    assert dropped.json() == {"disconnected": True}


async def test_http_exchange_blocks_and_fails_sends():
    app = _exchange_app()
    blocked = HttpExchange(app, [], [b""], path="/two", block_send=1).start()
    await asyncio.wait_for(blocked.send_blocked.wait(), 2)
    assert blocked.status == 200 and not blocked.done
    blocked.unblock_send()
    await blocked.finish()
    assert blocked.body == b"onetwo" and blocked.completed

    failing = HttpExchange(app, [], [b""], path="/two", fail_send=1)
    await failing.finish()
    assert failing.status == 200 and not failing.completed
    assert isinstance(failing.error, OSError) or failing.body == b""


def test_file_request_builds_a_multipart_exchange():
    exchange = file_request(
        FastAPI(), [("model", "m")], file=b"abc", extra_headers=[("x-a", "1")]
    )
    headers = dict(exchange.headers)
    assert headers["content-type"].startswith("multipart/form-data")
    assert int(headers["content-length"]) == sum(map(len, exchange.chunks))
    assert headers["x-a"] == "1"
    assert exchange.path == "/v1/audio/transcriptions"
    assert (b"host", b"testserver") in exchange.scope["headers"]
    no_length = file_request(FastAPI(), with_length=False, body=b"raw")
    assert "content-length" not in dict(no_length.headers)
    assert no_length.chunks == [b"raw"]


async def test_assert_core_error_reads_cores_error_shape():
    app = FastAPI()

    @app.post("/err")
    async def err() -> JSONResponse:
        return JSONResponse(
            {
                "error": {
                    "message": "malformed_request: no boundary",
                    "type": "BadRequestError",
                    "param": "content-type",
                    "code": 400,
                }
            },
            status_code=400,
        )

    exchange = await HttpExchange(app, [], [b""], path="/err").finish()
    assert_core_error(exchange, 400, "malformed_request", param="content-type")
    with pytest.raises(AssertionError):
        assert_core_error(exchange, 400, "request_too_large")
    assert error_of(exchange)["type"] == "BadRequestError"


# --------------------------------------------------------- serve_plugin


async def test_serve_plugin_builds_inits_and_starts_in_one_call():
    plugin = _RecordingPlugin()
    engine = FakeEngineClient()
    served = await serve_plugin(plugin, engine, api_key="secret")
    assert served.started and served.state is served.app.state
    assert served.engine is engine
    assert served.transcription is not None
    assert served.state.openai_serving_transcription is served.transcription
    assert served.models.model_name() == engine.model_config.served_model_name
    kinds = [event[0] for event in plugin.events]
    assert kinds == ["attach_router", "init_state", "start"]
    async with http_client(served.app, api_key="secret") as client:
        assert (await client.get("/v1/health/ready")).status_code == 200
    await served.stop()
    assert not served.started
    assert [event[0] for event in plugin.events][-1] == "stop"
    await served.stop()  # idempotent
    assert [event[0] for event in plugin.events].count("stop") == 1

    cold = await serve_plugin(
        _RecordingPlugin(), FakeEngineClient(), tasks=("realtime",), start=False
    )
    assert not cold.started and cold.transcription is None
    assert not hasattr(cold.state, "openai_serving_transcription")
    await cold.start()
    assert cold.started
    await cold.stop()


async def test_drain_plugin_sets_draining_and_passes_seconds(clock):
    plugin = _RecordingPlugin()
    served = await serve_plugin(plugin, FakeEngineClient())
    await drain_plugin(plugin, served.state, 7.5)
    assert served.state.draining is True
    drained = [event for event in plugin.events if event[0] == "drain"]
    assert len(drained) == 1
    draining_seen, timeout = drained[0][1]
    assert draining_seen is True and timeout == 7.5
    await served.stop()


def test_fake_transcription_keeps_the_served_models_sampling_settings():
    service = FakeTranscriptionServing()
    assert service.default_sampling_params == {}
    tuned = FakeTranscriptionServing(
        default_sampling_params={"temperature": 0.7}
    )
    assert tuned.default_sampling_params == {"temperature": 0.7}
    tuned.default_sampling_params["top_p"] = 0.9
    assert tuned.default_sampling_params == {"temperature": 0.7, "top_p": 0.9}
