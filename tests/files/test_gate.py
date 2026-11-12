"""The file gate: one counting gate for ``Recognize`` and the route."""

from __future__ import annotations

import asyncio

import grpc
import pytest
from support.fake_transcription import (
    Block,
    Return,
    sse,
    transcription_response,
)
from support.grpc_harness import FakeServicerContext
from support.http import assert_core_error, error_of, until
from support.riva import recognize, recognize_request

from .helpers import (
    GuardedCall,
    ListenerStopHook,
    TrackedStream,
    start_drain,
    start_stop,
)


def _pcm(seconds: float = 0.5) -> bytes:
    from support.audio import pcm16_bytes, pcm16_sine

    return pcm16_bytes(pcm16_sine(seconds))


# ------------------------------------------------------------- gate units


# @spec ING-FILE-080
async def test_gate_caps_in_flight_and_releases_exactly_once(clock):
    from vllm_riva_frontend.files import FileGate

    gate = FileGate(2, clock=clock)
    deadline = clock.monotonic() + 30
    first = await gate.acquire(deadline)
    second = await gate.acquire(deadline)
    assert gate.in_flight == 2
    third = asyncio.ensure_future(gate.acquire(deadline))
    await until(lambda: gate.waiting == 1, what="third waits")
    assert not third.done()

    first.release()
    first.release()  # idempotent: frees one slot, not two
    permit = await asyncio.wait_for(third, 1)
    assert gate.in_flight == 2
    assert gate.waiting == 0

    fourth = asyncio.ensure_future(gate.acquire(deadline))
    await until(lambda: gate.waiting == 1, what="fourth waits")
    assert not fourth.done(), "a repeated release over-admitted"
    second.release()
    await asyncio.wait_for(fourth, 1)
    permit.release()
    (await asyncio.sleep(0))
    assert gate.in_flight == 1
    fourth.result().release()
    assert gate.in_flight == 0


# @spec ING-FILE-019
async def test_gate_wait_expiry_is_busy_with_load_shed_authority(clock):
    from vllm_riva_frontend.errors import FrontendError
    from vllm_riva_frontend.files import FileGate

    gate = FileGate(1, clock=clock)
    held = await gate.acquire(clock.monotonic() + 30)
    waiter = asyncio.ensure_future(gate.acquire(clock.monotonic() + 30))
    await until(lambda: gate.waiting == 1, what="waiter parked")
    await clock.advance(29.9)
    assert not waiter.done()
    await clock.advance(0.2)
    with pytest.raises(FrontendError) as raised:
        await asyncio.wait_for(waiter, 1)
    assert raised.value.code == "busy"
    assert raised.value.authority == "load_shed"
    assert gate.waiting == 0
    assert gate.in_flight == 1
    held.release()
    assert gate.in_flight == 0


# @spec ING-FILE-053
async def test_closed_gate_refuses_new_waiters(clock):
    from vllm_riva_frontend.errors import FrontendError
    from vllm_riva_frontend.files import FileGate

    gate = FileGate(2, clock=clock)
    gate.close()
    with pytest.raises(FrontendError) as raised:
        await asyncio.wait_for(gate.acquire(clock.monotonic() + 30), 1)
    assert raised.value.code == "service_unavailable"
    assert gate.waiting == 0
    assert gate.in_flight == 0


# ---------------------------------------------------- one gate, two paths


# @spec ING-FILE-080
async def test_one_gate_covers_route_and_recognize_and_releases_once(
    files_env,
):
    from support.http import CORE_ROUTE, file_request

    env = await files_env(riva_frontend={"max_concurrent_files": 1})
    gate = env.plugin.file_gate
    block = Block(then=Return())
    env.service.script(block)
    route = env.request([("model", env.models.model_name())]).start()
    await asyncio.wait_for(block.entered.wait(), 2)
    assert gate.in_flight == 1

    rpc = asyncio.ensure_future(
        recognize(
            env.plugin.servicer,
            recognize_request(_pcm()),
            FakeServicerContext(),
        )
    )
    await until(lambda: gate.waiting == 1, what="rpc waits")
    assert len(env.calls) == 1, "Recognize passed a full gate"

    # A file sent to core's own route is outside the gate: it neither waits
    # for a permit nor takes one.
    core_block = Block(then=Return())
    env.service.script(core_block)
    core = file_request(env.app, path=CORE_ROUTE).start()
    await until(
        lambda: core_block.entered.is_set() or core.done, what="core's call"
    )
    assert core_block.entered.is_set(), (core.status, core.body)
    assert gate.in_flight == 1 and gate.waiting == 1
    core_block.release()
    await core.finish()
    assert core.status == 200, core.body

    block.release()
    await route.finish()
    assert route.status == 200
    outcome = await asyncio.wait_for(rpc, 2)
    assert outcome.response is not None, outcome.details
    assert len(env.calls) == 3
    await until(lambda: gate.in_flight == 0, what="release")

    # Each permit was released exactly once: the gate admits one file again
    # and makes the next wait.
    held = await gate.acquire(env.clock.monotonic() + 3600)
    extra = asyncio.ensure_future(gate.acquire(env.clock.monotonic() + 3600))
    await until(lambda: gate.waiting == 1, what="second waits")
    assert not extra.done(), "a permit was released twice"
    held.release()
    (await asyncio.wait_for(extra, 1)).release()
    assert gate.in_flight == 0


# @spec ING-FILE-080
async def test_route_permit_is_held_until_the_body_is_sent(files_env):
    env = await files_env(riva_frontend={"max_concurrent_files": 1})
    route = env.request(block_send=1).start()  # stall the body send
    await asyncio.wait_for(route.send_blocked.wait(), 2)
    assert len(env.calls) == 1
    assert env.plugin.file_gate.in_flight == 1, "released before delivery"
    route.unblock_send()
    await route.finish()
    assert route.status == 200
    await until(lambda: env.plugin.file_gate.in_flight == 0, what="release")


# @spec ING-FILE-080
async def test_stream_permit_is_held_until_the_last_event_is_sent(
    files_env,
):
    env = await files_env(riva_frontend={"max_concurrent_files": 1})
    gate_event = asyncio.Event()
    stream = TrackedStream(
        [sse({"n": 0}), sse({"n": 1}), "data: [DONE]\n\n"],
        engine=env.engine,
        blocks={1: gate_event},
    )
    env.service.script(Return(stream))
    route = env.request([("stream", "true")]).start()
    await asyncio.wait_for(stream.pull_entered.wait(), 2)
    await until(lambda: stream.pulling == 1, what="second pull")
    assert env.plugin.file_gate.in_flight == 1
    gate_event.set()
    await route.finish()
    assert route.status == 200
    assert b'"n": 1' in route.body
    await until(lambda: env.plugin.file_gate.in_flight == 0, what="release")
    assert env.engine.aborted == []


# @spec ING-FILE-080
async def test_full_gate_does_not_reserve_or_refuse_stream_capacity(
    files_env,
):
    from vllm_riva_frontend.admission import OwnerKind

    env = await files_env(riva_frontend={"max_concurrent_files": 1})
    block = Block()
    env.service.script(block)
    route = env.request().start()
    await asyncio.wait_for(block.entered.wait(), 2)
    admission = env.plugin.ingress.try_admit(OwnerKind("stream"))
    assert admission is not None, "a full file gate refused a stream"
    admission.release()
    block.release()
    await route.finish()


# ------------------------------------------------------------ gate expiry


# @spec ING-FILE-019
async def test_route_gate_wait_expiry_answers_busy_503(files_env):
    env = await files_env(riva_frontend={"max_concurrent_files": 1})
    block = Block()
    env.service.script(block)
    first = env.request().start()
    await asyncio.wait_for(block.entered.wait(), 2)
    second = env.request().start()
    await until(lambda: env.plugin.file_gate.waiting == 1, what="waiting")
    await env.clock.advance(env.plugin.config.file_gate_wait_timeout + 0.1)
    await second.finish()
    assert_core_error(second, 503, "busy")
    assert error_of(second)["message"].startswith("busy: authority: load_shed")
    assert len(env.calls) == 1
    block.release()
    await first.finish()
    assert first.status == 200


# @spec ING-FILE-019
async def test_recognize_gate_wait_expiry_is_busy(files_env):
    env = await files_env(riva_frontend={"max_concurrent_files": 1})
    block = Block()
    env.service.script(block)
    route = env.request().start()
    await asyncio.wait_for(block.entered.wait(), 2)
    rpc = asyncio.ensure_future(
        recognize(
            env.plugin.servicer,
            recognize_request(_pcm()),
            FakeServicerContext(),
        )
    )
    await until(lambda: env.plugin.file_gate.waiting == 1, what="waiting")
    await env.clock.advance(env.plugin.config.file_gate_wait_timeout + 0.1)
    outcome = await asyncio.wait_for(rpc, 2)
    assert outcome.status is grpc.StatusCode.RESOURCE_EXHAUSTED
    assert outcome.details.startswith("busy: authority: load_shed")
    block.release()
    await route.finish()


# ------------------------------------------------------------- shutdown


# @spec ING-FILE-086, ING-FILE-082
async def test_shutdown_cancels_route_file_with_503_before_headers(
    files_env,
):
    env = await files_env()
    block = Block()
    env.service.script(block)
    route = env.request().start()
    await asyncio.wait_for(block.entered.wait(), 2)
    drain = start_drain(env)
    await until(lambda: route.status is not None, what="503 sent")
    assert_core_error(route, 503, "service_unavailable")
    block.release()
    await route.finish()
    await asyncio.wait_for(drain, 5)


# @spec ING-FILE-086, ING-FILE-035
async def test_shutdown_terminates_the_event_stream_after_headers(
    files_env,
):
    env = await files_env()
    pull = asyncio.Event()
    stream = TrackedStream(
        [sse({"n": 0}), sse({"n": 1})], engine=env.engine, blocks={1: pull}
    )
    env.service.script(Return(stream))
    route = env.request([("stream", "true")]).start()
    await until(lambda: stream.pulling == 1, what="second pull")
    assert route.status == 200
    drain = start_drain(env)
    await until(lambda: env.engine.aborted, what="abort")
    pull.set()
    await route.finish()
    assert len(route.starts) == 1, "a status was sent after headers"
    assert route.status == 200
    await asyncio.wait_for(drain, 5)


# @spec ING-FILE-086, ING-FILE-082
async def test_shutdown_cancels_recognize_with_unavailable(files_env):
    env = await files_env()
    block = Block()
    env.service.script(block)
    rpc = asyncio.ensure_future(
        recognize(
            env.plugin.servicer,
            recognize_request(_pcm()),
            FakeServicerContext(),
        )
    )
    await asyncio.wait_for(block.entered.wait(), 2)
    drain = start_drain(env)
    block.release()
    outcome = await asyncio.wait_for(rpc, 5)
    assert outcome.status is grpc.StatusCode.UNAVAILABLE
    assert outcome.details.startswith("service_unavailable:")
    assert outcome.response is None
    await asyncio.wait_for(drain, 5)


# @spec ING-FILE-086, ING-FILE-089, ING-FILE-082
@pytest.mark.parametrize("shutdown", ["drain", "stop"])
async def test_shutdown_cancellation_selects_shutdown_cause(
    files_env, shutdown
):
    """An RPC termination after the listener stop began counts as shutdown.

    The client's RPC is terminated at the moment the plugin begins stopping
    its gRPC listener, with no cause selected before: the owner selects
    shutdown, not the RPC cancellation. As the spec requires of a listener
    stop test, this asserts the selected cause (projected on the fake
    context, which has no transport to lose the status) and the cleanup, not
    what a real client would receive.
    """
    env = await files_env()
    guarded = GuardedCall(env.service)
    block = Block(then=Return(transcription_response("late")))
    env.service.script(block)
    context = FakeServicerContext()
    hook = ListenerStopHook.install(env.plugin, context.cancel)
    try:
        rpc = asyncio.ensure_future(
            recognize(env.plugin.servicer, recognize_request(_pcm()), context)
        )
        await asyncio.wait_for(block.entered.wait(), 2)
        request_id = env.calls[0].request_id
        assert env.engine.aborted == []

        ended = start_drain(env) if shutdown == "drain" else start_stop(env)
        await until(lambda: hook.stops, what="the listener stop began")
        assert context.cancelled()
        await asyncio.wait({rpc}, timeout=5)
        assert rpc.done(), "Recognize did not answer the shutdown at once"
        assert not rpc.cancelled(), (
            "Recognize ended cancelled (no response),"
            " not with the shutdown cause"
        )
        outcome = rpc.result()
        # The cause: shutdown, projected as UNAVAILABLE ``service_unavailable``,
        # not the RPC cancellation (which would produce no status).
        assert outcome.response is None
        assert outcome.status is grpc.StatusCode.UNAVAILABLE, outcome.details
        assert outcome.details.startswith("service_unavailable:")

        # The cleanup: the engine request is aborted while core's call runs,
        # the call is never cancelled, and the settlement waits for it.
        await until(lambda: env.engine.aborted, what="abort")
        assert set(env.engine.aborted) == {request_id}
        assert not ended.done(), "the settlement did not wait for core's call"
        assert env.plugin.file_gate.in_flight == 1
        before = len(env.engine.aborted)
        block.release()
        await asyncio.wait_for(ended, 5)
        assert len(env.engine.aborted) > before, "no final abort"
        assert guarded.cancelled == 0, "the owner cancelled core's call"
        assert guarded.returned == 1
        assert not env.plugin.registry.records
        assert env.plugin.file_gate.in_flight == 0
        assert env.plugin.ingress.file_count == 0
    finally:
        block.release()


class _HeldCancelTask(asyncio.tasks._PyTask):  # type: ignore[name-defined]
    """A task whose cancellations wait while ``held`` is set.

    The plugin's own cancellation of the owner is held back, so the RPC
    termination the listener stop delivers (``terminate``) is the first
    cancellation the owner meets; ``deliver()`` then lets the held ones
    through.
    """

    def __init__(self, coro, *, loop) -> None:
        super().__init__(coro, loop=loop)
        self.held = True
        self.held_requests = 0

    def cancel(self, msg=None) -> bool:
        if self.held:
            self.held_requests += 1
            return not self.done()
        return super().cancel(msg)

    def terminate(self) -> None:
        """The transport's termination of the RPC: cancels at once."""
        super().cancel()

    def deliver(self) -> None:
        self.held = False
        for _ in range(self.held_requests):
            super().cancel()


# @spec ING-FILE-086, ING-LIFE-064
@pytest.mark.parametrize("shutdown", ["drain", "stop"])
async def test_listener_stop_termination_first_selects_shutdown(
    files_env, shutdown, monkeypatch
):
    """The listener stop's termination is the first cause ``Recognize`` meets.

    Unlike ``test_shutdown_cancellation_selects_shutdown_cause``, the
    plugin's own cancellation of the owner is held back until the owner has
    met the RPC's termination, which the listener stop delivers the moment
    the plugin begins stopping the listener (the transport cancels the
    handler and fires the context's callbacks). That termination counts as
    the plugin's cancellation: the owner selects shutdown, never the RPC
    cancellation of ING-FILE-089. As the spec requires of a listener stop
    test, this asserts the selected cause (projected on the fake context,
    which has no transport to lose the status) and the cleanup.
    """
    from vllm_riva_frontend import files

    # Every file owner the servicer builds, so the hook can read its cause.
    owners: list[files.FileRequestOwner] = []
    init = files.FileRequestOwner.__init__

    def recording_init(self, *args, **kwargs) -> None:
        init(self, *args, **kwargs)
        owners.append(self)

    monkeypatch.setattr(files.FileRequestOwner, "__init__", recording_init)
    env = await files_env()
    guarded = GuardedCall(env.service)
    block = Block(then=Return(transcription_response("late")))
    env.service.script(block)
    context = FakeServicerContext()
    loop = asyncio.get_running_loop()
    rpc = _HeldCancelTask(
        recognize(env.plugin.servicer, recognize_request(_pcm()), context),
        loop=loop,
    )
    # The causes the owner holds the moment the listener stop terminates
    # the RPC, read inside the hook immediately before the termination and
    # asserted below (an assertion raised in the hook would surface in the
    # plugin's stop, not here).
    causes_at_stop: list[object] = []

    def terminate() -> None:
        causes_at_stop.extend(owner.cause for owner in owners)
        context.cancel()
        rpc.terminate()

    hook = ListenerStopHook.install(env.plugin, terminate)
    try:
        await asyncio.wait_for(block.entered.wait(), 2)
        request_id = env.calls[0].request_id
        assert len(owners) == 1, "Recognize did not build one file owner"
        ended = start_drain(env) if shutdown == "drain" else start_stop(env)
        await until(lambda: hook.stops, what="the listener stop began")
        # No cause was selected before the termination: a shutdown cause
        # chosen early (by drain or stop before the listener stop) would
        # satisfy the assertions below without the termination counting.
        assert causes_at_stop == [None], causes_at_stop
        assert context.cancelled()
        # The owner meets the termination first and starts its cleanup.
        await until(lambda: env.engine.aborted, what="abort")
        assert set(env.engine.aborted) == {request_id}
        rpc.deliver()
        await asyncio.wait({rpc}, timeout=5)
        assert rpc.done(), "Recognize did not answer the shutdown at once"
        assert not rpc.cancelled(), (
            "Recognize ended cancelled (the RPC cancellation, no response),"
            " not with the shutdown cause"
        )
        outcome = rpc.result()
        assert outcome.response is None
        assert outcome.status is grpc.StatusCode.UNAVAILABLE, outcome.details
        assert outcome.details.startswith("service_unavailable:")
        # The cleanup: core's call is never cancelled and is awaited.
        assert not ended.done(), "the settlement did not wait for core's call"
        block.release()
        await asyncio.wait_for(ended, 5)
        assert guarded.cancelled == 0, "the owner cancelled core's call"
        assert guarded.returned == 1
        assert not env.plugin.registry.records
        assert env.plugin.file_gate.in_flight == 0
        assert env.plugin.ingress.file_count == 0
    finally:
        block.release()
        rpc.deliver()
        await asyncio.wait({rpc}, timeout=5)


# ------------------------------------------------------- ingress order


# @spec ING-FILE-055, ING-ADM-011
async def test_closed_ingress_answers_before_content_type_and_body(
    files_env,
):
    env = await files_env()
    env.plugin.ingress.close()
    exchange = await env.request(
        headers=[("content-type", "application/json")]
    ).finish()
    assert_core_error(exchange, 503, "service_unavailable")
    assert exchange.chunks_delivered == 0
    assert not env.plugin.registry.records
    assert env.calls == []


# @spec ING-FILE-055, ING-ADM-010
async def test_full_file_sub_cap_answers_busy_before_the_body(files_env):
    env = await files_env(riva_frontend={"max_concurrent_files": 1})
    # max_file_owners defaults to max_concurrent_files + 2 = 3.
    block = Block()
    env.service.script(block)
    running = env.request().start()
    await asyncio.wait_for(block.entered.wait(), 2)
    waiters = [env.request().start() for _ in range(2)]
    await until(lambda: env.plugin.file_gate.waiting == 2, what="two waiters")
    assert env.plugin.ingress.file_count == 3
    refused = await env.request().finish()
    assert_core_error(refused, 503, "busy")
    assert refused.chunks_delivered == 0
    block.release()
    await running.finish()
    for waiter in waiters:
        await waiter.finish()
        assert waiter.status == 200


# @spec ING-FILE-055
async def test_valid_request_waits_for_the_permit_before_calling_core(
    files_env,
):
    env = await files_env(riva_frontend={"max_concurrent_files": 1})
    held = await env.plugin.file_gate.acquire(env.clock.monotonic() + 3600)
    exchange = env.request().start()
    await until(lambda: env.plugin.file_gate.waiting == 1, what="waiting")
    assert exchange.chunks_delivered == len(exchange.chunks)
    assert env.calls == []
    held.release()
    await exchange.finish()
    assert exchange.status == 200
    assert len(env.calls) == 1


# @spec ING-FILE-053, ING-EP-082, ING-EP-102
async def test_core_draining_refuses_an_admitted_upload_at_the_gate(
    files_env,
):
    """Core's ``state.draining`` alone, with no ``drain`` call, closes the gate.

    The upload is admitted while core serves and is still sending its body
    when core sets ``state.draining``; it then reaches the full gate and is
    refused with ``service_unavailable``, and the slot freed afterwards
    grants it no permit (core is never called for it).
    """
    env = await files_env(riva_frontend={"max_concurrent_files": 1})
    block = Block()
    env.service.script(block)
    running = env.request().start()
    await asyncio.wait_for(block.entered.wait(), 2)
    upload = env.request(stall_before=0).start()
    await until(lambda: env.plugin.ingress.file_count == 2, what="admitted")
    env.state.draining = True  # core's flag; no plugin drain call
    upload.resume()
    await until(lambda: upload.status is not None, what="the refusal")
    assert_core_error(upload, 503, "service_unavailable")
    assert env.plugin.file_gate.waiting == 0
    block.release()
    await running.finish()
    await upload.finish()
    assert len(env.calls) == 1, "the refused upload reached core"
    assert env.plugin.file_gate.in_flight == 0
    assert env.plugin.drain_deadline is None


# @spec ING-FILE-053, ING-EP-082
async def test_gate_reads_core_draining_directly(clock):
    from vllm_riva_frontend.errors import FrontendError
    from vllm_riva_frontend.files import FileGate

    draining = [False]
    gate = FileGate(2, clock=clock, serving=lambda: not draining[0])
    held = await gate.acquire(clock.monotonic() + 30)
    draining[0] = True
    assert gate.closed
    with pytest.raises(FrontendError) as raised:
        await asyncio.wait_for(gate.acquire(clock.monotonic() + 30), 1)
    assert raised.value.code == "service_unavailable"
    assert gate.in_flight == 1
    held.release()
    assert gate.in_flight == 0
