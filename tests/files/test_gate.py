"""The file gate: one counting gate for ``Recognize`` and the route."""

from __future__ import annotations

import asyncio

import grpc
import pytest
from support.fake_transcription import Block, Return, sse
from support.grpc_harness import FakeServicerContext
from support.http import assert_core_error, error_of, until
from support.riva import recognize, recognize_request

from .helpers import TrackedStream


def _pcm(seconds: float = 0.5) -> bytes:
    from support.audio import pcm16_bytes, pcm16_sine

    return pcm16_bytes(pcm16_sine(seconds))


# ------------------------------------------------------------- gate units


# @spec ING-FILE-074
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
    from vllm_riva_frontend.files import FileGate

    from vllm_riva_frontend.errors import FrontendError

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
    from vllm_riva_frontend.files import FileGate

    from vllm_riva_frontend.errors import FrontendError

    gate = FileGate(2, clock=clock)
    gate.close()
    with pytest.raises(FrontendError) as raised:
        await asyncio.wait_for(gate.acquire(clock.monotonic() + 30), 1)
    assert raised.value.code == "service_unavailable"
    assert gate.waiting == 0
    assert gate.in_flight == 0


# ---------------------------------------------------- one gate, two paths


# @spec ING-FILE-074
async def test_one_gate_covers_route_and_recognize(files_env):
    env = await files_env(riva_frontend={"max_concurrent_files": 1})
    block = Block(then=Return())
    env.service.script(block)
    route = env.request([("model", env.models.model_name())]).start()
    await asyncio.wait_for(block.entered.wait(), 2)
    assert env.plugin.file_gate.in_flight == 1

    rpc = asyncio.ensure_future(
        recognize(
            env.plugin.servicer,
            recognize_request(_pcm()),
            FakeServicerContext(),
        )
    )
    await until(lambda: env.plugin.file_gate.waiting == 1, what="rpc waits")
    assert len(env.calls) == 1, "Recognize passed a full gate"

    block.release()
    await route.finish()
    assert route.status == 200
    outcome = await asyncio.wait_for(rpc, 2)
    assert outcome.response is not None, outcome.details
    assert len(env.calls) == 2
    assert env.plugin.file_gate.in_flight == 0


# @spec ING-FILE-074
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


# @spec ING-FILE-074
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


# @spec ING-FILE-074
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


async def _drain(env, timeout: float = 1.0) -> asyncio.Future:
    """Start ``drain``; a 1 s budget is below the engine reserve.

    So ``drain`` skips the settle point and cancels every owner at once
    (vehicle design, ``drain``).
    """
    from support.app import drain_plugin

    return asyncio.ensure_future(drain_plugin(env.plugin, env.state, timeout))


# @spec ING-FILE-048
async def test_shutdown_cancels_route_file_with_503_before_headers(
    files_env,
):
    env = await files_env()
    block = Block()
    env.service.script(block)
    route = env.request().start()
    await asyncio.wait_for(block.entered.wait(), 2)
    drain = await _drain(env)
    await until(lambda: route.status is not None, what="503 sent")
    assert_core_error(route, 503, "service_unavailable")
    block.release()
    await route.finish()
    await asyncio.wait_for(drain, 5)


# @spec ING-FILE-048, ING-FILE-035
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
    drain = await _drain(env)
    await until(lambda: env.engine.aborted, what="abort")
    pull.set()
    await route.finish()
    assert len(route.starts) == 1, "a status was sent after headers"
    assert route.status == 200
    await asyncio.wait_for(drain, 5)


# @spec ING-FILE-048
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
    drain = await _drain(env)
    block.release()
    outcome = await asyncio.wait_for(rpc, 5)
    assert outcome.status is grpc.StatusCode.UNAVAILABLE
    assert outcome.details.startswith("service_unavailable:")
    assert outcome.response is None
    await asyncio.wait_for(drain, 5)


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
