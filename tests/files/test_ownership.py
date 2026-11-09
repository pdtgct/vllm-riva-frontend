"""Request ID, abort ownership, terminal cause, bounds and accounting."""

from __future__ import annotations

import asyncio
import json

import pytest
from support.fake_transcription import (
    Block,
    FakeTranscriptionServing,
    Raise,
    Return,
    error_response,
    sse,
    transcription_response,
)
from support.http import (
    HttpExchange,
    assert_core_error,
    chunked,
    content_type,
    multipart_body,
    until,
)

from .helpers import GuardedCall, TrackedStream, is_uuid

ABORT_PERIOD = 0.25


def _record_aborts(env):
    """Record ``(request_id, permits in flight)`` at every engine abort."""
    seen: list[tuple[str, int]] = []
    inner = env.engine.abort

    async def abort(request_id):
        gate = env.plugin.file_gate
        seen.append((request_id, gate.in_flight if gate else -1))
        await inner(request_id)

    env.engine.abort = abort
    return seen


# ----------------------------------------------------------- request IDs


# @spec ING-FILE-026
async def test_route_fixes_the_engine_request_id(files_env):
    env = await files_env()
    exchange = env.request(
        extra_headers=[("X-Request-Id", "client-1"), ("x-request-id", "c2")]
    )
    await exchange.finish()
    assert exchange.status == 200, exchange.body
    (call,) = env.calls
    prefix, _, suffix = call.request_id.partition("-")
    assert prefix == "transcribe"
    assert is_uuid(suffix), call.request_id
    raw_headers = call.raw_request.scope["headers"]
    ids = [v for k, v in raw_headers if k.lower() == b"x-request-id"]
    assert ids == [suffix.encode()]
    assert (b"x-request-id", suffix.encode()) in raw_headers
    # The original request is not mutated.
    assert exchange.scope["headers"] == exchange.original_headers
    assert (b"x-request-id", b"client-1") in exchange.scope["headers"]
    # Other headers are kept on the copy.
    assert (b"content-type", content_type().encode()) in raw_headers


# @spec ING-FILE-026
async def test_each_request_gets_a_fresh_engine_request_id(files_env):
    env = await files_env()
    for _ in range(3):
        await env.request(extra_headers=[("x-request-id", "same")]).finish()
    ids = env.service.request_ids
    assert len(set(ids)) == 3
    assert "transcribe-same" not in ids


# @spec ING-FILE-026, ING-FILE-027
def test_build_raw_request_has_one_lowercase_request_id():
    from vllm_riva_frontend.files import build_raw_request

    request = build_raw_request()
    headers = request.scope["headers"]
    assert len(headers) == 1
    name, value = headers[0]
    assert name == b"x-request-id"
    assert is_uuid(value.decode())


# @spec ING-FILE-026
def test_build_raw_request_over_a_scope_replaces_client_ids():
    from vllm_riva_frontend.files import build_raw_request

    scope = {
        "type": "http",
        "headers": [
            (b"x-request-id", b"a"),
            (b"host", b"h"),
            (b"x-request-id", b"b"),
        ],
    }
    original = list(scope["headers"])
    request = build_raw_request(scope)
    headers = request.scope["headers"]
    ids = [v for k, v in headers if k == b"x-request-id"]
    assert len(ids) == 1 and is_uuid(ids[0].decode())
    assert (b"host", b"h") in headers
    assert scope["headers"] == original


# ------------------------------------------------- abort and settlement


# @spec ING-FILE-068, ING-FILE-029, ING-FILE-067, ING-FILE-007
async def test_disconnect_during_core_call_aborts_until_settled(files_env):
    env = await files_env()
    guarded = GuardedCall(env.service)
    aborts = _record_aborts(env)
    block = Block(then=Return(transcription_response("late")))
    env.service.script(block)
    exchange = env.request().start()
    await asyncio.wait_for(block.entered.wait(), 2)
    request_id = env.calls[0].request_id

    exchange.disconnect()
    await until(lambda: len(aborts) >= 1, what="first abort")
    first = len(aborts)
    await env.clock.advance(ABORT_PERIOD)
    await until(lambda: len(aborts) >= first + 1, what="retry")
    await env.clock.advance(ABORT_PERIOD)
    await until(lambda: len(aborts) >= first + 2, what="second retry")
    assert {rid for rid, _ in aborts} == {request_id}
    assert env.plugin.file_gate.in_flight == 1, "released before settlement"

    before_return = len(aborts)
    block.release()
    await until(lambda: env.plugin.file_gate.in_flight == 0, what="release")
    await exchange.finish()
    assert len(aborts) > before_return, "no final abort after the return"
    assert aborts[-1][1] == 1, "permit released before the final abort"
    assert guarded.cancelled == 0, "the owner cancelled core's call"
    assert guarded.returned == 1
    assert exchange.status is None, "a response after the disconnect"


# @spec ING-FILE-068, ING-FILE-069, ING-FILE-029
async def test_stream_disconnect_waits_for_the_outstanding_pull(files_env):
    env = await files_env()
    aborts = _record_aborts(env)
    pull = asyncio.Event()
    stream = TrackedStream(
        [sse({"n": 0}), sse({"n": 1}), sse({"n": 2})],
        engine=env.engine,
        blocks={1: pull},
    )
    env.service.script(Return(stream))
    exchange = env.request([("stream", "true")]).start()
    await until(lambda: stream.pulling == 1, what="second pull")
    exchange.disconnect()
    await until(lambda: len(aborts) >= 1, what="abort")
    await env.clock.advance(ABORT_PERIOD)
    await until(lambda: len(aborts) >= 2, what="retry")
    assert not stream.closed, "closed while a pull was running"

    before = len(aborts)
    pull.set()
    await until(lambda: stream.closed, what="aclose")
    assert stream.aborts_at_close is not None
    assert stream.aborts_at_close > before, "closed before the final abort"
    assert stream.max_active == 1, "overlapping pulls"
    assert stream.pull_cancelled == 0
    await until(lambda: env.plugin.file_gate.in_flight == 0, what="release")
    assert len(aborts) == stream.aborts_at_close
    await exchange.finish()


# @spec ING-FILE-069, ING-FILE-007
async def test_cause_after_exhaustion_closes_without_abort(files_env):
    env = await files_env()
    stream = TrackedStream([sse({"n": 0})], engine=env.engine)
    env.service.script(Return(stream))
    # messages: start, event, final empty body (sent after exhaustion)
    exchange = env.request([("stream", "true")], block_send=2).start()
    await asyncio.wait_for(exchange.send_blocked.wait(), 2)
    assert stream.index == 1
    exchange.disconnect()
    await settle_and_unblock(exchange)
    await exchange.finish()
    await until(lambda: env.plugin.file_gate.in_flight == 0, what="release")
    assert env.engine.aborted == []
    assert stream.max_active == 1


async def settle_and_unblock(exchange: HttpExchange) -> None:
    for _ in range(20):
        await asyncio.sleep(0)
    exchange.unblock_send()


# @spec ING-FILE-007, ING-FILE-073
async def test_disconnect_during_complete_send_suppresses_only(files_env):
    env = await files_env()
    exchange = env.request(block_send=0).start()
    await asyncio.wait_for(exchange.send_blocked.wait(), 2)
    exchange.disconnect()
    await settle_and_unblock(exchange)
    await exchange.finish()
    await until(lambda: env.plugin.file_gate.in_flight == 0, what="release")
    assert env.engine.aborted == []


# @spec ING-FILE-073, ING-FILE-064, ING-ADM-016
async def test_stalled_reader_of_complete_response_is_cut_without_abort(
    files_env,
):
    env = await files_env()
    exchange = env.request(block_send=0).start()
    await asyncio.wait_for(exchange.send_blocked.wait(), 2)
    timeout = env.plugin.config.response_send_timeout
    await env.clock.advance(timeout - 1)
    assert env.plugin.file_gate.in_flight == 1
    await env.clock.advance(2)
    await until(lambda: env.plugin.file_gate.in_flight == 0, what="release")
    assert env.engine.aborted == []
    exchange.unblock_send()
    await exchange.finish()
    assert len(exchange.starts) <= 1


# @spec ING-FILE-073, ING-FILE-068
async def test_stalled_reader_of_unfinished_stream_aborts(files_env):
    env = await files_env()
    stream = TrackedStream(
        [sse({"n": 0}), sse({"n": 1}), sse({"n": 2})], engine=env.engine
    )
    env.service.script(Return(stream))
    exchange = env.request([("stream", "true")], block_send=1).start()
    await asyncio.wait_for(exchange.send_blocked.wait(), 2)
    await env.clock.advance(env.plugin.config.response_send_timeout + 0.1)
    await until(lambda: env.engine.aborted, what="abort")
    request_id = env.calls[0].request_id
    assert set(env.engine.aborted) == {request_id}
    await until(lambda: env.plugin.file_gate.in_flight == 0, what="release")
    assert stream.closed
    exchange.unblock_send()
    await exchange.finish()


# ------------------------------------------------------------ deadlines


# @spec ING-FILE-064
async def test_execution_deadline_starts_at_the_permit(files_env):
    env = await files_env(riva_frontend={"max_concurrent_files": 1})
    first_block, second_block = Block(), Block()
    env.service.script(first_block, second_block)
    first = env.request().start()
    await asyncio.wait_for(first_block.entered.wait(), 2)
    second = env.request().start()
    await until(lambda: env.plugin.file_gate.waiting == 1, what="waiting")
    await env.clock.advance(20)  # waits at the gate, within its bound
    first_block.release()
    await first.finish()
    await asyncio.wait_for(second_block.entered.wait(), 2)

    budget = env.plugin.config.file_execution_timeout
    await env.clock.advance(budget - 1)
    assert second.status is None, "deadline counted the gate wait"
    assert env.engine.aborted == []
    await env.clock.advance(2)
    await until(lambda: env.engine.aborted, what="abort at the deadline")
    second_block.release()
    await second.finish()
    assert_core_error(second, 504, "request_timeout")


# @spec ING-FILE-064, ING-FILE-035
async def test_stream_deadline_runs_to_generator_exhaustion(files_env):
    env = await files_env()
    pull = asyncio.Event()
    stream = TrackedStream(
        [sse({"n": 0}), sse({"n": 1})], engine=env.engine, blocks={1: pull}
    )
    env.service.script(Return(stream))
    exchange = env.request([("stream", "true")]).start()
    await until(lambda: stream.pulling == 1, what="second pull")
    assert exchange.status == 200
    await env.clock.advance(env.plugin.config.file_execution_timeout + 0.1)
    await until(lambda: env.engine.aborted, what="abort")
    pull.set()
    await exchange.finish()
    assert len(exchange.starts) == 1, "an HTTP status after headers"
    assert stream.closed


# @spec ING-FILE-064
async def test_slow_send_after_inference_is_not_an_execution_timeout(
    files_env,
):
    budget_over = 1500  # above file_execution_timeout's default of 1200 s
    env = await files_env(riva_frontend={"response_send_timeout": budget_over})
    exchange = env.request(block_send=0).start()
    await asyncio.wait_for(exchange.send_blocked.wait(), 2)
    await env.clock.advance(env.plugin.config.file_execution_timeout + 10)
    assert env.engine.aborted == []
    exchange.unblock_send()
    await exchange.finish()
    assert exchange.status == 200, exchange.body


# @spec ING-FILE-035, ING-FILE-064
async def test_execution_timeout_before_headers_is_504_in_core_shape(
    files_env,
):
    env = await files_env()
    block = Block()
    env.service.script(block)
    exchange = env.request().start()
    await asyncio.wait_for(block.entered.wait(), 2)
    await env.clock.advance(env.plugin.config.file_execution_timeout + 0.1)
    await until(lambda: env.engine.aborted, what="abort")
    block.release()
    await exchange.finish()
    assert_core_error(exchange, 504, "request_timeout", param=None)


# @spec ING-FILE-035
async def test_failure_after_headers_ends_the_stream_without_a_status(
    files_env,
):
    env = await files_env()
    stream = TrackedStream(
        [sse({"n": 0}), sse({"n": 1})],
        engine=env.engine,
        raise_at={1: RuntimeError("generator failed")},
    )
    env.service.script(Return(stream))
    exchange = env.request([("stream", "true")]).start()
    await exchange.finish()
    assert len(exchange.starts) == 1
    assert exchange.status == 200
    assert exchange.response_headers["content-type"].startswith(
        "text/event-stream"
    )
    # The error is serializable, so the stream ends as core's generator ends
    # a failed stream: core's streaming error event
    # (``create_streaming_error_response``: core's error shape as one
    # ``data:`` event), then ``data: [DONE]``; the body is completed, not cut.
    assert exchange.completed
    events = [e for e in exchange.body.split(b"\n\n") if e]
    assert events[0] == sse({"n": 0}).strip().encode()
    assert events[1].startswith(b"data: "), events
    error = json.loads(events[1][len(b"data: ") :])["error"]
    assert set(error) == {"message", "type", "param", "code"}, error
    assert events[2:] == [b"data: [DONE]"], events


# ------------------------------------------------------- terminal cause


# @spec ING-FILE-067
async def test_disconnect_before_any_cause_wins_over_a_later_failure(
    files_env,
):
    env = await files_env()
    block = Block(then=Raise(RuntimeError("core failed late")))
    env.service.script(block)
    exchange = env.request().start()
    await asyncio.wait_for(block.entered.wait(), 2)
    exchange.disconnect()
    await until(lambda: env.engine.aborted, what="abort")
    block.release()
    await exchange.finish()
    assert exchange.status is None
    assert exchange.error is None


# @spec ING-FILE-067
async def test_disconnect_after_deadline_suppresses_the_response(files_env):
    env = await files_env()
    block = Block()
    env.service.script(block)
    exchange = env.request().start()
    await asyncio.wait_for(block.entered.wait(), 2)
    await env.clock.advance(env.plugin.config.file_execution_timeout + 0.1)
    await until(lambda: env.engine.aborted, what="abort")
    exchange.disconnect()
    block.release()
    await exchange.finish()
    assert exchange.status is None


# @spec ING-FILE-067
async def test_abort_failure_does_not_change_the_selected_response(
    files_env,
):
    env = await files_env()
    block = Block()
    env.service.script(block)

    async def failing_abort(request_id):
        env.engine.aborted.append(request_id)
        raise RuntimeError("abort failed")

    exchange = env.request().start()
    await asyncio.wait_for(block.entered.wait(), 2)
    env.engine.abort = failing_abort
    await env.clock.advance(env.plugin.config.file_execution_timeout + 0.1)
    await until(lambda: env.engine.aborted, what="abort attempted")
    block.release()
    await exchange.finish()
    assert_core_error(exchange, 504, "request_timeout")


# ---------------------------------------- cleanup bound and continuation


async def _handover(env, *, load_tracking=False):
    guarded = GuardedCall(env.service)
    block = Block()
    env.service.script(block)
    exchange = env.request().start()
    await asyncio.wait_for(block.entered.wait(), 2)
    exchange.disconnect()
    await until(lambda: env.engine.aborted, what="abort")
    await env.clock.advance(env.plugin.config.session_cleanup_timeout + 0.1)
    await exchange.finish()
    return guarded, block, exchange


# @spec ING-FILE-070, ING-FILE-071, ING-FILE-072
async def test_cleanup_bound_hands_over_to_a_registered_continuation(
    files_env,
):
    env = await files_env(load_tracking=True)
    guarded, block, exchange = await _handover(env)
    assert exchange.done, "the handler did not return at the cleanup bound"
    records = list(env.plugin.registry.records.values())
    assert len(records) == 1, "the record was dropped before settlement"
    assert records[0].continuation is not None
    assert env.plugin.file_gate.in_flight == 1
    assert env.plugin.ingress.file_count == 1
    assert env.state.server_load_metrics == 1

    retries = len(env.engine.aborted)
    await env.clock.advance(ABORT_PERIOD)
    await until(
        lambda: len(env.engine.aborted) > retries, what="continued retries"
    )

    settled = len(env.engine.aborted)
    block.release()
    await until(lambda: not env.plugin.registry.records, what="unregistered")
    assert len(env.engine.aborted) > settled, "no final abort"
    assert env.plugin.file_gate.in_flight == 0
    assert env.plugin.ingress.file_count == 0
    assert env.state.server_load_metrics == 0
    assert guarded.cancelled == 0


# @spec ING-FILE-071, ING-FILE-029
async def test_stop_cancels_the_continuation_and_its_children(files_env):
    env = await files_env()
    guarded, block, _ = await _handover(env)
    assert env.plugin.registry.records
    await env.served.stop(timeout=5)  # core's ``stop_endpoint_plugins``
    await until(lambda: guarded.cancelled == 1, what="core call cancelled")
    assert not any(
        not r.task.done() for r in env.plugin.registry.records.values()
    )


# @spec ING-FILE-070
async def test_handler_answers_at_the_cleanup_bound_without_settling(
    files_env,
):
    env = await files_env()
    block = Block()
    env.service.script(block)
    exchange = env.request().start()
    await asyncio.wait_for(block.entered.wait(), 2)
    await env.clock.advance(env.plugin.config.file_execution_timeout + 0.1)
    await until(lambda: env.engine.aborted, what="abort")
    await env.clock.advance(env.plugin.config.session_cleanup_timeout + 0.1)
    await until(lambda: exchange.status is not None, what="answer")
    assert_core_error(exchange, 504, "request_timeout")
    assert env.plugin.file_gate.in_flight == 1, "expiry taken as settlement"
    block.release()
    await until(lambda: env.plugin.file_gate.in_flight == 0, what="release")


# ------------------------------------------------------- load accounting


# @spec ING-FILE-072
async def test_server_load_counts_one_per_route_request(files_env):
    env = await files_env(load_tracking=True)
    block = Block()
    env.service.script(block)
    exchange = env.request().start()
    await asyncio.wait_for(block.entered.wait(), 2)
    assert env.state.server_load_metrics == 1
    block.release()
    await exchange.finish()
    assert exchange.status == 200
    await until(lambda: env.state.server_load_metrics == 0, what="decrement")


# @spec ING-FILE-072
async def test_server_load_is_decremented_once_after_a_failed_send(
    files_env,
):
    env = await files_env(load_tracking=True)
    exchange = env.request(fail_send=0).start()
    await exchange.finish()
    await until(lambda: env.plugin.file_gate.in_flight == 0, what="release")
    assert env.state.server_load_metrics == 0
    await env.clock.advance(1)
    assert env.state.server_load_metrics == 0


# @spec ING-FILE-072
async def test_server_load_is_untouched_when_tracking_is_off(files_env):
    env = await files_env(load_tracking=False)
    block = Block()
    env.service.script(block)
    exchange = env.request().start()
    await asyncio.wait_for(block.entered.wait(), 2)
    assert env.state.server_load_metrics == 0
    block.release()
    await exchange.finish()
    assert env.state.server_load_metrics == 0


# ------------------------------------------------------- core's response


async def _reference(behaviour, fields=(), file=b"RIFF-not-really-audio"):
    """The same request through core's own route (no plugin)."""
    from support.app import build_app, init_state
    from support.fake_engine import FakeEngineClient

    service = FakeTranscriptionServing()
    service.script(behaviour)
    app = build_app([], supported_tasks=("transcription",))
    await init_state(app, FakeEngineClient(), transcription=service)
    body = multipart_body(fields, file=file)
    exchange = HttpExchange(
        app,
        [("content-type", content_type()), ("content-length", str(len(body)))],
        chunked(body),
    )
    await exchange.finish()
    return exchange, service


def _same(behaviour_factory):
    return behaviour_factory(), behaviour_factory()


# @spec ING-FILE-066
@pytest.mark.parametrize(
    "make",
    [
        lambda: Return(transcription_response("hello world", seconds=3)),
        lambda: Return(error_response("bad language", 400, param="language")),
        lambda: Return(error_response("too big", 413)),
        lambda: Raise(ValueError("file exceeds max_model_len: 9 > 8")),
        lambda: Raise(RuntimeError("unexpected")),
    ],
    ids=["json", "error-400", "error-413", "raised-value", "raised-other"],
)
async def test_route_returns_core_response_as_core_route_does(files_env, make):
    env = await files_env()
    env.service.script(make())
    exchange = await env.request().finish()
    expected, reference_service = await _reference(make())
    assert exchange.status == expected.status
    assert exchange.json() == expected.json()
    assert env.calls[0].audio_data == reference_service.calls[0].audio_data


# @spec ING-FILE-066
async def test_route_returns_core_event_stream_as_core_route_does(
    files_env,
):
    from support.fake_transcription import event_stream

    events = [sse({"n": 0}), sse({"n": 1}), "data: [DONE]\n\n"]
    env = await files_env()
    env.service.script(Return(event_stream(*events)))
    exchange = await env.request([("stream", "true")]).finish()
    expected, _ = await _reference(
        Return(event_stream(*events)), fields=[("stream", "true")]
    )
    assert exchange.status == expected.status == 200
    assert exchange.response_headers["content-type"].startswith(
        "text/event-stream"
    )
    assert exchange.body == expected.body
