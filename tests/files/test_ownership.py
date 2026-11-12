"""Request ID, abort ownership, terminal cause, bounds and accounting."""

from __future__ import annotations

import asyncio
import inspect
import io
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
    CORE_ROUTE,
    HttpExchange,
    MiB,
    assert_core_error,
    chunked,
    content_type,
    multipart_body,
    until,
)

from .helpers import (
    GuardedCall,
    SlowWatcherCancel,
    TrackedStream,
    is_uuid,
    start_drain,
    start_stop,
)

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


# @spec ING-FILE-026, ING-FILE-084
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


# @spec ING-FILE-068, ING-FILE-077, ING-FILE-089, ING-FILE-007
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


# @spec ING-FILE-068, ING-FILE-069, ING-FILE-077
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


# @spec ING-FILE-007, ING-FILE-091
async def test_disconnect_during_complete_send_suppresses_only(files_env):
    env = await files_env()
    exchange = env.request(block_send=0).start()
    await asyncio.wait_for(exchange.send_blocked.wait(), 2)
    exchange.disconnect()
    await settle_and_unblock(exchange)
    await exchange.finish()
    await until(lambda: env.plugin.file_gate.in_flight == 0, what="release")
    assert env.engine.aborted == []


# @spec ING-FILE-091, ING-FILE-064, ING-ADM-016
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


# @spec ING-FILE-091, ING-FILE-068
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


# @spec ING-FILE-089
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


# @spec ING-FILE-089
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


# @spec ING-FILE-089
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


# @spec ING-FILE-089, ING-ERR-056, ING-FILE-064
@pytest.mark.parametrize(
    "then",
    [
        lambda: Raise(RuntimeError("core failed at the deadline")),
        lambda: Return(transcription_response("done at the deadline")),
    ],
    ids=["failure", "success"],
)
async def test_deadline_due_with_core_outcome_wins_at_the_call_boundary(
    files_env, then
):
    env = await files_env()
    block = Block(then=then())
    env.service.script(block)
    exchange = env.request().start()
    await asyncio.wait_for(block.entered.wait(), 2)
    await env.clock.settle()
    # The execution deadline started at the permit; no time has passed.
    deadline = env.clock.monotonic() + env.plugin.config.file_execution_timeout
    # Core's outcome and the deadline's expiry reach the owner together:
    # the expired deadline is arbitrated before the outcome is consumed.
    block.release()
    env.clock.set(deadline)
    await exchange.finish()
    assert_core_error(exchange, 504, "request_timeout")


# @spec ING-FILE-089, ING-ERR-056, ING-FILE-064, ING-FILE-035
@pytest.mark.parametrize("fails", [True, False], ids=["failure", "event"])
async def test_deadline_due_with_a_pull_wins_at_the_pull_boundary(
    files_env, fails
):
    env = await files_env()
    pull = asyncio.Event()
    stream = TrackedStream(
        [sse({"n": 0}), sse({"n": 1}), sse({"n": 2})],
        engine=env.engine,
        blocks={1: pull},
        raise_at={1: RuntimeError("pull failed at the deadline")}
        if fails
        else None,
    )
    env.service.script(Return(stream))
    exchange = env.request([("stream", "true")]).start()
    await until(lambda: stream.pulling == 1, what="second pull")
    await env.clock.settle()
    deadline = env.clock.monotonic() + env.plugin.config.file_execution_timeout
    # The pull's outcome (an event or a failure) and the deadline's expiry
    # reach the owner together.
    pull.set()
    env.clock.set(deadline)
    await exchange.finish()
    assert len(exchange.starts) == 1 and exchange.status == 200
    events = [e for e in exchange.body.split(b"\n\n") if e]
    assert events[0] == sse({"n": 0}).strip().encode()
    assert events[1].startswith(b"data: "), events
    error = json.loads(events[1][len(b"data: ") :]).get("error")
    assert error is not None, f"the pull's outcome was consumed: {events}"
    assert error["code"] == 504, error
    assert error["message"].startswith("request_timeout:"), error
    assert events[2:] == [b"data: [DONE]"], events
    await until(lambda: env.plugin.file_gate.in_flight == 0, what="release")
    assert stream.closed


async def _failing_abort(env):
    """Make every engine abort record its ID and then raise."""

    async def failing_abort(request_id):
        env.engine.aborted.append(request_id)
        raise RuntimeError("abort failed")

    env.engine.abort = failing_abort


# @spec ING-FILE-089
@pytest.mark.parametrize(
    ("first", "later", "status", "code"),
    [
        ("deadline", "shutdown", 504, "request_timeout"),
        ("deadline", "abort-failure", 504, "request_timeout"),
        ("deadline", "disconnect", None, None),
        ("disconnect", "shutdown", None, None),
        ("disconnect", "abort-failure", None, None),
        ("shutdown", "deadline", 503, "service_unavailable"),
    ],
    ids=[
        "deadline-then-shutdown",
        "deadline-then-abort-failure",
        "deadline-then-disconnect",
        "disconnect-then-shutdown",
        "disconnect-then-abort-failure",
        "shutdown-then-deadline",
    ],
)
async def test_terminal_cause_selection_rules(
    files_env, first, later, status, code
):
    """One terminal cause: the first decides, a later event changes nothing.

    A disconnect before any cause wins and produces no response; otherwise
    the first of shutdown, a deadline or a failure decides the response; a
    later disconnect suppresses the response and keeps the cause; a later
    abort failure or the plugin's shutdown changes nothing.
    """
    env = await files_env()
    guarded = GuardedCall(env.service)
    block = Block(then=Return(transcription_response("late")))
    env.service.script(block)
    exchange = env.request().start()
    await asyncio.wait_for(block.entered.wait(), 2)
    ended: asyncio.Future | None = None

    async def happen(event: str) -> None:
        nonlocal ended
        if event == "deadline":
            timeout = env.plugin.config.file_execution_timeout
            await env.clock.advance(timeout + 0.1)
        elif event == "disconnect":
            exchange.disconnect()
            await env.clock.settle()
        elif event == "shutdown":
            ended = start_drain(env)
            await env.clock.settle()
        elif event == "abort-failure":
            await _failing_abort(env)
            await env.clock.advance(ABORT_PERIOD)

    await happen(first)
    await until(lambda: env.engine.aborted, what="the first cause aborts")
    await happen(later)
    block.release()
    if ended is not None:
        await asyncio.wait_for(ended, 5)
    await exchange.finish()
    assert exchange.error is None, exchange.error
    if status is None:
        assert exchange.status is None, exchange.body
        assert exchange.messages == []
    else:
        assert len(exchange.starts) == 1
        assert_core_error(exchange, status, code)
    await until(lambda: env.plugin.file_gate.in_flight == 0, what="release")
    assert guarded.cancelled == 0


# @spec ING-FILE-082, ING-FILE-086
@pytest.mark.parametrize(
    ("selected", "status", "code"),
    [
        ("none", 503, "service_unavailable"),
        ("deadline", 504, "request_timeout"),
        ("disconnect", None, None),
    ],
)
async def test_plugin_cancel_answers_selected_cause_and_hands_cleanup_to_continuation(  # noqa: E501
    files_env, selected, status, code
):
    """The plugin's cancellation answers at once and hands cleanup over.

    Core's call is still running when ``drain``'s settlement cancels the
    owner: the handler answers at once with the cause already selected (or
    ``service_unavailable`` when none was), sends nothing after a
    disconnect, and its cleanup continuation keeps the record, permit,
    file-owner slot and load count until core's call has returned.
    """
    env = await files_env(load_tracking=True)
    guarded = GuardedCall(env.service)
    block = Block(then=Return(transcription_response("late")))
    env.service.script(block)
    exchange = env.request().start()
    await asyncio.wait_for(block.entered.wait(), 2)
    if selected == "deadline":
        await env.clock.advance(env.plugin.config.file_execution_timeout + 0.1)
        await until(lambda: env.engine.aborted, what="abort at the deadline")
    elif selected == "disconnect":
        exchange.disconnect()
        await until(lambda: env.engine.aborted, what="abort at the disconnect")
    await env.clock.settle()
    assert not exchange.done, "the handler answered before its cleanup bound"

    drain = start_drain(env)
    # The handler answers at once, while core's call still runs.
    await until(lambda: exchange.done, what="the handler answered at once")
    assert not block.released.is_set()
    assert exchange.error is None, exchange.error
    if status is None:
        assert exchange.status is None, exchange.body
        assert exchange.messages == []
    else:
        assert_core_error(exchange, status, code)
    records = list(env.plugin.registry.records.values())
    assert len(records) == 1, "the record was dropped before settlement"
    assert records[0].continuation is not None
    assert not records[0].continuation.done()
    assert env.plugin.file_gate.in_flight == 1
    assert env.plugin.ingress.file_count == 1
    assert env.state.server_load_metrics == 1
    await env.clock.settle()
    assert not drain.done(), "the settlement did not await the continuation"

    before = len(env.engine.aborted)
    block.release()
    await asyncio.wait_for(drain, 5)
    assert len(env.engine.aborted) > before, "no final abort"
    assert guarded.cancelled == 0, "the owner cancelled core's call"
    assert not env.plugin.registry.records
    assert env.plugin.file_gate.in_flight == 0
    assert env.plugin.ingress.file_count == 0
    assert env.state.server_load_metrics == 0


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


# @spec ING-FILE-078, ING-FILE-079, ING-FILE-090
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


# @spec ING-FILE-078
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


# @spec ING-FILE-078, ING-FILE-079, ING-FILE-080, ING-FILE-087
async def test_cleanup_holds_accounting_until_the_watcher_settles(files_env):
    env = await files_env()
    env.service.script(Raise(RuntimeError("core failed")))
    exchange = env.request()
    watcher = SlowWatcherCancel(exchange)
    exchange.start()
    await asyncio.wait_for(watcher.cancelled.wait(), 2)
    await env.clock.settle()
    # The watcher's cancellation has not settled: nothing is released.
    (record,) = env.plugin.registry.records.values()
    assert any(not child.done() for child in record.children)
    assert env.plugin.file_gate.in_flight == 1, "released before the watcher"
    assert env.plugin.ingress.file_count == 1

    # At the cleanup bound the handler answers and hands the rest over.
    await env.clock.advance(env.plugin.config.session_cleanup_timeout + 0.1)
    await exchange.finish()
    assert exchange.status is not None
    assert list(env.plugin.registry.records.values()) == [record]
    assert record.continuation is not None
    assert not record.continuation.done()
    assert env.plugin.file_gate.in_flight == 1
    assert env.plugin.ingress.file_count == 1

    watcher.release()
    await until(lambda: not env.plugin.registry.records, what="unregistered")
    assert env.plugin.file_gate.in_flight == 0
    assert env.plugin.ingress.file_count == 0


# @spec ING-FILE-078, ING-FILE-079, ING-EP-093
async def test_watcher_before_the_permit_is_settled_or_stopped(files_env):
    env = await files_env(riva_frontend={"max_concurrent_files": 1})
    block = Block()
    env.service.script(block)
    first = env.request().start()
    await asyncio.wait_for(block.entered.wait(), 2)
    second = env.request()
    watcher = SlowWatcherCancel(second)
    second.start()
    await until(lambda: env.plugin.file_gate.waiting == 1, what="waiting")
    await env.clock.advance(env.plugin.config.file_gate_wait_timeout + 0.1)
    await asyncio.wait_for(watcher.cancelled.wait(), 2)
    await env.clock.settle()
    # Refused at the gate, before the permit: the record and the
    # file-owner slot stay until the watcher has settled.
    assert len(env.plugin.registry.records) == 2
    assert env.plugin.ingress.file_count == 2, "released before the watcher"

    await env.clock.advance(env.plugin.config.session_cleanup_timeout + 0.1)
    await second.finish()
    assert_core_error(second, 503, "busy")
    assert len(env.plugin.registry.records) == 2
    assert env.plugin.ingress.file_count == 2

    # ``stop``'s settlement cancels the owners and awaits every
    # continuation with the children it holds, cooperatively: the second
    # request's watcher and the first request's core call.
    ended = start_stop(env)
    await env.clock.settle()
    watcher.release()
    block.release()
    await asyncio.wait_for(ended, 5)
    assert not env.plugin.registry.records
    assert env.plugin.ingress.file_count == 0
    assert env.plugin.file_gate.in_flight == 0
    await first.finish()


# @spec ING-FILE-077, ING-FILE-069
@pytest.mark.parametrize("shutdown", ["drain", "stop"])
@pytest.mark.parametrize("phase", ["call", "pull"])
async def test_owner_never_cancels_core_call_or_pull_continuation_inherits_them(
    files_env, phase, shutdown
):
    """Neither the owner nor the settlement cancels core's call or a pull.

    The handler hands its cleanup over at the cleanup bound; the
    continuation inherits the outstanding call (or pull), keeps aborting
    while it awaits it through the settlement of ``drain`` or ``stop``,
    then sends the final awaited abort and only then closes the generator,
    beginning no new pull.
    """
    env = await files_env()
    guarded = GuardedCall(env.service)
    stream: TrackedStream | None = None
    if phase == "call":
        block = Block(then=Return(transcription_response("late")))
        env.service.script(block)
        exchange = env.request().start()
        await asyncio.wait_for(block.entered.wait(), 2)
        finish_outstanding = block.release
    else:
        pull = asyncio.Event()
        stream = TrackedStream(
            [sse({"n": 0}), sse({"n": 1}), sse({"n": 2})],
            engine=env.engine,
            blocks={1: pull},
        )
        env.service.script(Return(stream))
        exchange = env.request([("stream", "true")]).start()
        await until(lambda: stream.pulling == 1, what="second pull")
        finish_outstanding = pull.set
    exchange.disconnect()
    await until(lambda: env.engine.aborted, what="abort")
    await env.clock.advance(env.plugin.config.session_cleanup_timeout + 0.1)
    await exchange.finish()
    (record,) = env.plugin.registry.records.values()
    assert record.continuation is not None
    assert not record.continuation.done()
    pulls = stream.pulls if stream is not None else 0

    ended = start_drain(env) if shutdown == "drain" else start_stop(env)
    await env.clock.settle()
    assert not ended.done(), "the settlement did not await the continuation"
    retries = len(env.engine.aborted)
    await env.clock.advance(ABORT_PERIOD)
    await until(
        lambda: len(env.engine.aborted) > retries,
        what="the continuation keeps aborting during the settlement",
    )
    assert guarded.cancelled == 0, "core's call was cancelled"
    if stream is not None:
        assert stream.pull_cancelled == 0, "the outstanding pull was cancelled"
        assert not stream.closed, "closed while a pull was running"

    before = len(env.engine.aborted)
    finish_outstanding()
    await asyncio.wait_for(ended, 5)
    assert len(env.engine.aborted) > before, "no final abort"
    assert set(env.engine.aborted) == {env.calls[0].request_id}
    assert guarded.cancelled == 0
    if stream is not None:
        assert stream.closed
        assert stream.aborts_at_close is not None
        assert stream.aborts_at_close > before, "closed before the final abort"
        assert stream.pulls == pulls, "a new pull began after the cause"
        assert stream.max_active == 1
    else:
        assert guarded.returned == 1
    assert not env.plugin.registry.records
    assert env.plugin.file_gate.in_flight == 0


# @spec ING-FILE-078
@pytest.mark.parametrize(
    "context", ["running", "core-draining", "drain-started"]
)
async def test_cleanup_deadline_fixed_at_cause_selection(files_env, context):
    """The cleanup deadline starts at the cause and nothing moves it.

    The cause (the execution deadline) is selected long after the request
    began, so a deadline that started earlier would expire earlier. Core's
    ``state.draining`` with no ``drain`` call leaves it unclipped
    (``session_cleanup_timeout``), and a ``drain`` that begins after the
    cause neither restarts nor extends it.
    """
    env = await files_env()
    block = Block()
    env.service.script(block)
    exchange = env.request().start()
    await asyncio.wait_for(block.entered.wait(), 2)
    cleanup = env.plugin.config.session_cleanup_timeout
    await env.clock.advance(env.plugin.config.file_execution_timeout + 0.1)
    await until(lambda: env.engine.aborted, what="abort at the deadline")
    selected = env.clock.monotonic()
    drain: asyncio.Future | None = None
    await env.clock.advance(cleanup / 2)
    if context == "core-draining":
        env.state.draining = True
    elif context == "drain-started":
        drain = start_drain(env, timeout=10 * cleanup)
    await env.clock.settle()
    await env.clock.advance(selected + cleanup - 0.5 - env.clock.monotonic())
    assert exchange.status is None, "the cleanup deadline expired early"
    await env.clock.advance(1.0)
    await until(lambda: exchange.status is not None, what="answer at the bound")
    assert_core_error(exchange, 504, "request_timeout")
    # The expiry is not taken to mean the children settled.
    assert env.plugin.file_gate.in_flight == 1
    block.release()
    await until(lambda: env.plugin.file_gate.in_flight == 0, what="release")
    await exchange.finish()
    if drain is not None:
        await env.clock.advance(20 * cleanup)
        await asyncio.wait_for(drain, 5)


# @spec ING-FILE-078
@pytest.mark.parametrize("context", ["running", "graceful", "drain-after"])
async def test_owner_cleanup_deadline_is_clipped_to_d_at_selection_only(
    files_env, context
):
    """The owner's cleanup deadline, read where the route cannot hide it.

    A ``Recognize`` owner runs on the plugin (its ``lifecycle``, the source
    of the drain state) but under a registry of its own, so no settle-point
    or settlement cancellation reaches it and only its cleanup deadline ends
    its wait: ``run`` returns when that deadline expires (the handler
    answers, ING-FILE-078) while core's call still runs. The cause is the
    execution deadline, selected at ``t0 + 4`` with ``session_cleanup_timeout``
    30 s:

    * ``running``: no drain; the deadline is ``t0 + 34``;
    * ``graceful``: a ``drain(20)`` call began at ``t0 + 1`` (``D = t0 +
      16``, graceful work until its settle point), so the deadline is the
      smaller of 30 s and the time left to ``D``: ``D`` itself;
    * ``drain-after``: core's ``state.draining`` with no ``drain`` call at
      the selection (unclipped, ``t0 + 34``); a ``drain(20)`` begun at
      ``t0 + 5`` with ``D = t0 + 20`` neither clips nor restarts it.
    """
    from support.app import drain_plugin
    from support.audio import pcm16_sine, wav_pcm16

    from vllm_riva_frontend import files
    from vllm_riva_frontend.admission import OwnerKind
    from vllm_riva_frontend.lifecycle import OwnerRegistry

    env = await files_env(
        riva_frontend={
            "session_cleanup_timeout": 30.0,
            "file_execution_timeout": 4.0,
        }
    )
    plugin, clock = env.plugin, env.clock
    block = Block(then=Return(transcription_response("late")))
    env.service.script(block)
    registry = OwnerRegistry()
    wav = wav_pcm16(pcm16_sine(0.5))

    async def run() -> object:
        record = registry.register(OwnerKind.recognize, asyncio.current_task())
        owner = files.FileRequestOwner(
            service=env.service,
            engine_client=env.engine,
            config=plugin.config,
            clock=clock,
            registry=registry,
            record=record,
            catalog=files._catalog(plugin),
            gate=files.FileGate(1, clock=clock),
            lifecycle=plugin,
            state=env.state,
        )
        owners.append(owner)
        request = files._recognize_request(wav, "en-US", env.service)
        return await owner.run(wav, request, files.build_raw_request())

    owners: list[object] = []
    t0 = clock.monotonic()
    task = asyncio.ensure_future(run())
    drain: asyncio.Future | None = None
    try:
        await asyncio.wait_for(block.entered.wait(), 2)
        if context == "graceful":
            await clock.advance(1.0)
            drain = asyncio.ensure_future(drain_plugin(plugin, env.state, 20))
            expected = t0 + 1 + 20 - 5  # D
        else:
            expected = t0 + 4 + 30
            if context == "drain-after":
                env.state.draining = True
        await clock.advance(t0 + 4.1 - clock.monotonic())
        await until(lambda: env.engine.aborted, what="the deadline's abort")
        (owner,) = owners
        assert owner.cause is not None and owner.cause.kind == "deadline"
        selected = owner.cause
        if context == "drain-after":
            await clock.advance(t0 + 5 - clock.monotonic())
            drain = asyncio.ensure_future(drain_plugin(plugin, env.state, 20))
        await clock.advance(expected - 0.5 - clock.monotonic())
        assert not task.done(), "the cleanup deadline expired early"
        await clock.advance(1.0)
        await until(lambda: task.done(), what="the answer at the deadline")
        # The expiry is not taken to mean core's call settled.
        assert not block.released.is_set()
    finally:
        block.release()
        await clock.settle()
        if not task.done():
            await asyncio.wait({task}, timeout=5)
        if drain is not None:
            for _ in range(60):
                if drain.done():
                    break
                await clock.advance(1.0)
            await asyncio.wait({drain}, timeout=5)
    (owner,) = owners
    # The owner returned normally (no exception, not cancelled) with the
    # deadline cause it selected, which nothing later replaced; ``Recognize``
    # projects that cause, and core's result is never taken.
    assert not task.cancelled(), "the owner ended cancelled"
    assert task.exception() is None, task.exception()
    assert task.result() is None, task.result()
    assert owner.cause is selected, owner.cause
    continuation = owner.continuation
    assert continuation is not None
    await until(lambda: continuation.done(), what="the continuation's end")
    # The cleanup continuation also ended normally.
    assert not continuation.cancelled(), "the continuation ended cancelled"
    assert continuation.exception() is None, continuation.exception()
    assert continuation.result() is None
    assert owner.cause is selected, owner.cause


# @spec ING-FILE-079, ING-FILE-090
@pytest.mark.parametrize("shutdown", ["drain", "stop"])
async def test_continuation_owns_cleanup_and_settlement_awaits_it(
    files_env, shutdown
):
    env = await files_env(load_tracking=True)
    guarded, block, exchange = await _handover(env)
    assert exchange.done
    (record,) = env.plugin.registry.records.values()
    assert record.continuation is not None
    assert not record.continuation.done()

    ended = start_drain(env) if shutdown == "drain" else start_stop(env)
    await env.clock.settle()
    assert not ended.done(), "the settlement did not await the continuation"
    # The continuation still holds the record and every count.
    assert list(env.plugin.registry.records.values()) == [record]
    assert env.plugin.file_gate.in_flight == 1
    assert env.plugin.ingress.file_count == 1
    assert env.state.server_load_metrics == 1
    retries = len(env.engine.aborted)
    await env.clock.advance(ABORT_PERIOD)
    await until(
        lambda: len(env.engine.aborted) > retries, what="continued retries"
    )

    before = len(env.engine.aborted)
    block.release()
    await asyncio.wait_for(ended, 5)
    assert record.continuation.done()
    assert len(env.engine.aborted) > before, "no final abort"
    assert guarded.cancelled == 0
    assert guarded.returned == 1
    assert not env.plugin.registry.records
    assert env.plugin.file_gate.in_flight == 0
    assert env.plugin.ingress.file_count == 0
    assert env.state.server_load_metrics == 0


# ------------------------------------------------------- load accounting


class _ForbiddenCalls:
    """A profiler that records every run of core's ``load_aware_call``.

    Both the decorator itself and the ``wrapper`` it returns are matched by
    code object, so any use (decorating at run time, or calling anything it
    decorated, core's own route function included) is seen.
    """

    def __init__(self) -> None:
        import types

        from vllm.entrypoints.serve.utils import api_utils

        decorator = api_utils.load_aware_call.__code__
        inner = [
            const
            for const in decorator.co_consts
            if isinstance(const, types.CodeType) and const.co_name == "wrapper"
        ]
        assert inner, "core's load_aware_call has no wrapper"
        self._codes = {decorator, *inner}
        self.hits: list[str] = []

    def __enter__(self) -> _ForbiddenCalls:
        import sys

        self._previous = sys.getprofile()

        def profile(frame, event, arg):
            if event == "call" and frame.f_code in self._codes:
                self.hits.append(frame.f_code.co_name)
            if self._previous is not None:
                self._previous(frame, event, arg)

        sys.setprofile(profile)
        return self

    def __exit__(self, *exc: object) -> None:
        import sys

        sys.setprofile(self._previous)


# @spec ING-FILE-090
async def test_file_route_does_not_use_core_load_aware_call(files_env):
    """A forbidden-call spy: the frontend's route never runs core's wrapper.

    A control first decorates and calls a function with core's
    ``load_aware_call`` under the spy, showing it sees both; the frontend's
    file route, with load tracking on, must then run neither.
    """
    from types import SimpleNamespace

    from vllm.entrypoints.serve.utils import api_utils

    env = await files_env(load_tracking=True)

    async def handler(*args, **kwargs):
        return None

    raw = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    with _ForbiddenCalls() as control:
        await api_utils.load_aware_call(handler)(None, raw)
    assert control.hits == ["load_aware_call", "wrapper"], control.hits

    env.service.script(Return(transcription_response("frontend route")))
    with _ForbiddenCalls() as spy:
        exchange = await env.request().finish()
    assert exchange.status == 200
    assert spy.hits == [], "the frontend's route used core's load_aware_call"
    await until(lambda: env.state.server_load_metrics == 0, what="decrement")


# @spec ING-FILE-090
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


# @spec ING-FILE-090
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


# @spec ING-FILE-090
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


# @spec ING-FILE-090, ING-FILE-087
async def test_server_load_counted_once_after_settlement_and_reraise(
    files_env,
):
    env = await files_env(load_tracking=True)
    env.service.script(Raise(RuntimeError("core failed")))
    exchange = env.request()
    watcher = SlowWatcherCancel(exchange)
    exchange.start()
    await asyncio.wait_for(watcher.cancelled.wait(), 2)
    await env.clock.settle()
    assert env.state.server_load_metrics == 1, "decremented before settlement"
    await env.clock.advance(env.plugin.config.session_cleanup_timeout + 0.1)
    await exchange.finish()
    # Core's exception handlers answered the re-raised exception ...
    assert exchange.status == 500, exchange.body
    # ... and the count stays until the frontend's children have settled.
    assert env.state.server_load_metrics == 1
    watcher.release()
    await until(lambda: not env.plugin.registry.records, what="unregistered")
    assert env.state.server_load_metrics == 0
    await env.clock.advance(1)
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
        path=CORE_ROUTE,
    )
    await exchange.finish()
    return exchange, service


def _same(behaviour_factory):
    return behaviour_factory(), behaviour_factory()


# @spec ING-FILE-066, ING-FILE-087, ING-FILE-081
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


# @spec ING-FILE-087
async def test_reraised_exception_left_to_core_handlers_continuation_keeps_accounting(  # noqa: E501
    files_env,
):
    env = await files_env(load_tracking=True)
    env.service.script(Raise(RuntimeError("unexpected")))
    exchange = env.request()
    watcher = SlowWatcherCancel(exchange)
    exchange.start()
    await asyncio.wait_for(watcher.cancelled.wait(), 2)
    await env.clock.advance(env.plugin.config.session_cleanup_timeout + 0.1)
    await exchange.finish()
    # Core's application exception handlers delivered the response, exactly
    # as for core's own route; the frontend sent nothing of its own.
    expected, _ = await _reference(Raise(RuntimeError("unexpected")))
    assert exchange.status == expected.status
    assert exchange.json() == expected.json()
    assert len(exchange.starts) == 1
    # The continuation keeps the record, permit, file-owner slot and load
    # count until the frontend's children have settled.
    (record,) = env.plugin.registry.records.values()
    assert record.continuation is not None
    assert env.plugin.file_gate.in_flight == 1
    assert env.plugin.ingress.file_count == 1
    assert env.state.server_load_metrics == 1
    watcher.release()
    await until(lambda: not env.plugin.registry.records, what="unregistered")
    assert env.plugin.file_gate.in_flight == 0
    assert env.plugin.ingress.file_count == 0
    assert env.state.server_load_metrics == 0


# @spec ING-FILE-087, ING-FILE-091
async def test_returned_error_response_is_delivered_under_the_send_bound(
    files_env,
):
    env = await files_env()
    env.service.script(
        Return(error_response("bad language", 400, param="language"))
    )
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
    assert not exchange.completed, "a send started after its bound"


# ------------------------------------------------------------ send bound


# @spec ING-FILE-091
@pytest.mark.parametrize("shutdown", ["drain", "stop"])
async def test_response_send_bound_and_terminal_attempt(files_env, shutdown):
    """Once the bound counts as passed, the response gets one attempt.

    Core's service completed normally and the response's first send is
    stalled within its bound. When ``drain`` skips its graceful steps, or
    after the plugin is torn down, every send bound counts as passed: the
    stalled send is the single terminal attempt and is cut at its
    suspension, no other send starts, nothing is aborted (the service had
    completed) and the permit is released after settlement.
    """
    env = await files_env()
    exchange = env.request(block_send=0).start()
    await asyncio.wait_for(exchange.send_blocked.wait(), 2)
    assert env.plugin.file_gate.in_flight == 1
    ended = start_drain(env) if shutdown == "drain" else start_stop(env)
    await until(
        ended.done,
        timeout=5,
        what="the settlement ended without waiting for the stalled send",
    )
    assert env.engine.aborted == []
    assert not env.plugin.registry.records
    assert env.plugin.file_gate.in_flight == 0
    exchange.unblock_send()
    await exchange.finish()
    assert exchange.messages == [], "a send was made after the attempt"


# ------------------------------------------------------- core's contract


def _core_serving():
    from vllm.entrypoints.speech_to_text.transcription.serving import (
        OpenAIServingTranscription,
    )

    return OpenAIServingTranscription


# @spec ING-FILE-084
def test_core_state_construction_builds_the_service_attribute():
    from argparse import Namespace
    from types import SimpleNamespace
    from unittest import mock

    from starlette.datastructures import State
    from vllm.entrypoints.speech_to_text import factories
    from vllm.entrypoints.speech_to_text.transcription import api_router

    state = State()
    state.openai_serving_models = object()
    with (
        mock.patch(
            "vllm.entrypoints.speech_to_text.transcription.serving."
            "OpenAIServingTranscription"
        ) as serving,
        mock.patch(
            "vllm.entrypoints.speech_to_text.translation.serving."
            "OpenAIServingTranslation"
        ),
    ):
        factories.init_speech_to_text_state(
            object(),
            state,
            Namespace(enable_force_include_usage=False),
            None,
            ("transcription",),
        )
    assert state.openai_serving_transcription is serving.return_value
    # Core's own route reads the service from the same attribute.
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    assert api_router.transcription(request) is serving.return_value


# @spec ING-FILE-084
def test_core_create_transcription_signature():
    method = _core_serving().create_transcription
    assert inspect.iscoroutinefunction(method)
    parameters = inspect.signature(method).parameters
    assert list(parameters) == ["self", "audio_data", "request", "raw_request"]
    assert parameters["raw_request"].default is None


# @spec ING-FILE-084
def test_core_request_model_carries_the_fields_the_frontend_sets():
    from fastapi import UploadFile
    from vllm.entrypoints.speech_to_text.transcription.protocol import (
        TranscriptionRequest,
    )

    fields = TranscriptionRequest.model_fields
    for name in (
        "file",
        "model",
        "language",
        "response_format",
        "stream",
        "vllm_xargs",
        "use_beam_search",
        "max_completion_tokens",
    ):
        assert name in fields, name
    request = TranscriptionRequest(
        file=UploadFile(io.BytesIO(b"RIFF"), filename="a.wav"),
        language="en",
        response_format="json",
        vllm_xargs={"locale": "en-US"},
    )
    assert request.vllm_xargs == {"locale": "en-US"}
    assert request.stream is False
    assert request.use_beam_search is False
    assert request.max_completion_tokens is None


# @spec ING-FILE-084
async def test_core_read_upload_with_limit_reads_an_in_memory_upload():
    from fastapi import UploadFile
    from vllm.entrypoints.speech_to_text.base.utils import (
        read_upload_with_limit,
    )
    from vllm.exceptions import VLLMValidationError

    assert list(inspect.signature(read_upload_with_limit).parameters) == [
        "file",
        "max_size_mb",
    ]
    data = b"a" * 4096
    upload = UploadFile(io.BytesIO(data), filename="a.wav")
    assert await read_upload_with_limit(upload) == data
    over = UploadFile(io.BytesIO(b"a" * (MiB + 1)), filename="a.wav")
    with pytest.raises(VLLMValidationError):
        await read_upload_with_limit(over, max_size_mb=1)


# @spec ING-FILE-084
def test_core_derives_the_engine_request_id_from_x_request_id():
    from starlette.requests import Request
    from vllm.entrypoints.serve.engine.serving import BaseServing
    from vllm.entrypoints.speech_to_text.base.serving import (
        SpeechToTextBaseServing,
    )

    request = Request({"type": "http", "headers": [(b"x-request-id", b"abc")]})
    assert BaseServing._base_request_id(request) == "abc"
    bare = Request({"type": "http", "headers": []})
    assert BaseServing._base_request_id(bare) != "abc"
    # The engine request ID is ``<task type>-<base ID>``, with the
    # transcription service's task type ``transcribe``.
    source = inspect.getsource(SpeechToTextBaseServing._create_speech_to_text)
    assert 'f"{self.task_type}-{self._base_request_id(raw_request)}"' in source
    init = inspect.getsource(_core_serving().__init__)
    assert 'task_type="transcribe"' in init


# @spec ING-FILE-084
def test_core_engine_client_abort_signature():
    from vllm.engine.protocol import EngineClient

    assert inspect.iscoroutinefunction(EngineClient.abort)
    parameters = list(inspect.signature(EngineClient.abort).parameters)
    assert parameters == ["self", "request_id"]
