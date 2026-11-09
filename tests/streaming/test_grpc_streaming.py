"""``StreamingRecognize`` wire contract: order, projection, final, bounds."""

from __future__ import annotations

import asyncio
import inspect

import grpc
import numpy as np
from riva.client.proto import riva_asr_pb2 as rasr
from riva.client.proto import riva_asr_pb2_grpc as rasr_grpc
from support.app import drain_plugin
from support.audio import float_from_pcm16, pcm16_bytes, pcm16_ramp, pcm16_sine
from support.fake_session import (
    Block,
    FakeSessionFactory,
    Raise,
    Return,
    failed,
)
from support.riva import audio_request, config_request, recognition_config

from .conftest import (
    PRE_SUBMIT,
    lease_names,
    opened,
    opens,
    riva_options,
    serving,
    start_stream,
    until,
)

AUDIO = pcm16_bytes(pcm16_sine(0.02))  # 320 samples, below the pre-submit bound

RPC_NAMES = (
    "StreamingRecognize",
    "Recognize",
    "GetRivaSpeechRecognitionConfig",
)


def _tail_after_feeds(names: list[str]) -> list[str]:
    last_feed = max(
        (i for i, name in enumerate(names) if name == "feed"), default=-1
    )
    return names[last_feed + 1 :]


# @spec ING-GRPC-001
async def test_servicer_implements_the_three_rpcs_of_the_public_stubs(served):
    servicer = served.servicer
    base = rasr_grpc.RivaSpeechRecognitionServicer
    for name in RPC_NAMES:
        method = getattr(type(servicer), name, None)
        assert method is not None, name
        assert method is not getattr(base, name), f"{name} not implemented"
    server = grpc.aio.server()
    rasr_grpc.add_RivaSpeechRecognitionServicer_to_server(servicer, server)
    import vllm_riva_frontend.grpc as frontend_grpc

    source = inspect.getsource(frontend_grpc)
    assert "riva.client.proto" in source


# @spec ING-GRPC-001
# @spec ING-GRPC-026
async def test_streaming_responses_are_public_stub_messages(served):
    call = start_stream(served, [config_request()])
    lease = await opened(served, call)
    lease.script("feed", Return(["hello"]))
    call.push(audio_request(AUDIO))
    call.half_close()
    await call.wait()
    assert call.responses
    for response in call.responses:
        assert type(response) is rasr.StreamingRecognizeResponse


# @spec ING-GRPC-002
async def test_open_completes_before_the_request_iterator_advances(served):
    gate = Block()
    served.factory.script("open", gate)
    call = start_stream(
        served, [config_request(), audio_request(AUDIO), audio_request(AUDIO)]
    )
    await asyncio.wait_for(gate.entered.wait(), 2.0)
    await asyncio.sleep(0.01)
    assert call.requests.pulls == 1
    assert len(call.requests.delivered) == 1
    gate.release()
    await until(lambda: len(served.factory.leases) == 1, what="lease")
    call.half_close()
    await call.wait()
    assert call.status == grpc.StatusCode.OK, call.details


# @spec ING-GRPC-002
async def test_session_opens_at_560ms_with_the_validated_locale(served):
    call = start_stream(
        served,
        [config_request(recognition_config(language_code="de-DE"))],
    )
    await opened(served, call)
    call.half_close()
    await call.wait()
    assert opens(served) == [{"cadence": "560ms", "locale": "de-DE"}]


# @spec ING-GRPC-002
async def test_changed_hypotheses_are_interims_and_duplicates_are_skipped(
    served,
):
    call = start_stream(served, [config_request(interim_results=True)])
    lease = await opened(served, call)
    lease.script(
        "feed",
        Return(["hel"]),
        Return(["hel"]),
        Return([]),
        Return(["hel", "hello"]),
    )
    for _ in range(4):
        call.push(audio_request(AUDIO))
    await until(lambda: len(lease.accepted) >= 4, what="four feeds")
    call.half_close()
    await call.wait()
    assert call.interims == ["hel", "hello"]
    for response in call.interim_responses:
        assert response.results[0].is_final is False


# @spec ING-GRPC-002
async def test_interim_results_false_sends_only_the_final(served):
    call = start_stream(served, [config_request(interim_results=False)])
    lease = await opened(served, call)
    lease.script("feed", Return(["hel"]), Return(["hello"]))
    call.push(audio_request(AUDIO))
    call.push(audio_request(AUDIO))
    await until(lambda: len(lease.accepted) >= 2, what="two feeds")
    call.half_close()
    await call.wait()
    assert call.interims == []
    assert len(call.finals) == 1


# @spec ING-GRPC-002
async def test_half_close_flushes_finishes_releases_then_sends_one_final(
    served,
):
    call = start_stream(served, [config_request()])
    lease = await opened(served, call)
    gate = Block()
    lease.script("release", gate)
    call.push(audio_request(AUDIO))
    await until(lambda: lease.accepted, what="feed")
    call.half_close()
    await asyncio.wait_for(gate.entered.wait(), 2.0)
    await asyncio.sleep(0.01)
    assert call.finals == [], "final sent before release succeeded"
    assert not call.done
    gate.release()
    await call.wait()
    assert call.status == grpc.StatusCode.OK, call.details
    assert call.finals == ["hello world"]
    assert call.responses[-1] is call.final_responses[0]
    assert _tail_after_feeds(lease_names(served)) == [
        "flush",
        "finish",
        "release",
    ]


# @spec ING-GRPC-002
async def test_later_streaming_config_is_protocol_order_with_abnormal_cleanup(
    served,
):
    call = start_stream(served, [config_request()])
    await opened(served, call)
    call.push(audio_request(AUDIO))
    call.push(config_request())
    await call.wait()
    assert call.status == grpc.StatusCode.FAILED_PRECONDITION
    assert call.details.startswith("protocol_order:"), call.details
    names = lease_names(served)
    assert "finish" not in names
    assert names[-2:] == ["abort", "release"]
    assert call.finals == []


# @spec ING-GRPC-002
async def test_audio_first_is_protocol_order_and_opens_nothing(served):
    call = start_stream(served, [audio_request(AUDIO)])
    await call.wait()
    assert call.status == grpc.StatusCode.FAILED_PRECONDITION
    assert call.details.startswith("protocol_order:"), call.details
    assert opens(served) == []


# @spec ING-GRPC-026
async def test_each_response_has_one_result_one_alternative_and_no_extra_fields(
    served,
):
    call = start_stream(served, [config_request()])
    lease = await opened(served, call)
    lease.script("feed", Return(["partial"]))
    call.push(audio_request(AUDIO))
    await until(lambda: lease.accepted, what="feed")
    call.half_close()
    await call.wait()
    assert len(call.interims) == 1 and len(call.finals) == 1
    for response in call.responses:
        assert len(response.results) == 1
        result = response.results[0]
        assert len(result.alternatives) == 1
        set_fields = {fd.name for fd, _ in result.ListFields()}
        assert set_fields <= {"alternatives", "is_final"}, set_fields
        alternative = result.alternatives[0]
        alt_fields = {fd.name for fd, _ in alternative.ListFields()}
        assert alt_fields <= {"transcript"}, alt_fields
        assert len(alternative.words) == 0
        assert result.stability == 0.0
        assert result.channel_tag == 0
    assert call.interims == ["partial"]
    assert call.final_responses[0].results[0].is_final is True


# @spec ING-GRPC-027
async def test_transcript_text_is_relayed_unchanged(clock):
    text = "  Grüße, WORLD -- ünïcode\tend  "
    async with serving(
        clock=clock, factory=FakeSessionFactory(final_text=text)
    ) as served:
        call = start_stream(served, [config_request()])
        lease = await opened(served, call)
        lease.script("feed", Return([" Grüße,"]))
        call.push(audio_request(AUDIO))
        await until(lambda: lease.accepted, what="feed")
        call.half_close()
        await call.wait()
    assert call.interims == [" Grüße,"]
    assert call.finals == [text]


async def _drain_to_settle_point(served):
    """Start ``drain`` with 60 s and move the clock to its settle point.

    ``D`` is ``now + 60 - max(5, 6)``; the settle point is ``D - 5``.
    """
    task = asyncio.create_task(drain_plugin(served.plugin, served.state, 60.0))
    await served.clock.advance(49.5)
    return task


# @spec ING-GRPC-028
# @spec ING-ERR-027
async def test_shutdown_finalization_sends_one_final_then_ok(served):
    call = start_stream(served, [config_request()])
    lease = await opened(served, call)
    call.push(audio_request(AUDIO))
    await until(lambda: lease.accepted, what="feed")
    drain = await _drain_to_settle_point(served)
    await call.wait()
    assert call.status == grpc.StatusCode.OK, call.details
    assert call.finals == ["hello world"]
    assert call.responses[-1] is call.final_responses[0]
    assert "flush" in lease_names(served)
    await served.clock.advance(10)
    await asyncio.wait_for(drain, 5.0)


# @spec ING-GRPC-015
async def test_cancellation_during_flush_emits_no_fabricated_final(served):
    call = start_stream(served, [config_request()])
    lease = await opened(served, call)
    gate = Block()
    lease.script("flush", gate)
    lease.script("release", Raise(failed("abnormal_end")))
    call.push(audio_request(AUDIO))
    await until(lambda: lease.accepted, what="feed")
    call.half_close()
    await asyncio.wait_for(gate.entered.wait(), 2.0)
    call.context.cancel()
    assert call.task is not None
    call.task.cancel()
    await asyncio.wait({call.task}, timeout=5.0)
    await served.clock.settle(50)
    assert call.finals == []
    assert all(
        abort.code == grpc.StatusCode.CANCELLED for abort in call.context.aborts
    ), call.context.aborts


# @spec ING-GRPC-015
async def test_rpc_deadline_wins_and_cleanup_failure_keeps_its_status(served):
    call = start_stream(served, [config_request()], time_remaining=5.0)
    lease = await opened(served, call)
    lease.script("flush", Block())
    lease.script("abort", Raise(failed("abnormal_end")))
    lease.script("release", Raise(failed("abnormal_end")))
    call.push(audio_request(AUDIO))
    await until(lambda: lease.accepted, what="feed")
    call.half_close()
    await until(lambda: "flush" in lease_names(served), what="flush")
    await served.clock.advance(6.0)
    await call.wait()
    assert call.status == grpc.StatusCode.DEADLINE_EXCEEDED, call.details
    assert call.finals == []


# @spec ING-GRPC-015
# @spec ING-LIFE-020
async def test_finish_failure_on_the_normal_path_emits_no_final(served):
    call = start_stream(served, [config_request()])
    lease = await opened(served, call)
    lease.script("finish", Raise(failed("abnormal_end")))
    call.push(audio_request(AUDIO))
    await until(lambda: lease.accepted, what="feed")
    call.half_close()
    await call.wait()
    assert call.status == grpc.StatusCode.INTERNAL, call.details
    assert call.details.startswith("internal:"), call.details
    assert call.finals == []


# @spec ING-GRPC-015
# @spec ING-ERR-059
async def test_failed_shutdown_finalization_is_unavailable_with_no_final(
    served,
):
    call = start_stream(served, [config_request()])
    lease = await opened(served, call)
    lease.script("flush", Raise(failed("abnormal_end")))
    call.push(audio_request(AUDIO))
    await until(lambda: lease.accepted, what="feed")
    drain = await _drain_to_settle_point(served)
    await call.wait()
    assert call.status == grpc.StatusCode.UNAVAILABLE, call.details
    assert call.details.startswith("service_unavailable:"), call.details
    assert call.finals == []
    await served.clock.advance(10)
    await asyncio.wait_for(drain, 5.0)


# @spec ING-GRPC-019
async def test_large_message_is_split_into_bounded_feeds_while_reads_pause(
    served,
):
    count = 3 * PRE_SUBMIT + 7
    samples = pcm16_ramp(count, start=-2000, step=1)
    call = start_stream(served, [config_request()])
    lease = await opened(served, call)
    gate = Block()
    lease.script("feed", gate)
    call.push(audio_request(pcm16_bytes(samples)))
    call.push(audio_request(AUDIO))
    await asyncio.wait_for(gate.entered.wait(), 2.0)
    await asyncio.sleep(0.01)
    assert len(call.requests.delivered) == 2, "read ahead while feeding"
    gate.release()
    await until(lambda: sum(lease.accepted) >= count + 320, what="all audio")
    call.half_close()
    await call.wait()
    assert call.status == grpc.StatusCode.OK, call.details
    assert all(n <= PRE_SUBMIT for n in lease.accepted), lease.accepted
    first_message = lease.fed_samples[:count]
    np.testing.assert_array_equal(first_message, float_from_pcm16(samples))


# @spec ING-GRPC-020
async def test_a_legal_large_message_never_projects_buffer_overflow(served):
    samples = pcm16_ramp(10 * PRE_SUBMIT, start=-8000, step=1)
    call = start_stream(served, [config_request()])
    lease = await opened(served, call)
    call.push(audio_request(pcm16_bytes(samples)))
    await until(lambda: sum(lease.accepted) >= samples.size, what="feeds")
    call.half_close()
    await call.wait()
    assert call.status == grpc.StatusCode.OK, call.details
    assert "buffer_overflow" not in call.details
    assert len(call.finals) == 1


# @spec ING-GRPC-033
async def test_stream_message_and_duration_are_not_limited_by_unary_caps(clock):
    options = riva_options(
        unary_max_encoded_audio_bytes=8192,
        grpc_config_envelope_max_bytes=1024,
        grpc_receive_max_bytes=8192 + 1024,
        max_riff_header_bytes=2048,
        unary_max_decoded_duration_seconds=1,
    )
    async with serving(clock=clock, options=options) as served:
        call = start_stream(served, [config_request()])
        lease = await opened(served, call)
        # one message above the unary encoded cap, and 2.5 s in total
        big = pcm16_bytes(pcm16_sine(0.5))  # 16000 bytes
        for _ in range(5):
            call.push(audio_request(big))
        await until(lambda: sum(lease.accepted) >= 5 * 8000, what="feeds")
        call.half_close()
        await call.wait()
    assert call.status == grpc.StatusCode.OK, call.details
    assert len(call.finals) == 1


# @spec ING-GRPC-033
async def test_resource_failure_after_open_ends_with_abnormal_cleanup(served):
    call = start_stream(served, [config_request()])
    lease = await opened(served, call)
    lease.script("feed", Raise(failed("input_capacity")))
    call.push(audio_request(AUDIO))
    await call.wait()
    assert call.status == grpc.StatusCode.RESOURCE_EXHAUSTED, call.details
    assert call.details.startswith("buffer_overflow:"), call.details
    names = lease_names(served)
    assert "finish" not in names
    assert names[-2:] == ["abort", "release"]
    assert call.finals == []


# @spec ING-GRPC-033
# @spec ING-LIFE-043
async def test_max_session_duration_bounds_a_stream_when_set(clock):
    async with serving(
        clock=clock, options=riva_options(max_session_duration=20)
    ) as served:
        call = start_stream(served, [config_request()])
        lease = await opened(served, call)
        for index in range(3):
            call.push(audio_request(AUDIO))
            await until(
                lambda index=index: len(lease.accepted) > index, what="feed"
            )
            await clock.advance(8.0)
        await call.wait()
    assert call.status == grpc.StatusCode.DEADLINE_EXCEEDED, call.details
    assert call.details.startswith("request_timeout:"), call.details
    assert "abort" in lease_names(served)
    assert call.finals == []
