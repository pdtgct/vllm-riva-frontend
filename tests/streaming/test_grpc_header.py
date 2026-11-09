"""RIFF/WAVE header inference under ``ENCODING_UNSPECIFIED``."""

from __future__ import annotations

import asyncio
import io
import wave

import grpc
import numpy as np
from riva.client.proto import riva_asr_pb2 as rasr
from support.audio import (
    DATA_LENGTH_UNKNOWN,
    float_from_pcm16,
    g711_bytes,
    pcm16_bytes,
    pcm16_ramp,
    wav_bytes,
    wav_pcm16,
)
from support.riva import (
    LINEAR_PCM,
    UNSPECIFIED,
    audio_request,
    config_request,
    recognition_config,
    recognize,
    recognize_request,
)

from .conftest import (
    lease_names,
    opened,
    saturate_gate,
    serving,
    small_unary_options,
    start_stream,
    until,
)

SAMPLES = pcm16_ramp(1600, start=-800, step=1)


def _unspecified(**fields) -> rasr.StreamingRecognizeRequest:
    return config_request(
        recognition_config(encoding=UNSPECIFIED, sample_rate_hertz=0, **fields)
    )


async def _stream_bytes(served, chunks, *, config=None):
    index = len(served.factory.leases)
    call = start_stream(served, [config or _unspecified()])
    lease = await opened(served, call, index=index)
    for chunk in chunks:
        call.push(audio_request(chunk))
    call.half_close()
    await call.wait()
    return call, lease


# @spec ING-GRPC-029
async def test_header_is_resolved_stripped_and_audio_fed_after_open(served):
    data = wav_pcm16(SAMPLES)
    call, lease = await _stream_bytes(served, [data])
    assert call.status == grpc.StatusCode.OK, call.details
    np.testing.assert_array_equal(lease.fed_samples, float_from_pcm16(SAMPLES))
    names = served.factory.names()
    assert names.index("open") < names.index("feed")


# @spec ING-GRPC-029
async def test_no_feed_until_the_split_header_resolves(served):
    data = wav_pcm16(SAMPLES)
    call = start_stream(served, [_unspecified()])
    lease = await opened(served, call)
    call.push(audio_request(data[:20]))
    await until(
        lambda: len(call.requests.delivered) == 2, what="partial header"
    )
    await served.clock.settle(50)
    assert "feed" not in lease_names(served)
    call.push(audio_request(data[20:]))
    call.half_close()
    await call.wait()
    assert call.status == grpc.StatusCode.OK, call.details
    np.testing.assert_array_equal(lease.fed_samples, float_from_pcm16(SAMPLES))


# @spec ING-GRPC-029
async def test_g711_header_resolves_an_accept_matrix_cell(served):
    payload = g711_bytes(pcm16_ramp(800, start=-400, step=1), "mulaw")
    call, lease = await _stream_bytes(
        served, [wav_bytes(payload, encoding="mulaw", rate=8_000)]
    )
    assert call.status == grpc.StatusCode.OK, call.details
    assert sum(lease.accepted) > 0


# @spec ING-GRPC-029
# @spec ING-GRPC-018
async def test_off_matrix_header_is_unsupported_format_after_open(served):
    data = wav_pcm16(SAMPLES, rate=44_100)
    call, _ = await _stream_bytes(served, [data])
    assert call.status == grpc.StatusCode.INVALID_ARGUMENT
    assert call.details.startswith("unsupported_format:"), call.details
    assert "sample_rate_hertz" in call.details
    names = lease_names(served)
    assert "feed" not in names
    assert names[-2:] == ["abort", "release"]


# @spec ING-GRPC-029
async def test_stereo_header_is_unsupported_format_naming_channels(served):
    data = wav_bytes(pcm16_bytes(np.repeat(SAMPLES, 2)), channels=2)
    call, _ = await _stream_bytes(served, [data])
    assert call.status == grpc.StatusCode.INVALID_ARGUMENT
    assert call.details.startswith("unsupported_format:"), call.details
    assert "audio_channel_count" in call.details


# @spec ING-GRPC-018
# @spec ING-GRPC-029
async def test_declared_rate_is_a_constraint_the_header_must_match(served):
    data = wav_pcm16(SAMPLES, rate=8_000)
    call, _ = await _stream_bytes(
        served,
        [data],
        config=config_request(
            recognition_config(encoding=UNSPECIFIED, sample_rate_hertz=16_000)
        ),
    )
    assert call.status == grpc.StatusCode.INVALID_ARGUMENT
    assert call.details.startswith("unsupported_format:"), call.details
    assert "sample_rate_hertz" in call.details
    assert "feed" not in lease_names(served)


# @spec ING-GRPC-018
async def test_matching_declared_rate_and_channels_are_accepted(served):
    data = wav_pcm16(SAMPLES, rate=16_000)
    call, lease = await _stream_bytes(
        served,
        [data],
        config=config_request(
            recognition_config(
                encoding=UNSPECIFIED,
                sample_rate_hertz=16_000,
                audio_channel_count=1,
            )
        ),
    )
    assert call.status == grpc.StatusCode.OK, call.details
    np.testing.assert_array_equal(lease.fed_samples, float_from_pcm16(SAMPLES))


# @spec ING-GRPC-029
async def test_no_complete_header_within_the_bound_is_unsupported_format(clock):
    async with serving(clock=clock, options=small_unary_options()) as served:
        filler = (b"LIST", b"\x00" * 3000)  # pushes ``data`` past 2048 bytes
        data = wav_bytes(pcm16_bytes(SAMPLES), extra_chunks=[filler])
        call, _ = await _stream_bytes(
            served, [data[i : i + 512] for i in range(0, len(data), 512)]
        )
        assert call.status == grpc.StatusCode.INVALID_ARGUMENT
        assert call.details.startswith("unsupported_format:"), call.details
        assert "feed" not in lease_names(served)


# @spec ING-GRPC-029
async def test_non_riff_bytes_are_unsupported_format(served):
    call, _ = await _stream_bytes(served, [b"OggS" + b"\x00" * 200])
    assert call.status == grpc.StatusCode.INVALID_ARGUMENT
    assert call.details.startswith("unsupported_format:"), call.details
    assert "feed" not in lease_names(served)


# @spec ING-GRPC-029
async def test_a_declared_encoding_is_never_overridden_by_a_header(served):
    data = wav_pcm16(SAMPLES)
    call, lease = await _stream_bytes(
        served,
        [data],
        config=config_request(recognition_config(encoding=LINEAR_PCM)),
    )
    assert call.status == grpc.StatusCode.OK, call.details
    expected = float_from_pcm16(np.frombuffer(data, dtype="<i2"))
    np.testing.assert_array_equal(lease.fed_samples, expected)


# @spec ING-GRPC-021
async def test_sentinel_data_lengths_mean_audio_to_the_end_of_the_stream(
    served,
):
    for length in (0, DATA_LENGTH_UNKNOWN):
        data = wav_bytes(pcm16_bytes(SAMPLES), data_length=length)
        call, lease = await _stream_bytes(served, [data[:100], data[100:]])
        assert call.status == grpc.StatusCode.OK, call.details
        np.testing.assert_array_equal(
            lease.fed_samples, float_from_pcm16(SAMPLES)
        )


# @spec ING-GRPC-021
async def test_explicit_data_length_truncates_and_later_chunks_are_ignored(
    served,
):
    kept = SAMPLES[:320]
    trailer = b"LIST" + (8).to_bytes(4, "little") + b"INFOabcd"
    data = wav_bytes(pcm16_bytes(kept), data_length=640) + trailer
    data += pcm16_bytes(SAMPLES[320:])
    call, lease = await _stream_bytes(served, [data])
    assert call.status == grpc.StatusCode.OK, call.details
    np.testing.assert_array_equal(lease.fed_samples, float_from_pcm16(kept))


# ------------------------------------------------------------ unary


async def _recognize(served, audio):
    config = recognition_config(encoding=UNSPECIFIED, sample_rate_hertz=0)
    outcome = await asyncio.wait_for(
        recognize(served.servicer, recognize_request(audio, config)), 2.0
    )
    assert outcome.error is None, outcome.error
    return outcome.response, outcome.abort


# @spec ING-GRPC-029
async def test_unary_header_is_validated_before_a_file_gate_permit(served):
    permits = await saturate_gate(served)
    try:
        _, abort = await _recognize(served, wav_pcm16(SAMPLES, rate=44_100))
    finally:
        for permit in permits:
            permit.release()
    assert abort is not None
    assert abort.code == grpc.StatusCode.INVALID_ARGUMENT
    assert abort.details.startswith("unsupported_format:"), abort.details
    assert served.transcription.calls == []


# @spec ING-GRPC-029
# @spec ING-GRPC-021
async def test_unary_header_is_stripped_and_explicit_length_truncates(served):
    data = wav_bytes(pcm16_bytes(SAMPLES), data_length=640)
    response, abort = await _recognize(served, data)
    assert abort is None, abort
    assert isinstance(response, rasr.RecognizeResponse)
    (call,) = served.transcription.calls
    with wave.open(io.BytesIO(call.audio_data)) as handed:
        assert handed.getframerate() == 16_000
        assert handed.getnchannels() == 1
        assert handed.getnframes() == 320
        frames = np.frombuffer(handed.readframes(320), dtype="<i2")
    np.testing.assert_array_equal(frames, SAMPLES[:320])
