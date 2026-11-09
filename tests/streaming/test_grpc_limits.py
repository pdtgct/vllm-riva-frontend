"""Message limits: the framework cap and the unary size and duration caps."""

from __future__ import annotations

import asyncio
import os
import sys

import grpc
import pytest
from riva.client.proto import riva_asr_pb2 as rasr
from riva.client.proto import riva_asr_pb2_grpc as rasr_grpc
from support.audio import (
    g711_bytes,
    pcm16_bytes,
    pcm16_sine,
    wav_pcm16,
)
from support.fake_session import Block, Raise
from support.grpc_harness import FakeServicerContext, RpcAbort, aio_channel
from support.riva import LINEAR_PCM, MULAW, UNSPECIFIED, recognition_config

from .conftest import (
    riva_options,
    saturate_gate,
    serving,
    small_unary_options,
)

LIMIT_OPTIONS = (
    "unary_max_encoded_audio_bytes",
    "unary_max_decoded_duration_seconds",
    "max_riff_header_bytes",
    "grpc_config_envelope_max_bytes",
)


def _failure_text(exc: BaseException) -> str:
    failures = getattr(exc, "failures", None) or []
    return " ".join([str(exc), *map(str, failures)])


@pytest.mark.parametrize("option", LIMIT_OPTIONS)
@pytest.mark.parametrize("value", [0, -1, float("inf")])
# @spec ING-GRPC-031
async def test_init_state_requires_finite_positive_limits(clock, option, value):
    with pytest.raises(Exception) as caught:
        async with serving(
            clock=clock, options=riva_options(**{option: value}), start=False
        ):
            pass
    assert option in _failure_text(caught.value)


# @spec ING-GRPC-031
async def test_init_state_requires_the_receive_cap_relation(clock):
    options = small_unary_options(grpc_receive_max_bytes=8192 + 1024 + 1)
    with pytest.raises(Exception) as caught:
        async with serving(clock=clock, options=options, start=False):
            pass
    assert "grpc_receive_max_bytes" in _failure_text(caught.value)


# @spec ING-GRPC-031
async def test_default_receive_cap_is_audio_limit_plus_envelope(clock):
    async with serving(clock=clock, start=False) as served:
        config = served.plugin.config
        assert config.grpc_receive_max_bytes == (
            config.unary_max_encoded_audio_bytes
            + config.grpc_config_envelope_max_bytes
        )


# @spec ING-GRPC-031
async def test_framework_cap_applies_to_both_rpcs_on_the_listener(clock):
    options = small_unary_options()
    address = options["grpc_bind"]
    too_big = b"\x00" * (8192 + 1024 + 16)
    async with (
        serving(clock=clock, options=options),
        aio_channel(address) as channel,
    ):
        stub = rasr_grpc.RivaSpeechRecognitionStub(channel)
        with pytest.raises(grpc.aio.AioRpcError) as unary:
            await stub.Recognize(
                rasr.RecognizeRequest(
                    config=recognition_config(), audio=too_big
                ),
                timeout=5.0,
            )
        assert unary.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED

        async def requests():
            yield rasr.StreamingRecognizeRequest(
                streaming_config=rasr.StreamingRecognitionConfig(
                    config=recognition_config()
                )
            )
            yield rasr.StreamingRecognizeRequest(audio_content=too_big)

        with pytest.raises(grpc.aio.AioRpcError) as streaming:
            async for _ in stub.StreamingRecognize(requests(), timeout=5.0):
                pass
        assert streaming.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED


# ------------------------------------------------------------ unary


async def _recognize_with_full_gate(served, config, audio):
    permits = await saturate_gate(served)
    context = FakeServicerContext()
    try:
        with pytest.raises(RpcAbort) as caught:
            await asyncio.wait_for(
                served.servicer.Recognize(
                    rasr.RecognizeRequest(config=config, audio=audio), context
                ),
                2.0,
            )
    finally:
        for permit in permits:
            permit.release()
    return caught.value


# @spec ING-GRPC-032
async def test_unary_encoded_overflow_is_request_too_large_before_a_permit(
    clock,
):
    async with serving(clock=clock, options=small_unary_options()) as served:
        abort = await _recognize_with_full_gate(
            served, recognition_config(), b"\x00\x00" * 4100
        )
        assert abort.code == grpc.StatusCode.RESOURCE_EXHAUSTED
        assert abort.details.startswith("request_too_large:"), abort.details
        assert served.transcription.calls == []


def _duration_options():
    return riva_options(
        unary_max_encoded_audio_bytes=32_768,
        grpc_config_envelope_max_bytes=1024,
        grpc_receive_max_bytes=32_768 + 1024,
        max_riff_header_bytes=2048,
        unary_max_decoded_duration_seconds=1,
    )


@pytest.mark.parametrize(
    ("config", "audio"),
    [
        (
            recognition_config(encoding=MULAW, sample_rate_hertz=8_000),
            g711_bytes(pcm16_sine(1.5, rate=8_000), "mulaw"),
        ),
        (
            recognition_config(encoding=LINEAR_PCM, sample_rate_hertz=8_000),
            pcm16_bytes(pcm16_sine(1.2, rate=8_000)),
        ),
        (
            recognition_config(encoding=UNSPECIFIED, sample_rate_hertz=0),
            wav_pcm16(pcm16_sine(1.2, rate=8_000), rate=8_000),
        ),
    ],
    ids=["mulaw-8k", "pcm16-8k", "riff-pcm16-8k"],
)
# @spec ING-GRPC-032
async def test_unary_duration_overflow_is_request_too_large_before_a_permit(
    clock, config, audio
):
    async with serving(clock=clock, options=_duration_options()) as served:
        abort = await _recognize_with_full_gate(served, config, audio)
        assert abort.code == grpc.StatusCode.RESOURCE_EXHAUSTED
        assert abort.details.startswith("request_too_large:"), abort.details
        assert served.transcription.calls == []


# @spec ING-GRPC-032
async def test_unary_audio_at_the_duration_limit_is_accepted(clock):
    async with serving(clock=clock, options=_duration_options()) as served:
        audio = g711_bytes(pcm16_sine(1.0, rate=8_000), "mulaw")
        response = await asyncio.wait_for(
            served.servicer.Recognize(
                rasr.RecognizeRequest(
                    config=recognition_config(
                        encoding=MULAW, sample_rate_hertz=8_000
                    ),
                    audio=audio,
                ),
                FakeServicerContext(),
            ),
            2.0,
        )
        assert isinstance(response, rasr.RecognizeResponse)
        assert len(served.transcription.calls) == 1


# ------------------------------------------------- no filesystem writes

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
_RECORDING: list[list[tuple[str, tuple]]] = []


def _audit(event: str, args: tuple) -> None:
    if not _RECORDING:
        return
    if event == "open":
        path, mode, flags = (tuple(args) + (None, None, None))[:3]
        text = str(path)
        if text.endswith((".pyc", ".pyc.tmp")) or "__pycache__" in text:
            return
        writes = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and flags & _WRITE_FLAGS
        )
        if writes:
            _RECORDING[-1].append((event, args))
    elif event in {
        "os.mkdir",
        "os.rename",
        "os.symlink",
        "shutil.copyfile",
        "tempfile.mkstemp",
        "tempfile.mkdtemp",
    }:
        _RECORDING[-1].append((event, args))


_HOOKED = False


def _install_hook() -> None:
    global _HOOKED
    if not _HOOKED:
        sys.addaudithook(_audit)
        _HOOKED = True


class _Writes:
    def __enter__(self) -> list[tuple[str, tuple]]:
        _install_hook()
        _RECORDING.append([])
        return _RECORDING[-1]

    def __exit__(self, *exc: object) -> None:
        _RECORDING.pop()


# @spec ING-GRPC-010
async def test_unary_recognize_writes_nothing_to_the_filesystem(clock):
    import vllm_riva_frontend.files  # noqa: F401  (import before recording)
    import vllm_riva_frontend.frontend  # noqa: F401

    async with serving(clock=clock, options=small_unary_options()) as served:
        good = rasr.RecognizeRequest(
            config=recognition_config(),
            audio=pcm16_bytes(pcm16_sine(0.1)),
        )
        too_big = rasr.RecognizeRequest(
            config=recognition_config(), audio=b"\x00\x00" * 4100
        )
        riff = rasr.RecognizeRequest(
            config=recognition_config(
                encoding=UNSPECIFIED, sample_rate_hertz=0
            ),
            audio=wav_pcm16(pcm16_sine(0.1)),
        )
        with _Writes() as writes:
            # success, both declared and header-inferred
            await served.servicer.Recognize(good, FakeServicerContext())
            await served.servicer.Recognize(riff, FakeServicerContext())
            # rejection
            with pytest.raises(RpcAbort):
                await served.servicer.Recognize(too_big, FakeServicerContext())
            # failure inside core's service
            served.transcription.script(Raise(RuntimeError("engine fault")))
            with pytest.raises(RpcAbort):
                await served.servicer.Recognize(good, FakeServicerContext())
            # cancellation while core works
            gate = Block()
            served.transcription.script(gate)
            context = FakeServicerContext()
            task = asyncio.create_task(served.servicer.Recognize(good, context))
            await asyncio.wait_for(gate.entered.wait(), 2.0)
            context.cancel()
            task.cancel()
            await asyncio.wait({task}, timeout=5.0)
            # the RPC deadline while core works
            served.transcription.script(Block())
            context = FakeServicerContext(
                time_remaining=1.0, clock=served.clock
            )
            task = asyncio.create_task(served.servicer.Recognize(good, context))
            await served.clock.settle(50)
            await served.clock.advance(2.0)
            await asyncio.wait({task}, timeout=5.0)
            if not task.done():
                task.cancel()
                await asyncio.wait({task}, timeout=5.0)
            await served.clock.settle(50)
        assert writes == [], writes
