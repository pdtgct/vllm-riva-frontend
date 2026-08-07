"""GPU-free public-proto tests for the direct-lease Riva gRPC adapter."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace

import grpc
import pytest
from grpc_health.v1 import health
from riva.client.proto import riva_asr_pb2 as rasr
from riva.client.proto import riva_audio_pb2 as raud

import vllm_riva_frontend.grpc as grpc_frontend
from vllm_riva_frontend.admission import LoadShedGate
from vllm_riva_frontend.config import FrontendConfig
from vllm_riva_frontend.frontend import FormatError, RiffFormat
from vllm_riva_frontend.grpc import (
    RivaServicer,
    build_servicer,
    register_aio_services,
)
from vllm_riva_frontend.lease import DirectLeaseOwner


def _config() -> FrontendConfig:
    return FrontendConfig(
        grpc_bind="127.0.0.1:50051",
        grpc_receive_max_bytes=1152,
        grpc_config_envelope_max_bytes=128,
        unary_max_encoded_audio_bytes=1024,
        unary_max_decoded_duration_seconds=60.0,
        max_riff_header_bytes=512,
        load_shed_max_sessions=4,
        pre_submit_max_samples=16000,
        preconfiguration_timeout=5.0,
        session_idle_timeout=60.0,
        session_finalization_timeout=30.0,
        session_cleanup_timeout=1.0,
        plugin_shutdown_grace=31.0,
        ws_receive_max_bytes=2048,
        ws_event_envelope_max_bytes=1024,
        http_multipart_envelope_max_bytes=1152,
        http_content_type_max_bytes=128,
        http_request_header_max_bytes=256,
        http_multipart_boundary_max_bytes=64,
        http_multipart_max_parts=4,
        http_multipart_max_header_bytes=128,
        http_text_field_max_bytes=128,
        http_request_timeout=60.0,
        grpc_keepalive_seconds=None,
        max_session_duration=None,
        resampler_identifier="scipy-poly-v1",
    )


class RpcAbort(Exception):
    """The test context's terminal gRPC projection."""

    def __init__(self, code: grpc.StatusCode, details: str) -> None:
        self.code = code
        self.details = details


class FakeContext:
    """Minimal async grpc.aio context with observable aborts."""

    def __init__(
        self, *, cancelled: bool = False, time_remaining: float | None = None
    ) -> None:
        self._cancelled = cancelled
        self._time_remaining = time_remaining

    def cancelled(self) -> bool:
        return self._cancelled

    def time_remaining(self) -> float | None:
        return self._time_remaining

    async def abort(self, code: grpc.StatusCode, details: str) -> None:
        raise RpcAbort(code, details)


class FakeAdmissionLease:
    """One synchronously releasable host-admission owner."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.released = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.events.append("admission:release")


class FakeAdmission:
    """Host admission capability with an explicitly controllable state."""

    def __init__(
        self, open_: bool = True, *, events: list[str] | None = None
    ) -> None:
        self.open_ = open_
        self.events = events if events is not None else []
        self.leases: list[FakeAdmissionLease] = []

    def is_open(self) -> bool:
        return self.open_

    def try_acquire(self) -> FakeAdmissionLease | None:
        self.events.append("admission:try")
        if not self.open_:
            return None
        lease = FakeAdmissionLease(self.events)
        self.leases.append(lease)
        return lease


class RecordingOwnerToken:
    """Lifecycle ownership token whose release order is observable."""

    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.released = False

    async def release(self) -> None:
        self.released = True
        self._events.append("owner:release")


class FakeLease:
    """One fake RFC-1 lease with public transport-neutral operations."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def feed(self, samples: object, *, on_accepted: object) -> list[str]:
        self.calls.append("feed")
        on_accepted(len(samples))
        return ["partial"]

    async def update_locale(self, locale: str) -> object:
        del locale
        return None

    async def flush(self) -> str:
        self.calls.append("flush")
        return "final"

    async def finish(self) -> None:
        self.calls.append("finish")

    async def abort(self) -> None:
        self.calls.append("abort")

    async def release(self) -> None:
        self.calls.append("release")


class BlockingFinalizeLease(FakeLease):
    """Lease whose normal tail never settles unless its owner cancels it."""

    async def flush(self) -> str:
        self.calls.append("flush")
        await asyncio.Event().wait()
        raise AssertionError("cancelled flush must not return")


class InvalidCreditLease(FakeLease):
    """Lease fake that violates exact accepted-credit accounting."""

    async def feed(self, samples: object, *, on_accepted: object) -> list[str]:
        self.calls.append("feed")
        on_accepted(len(samples) - 1)
        return []


class MultiHypothesisLease(FakeLease):
    """Lease fake returning every changed park hypothesis in one feed."""

    async def feed(self, samples: object, *, on_accepted: object) -> list[str]:
        self.calls.append("feed")
        on_accepted(len(samples))
        return ["one", "one", "one two"]


class FailingFeedLease(FakeLease):
    """Lease fake exposing an unexpected provider exception."""

    async def feed(self, samples: object, *, on_accepted: object) -> list[str]:
        del samples, on_accepted
        self.calls.append("feed")
        raise ValueError("provider exploded")


class BlockingFeedLease(FakeLease):
    """Lease fake whose feed never acknowledges receipt credit."""

    async def feed(self, samples: object, *, on_accepted: object) -> list[str]:
        del samples, on_accepted
        self.calls.append("feed")
        await asyncio.Event().wait()
        return []


class FakeFactory:
    def __init__(self) -> None:
        self.lease = FakeLease()
        self.opened: list[tuple[str, str]] = []

    async def open(self, *, cadence: str, locale: str) -> FakeLease:
        self.opened.append((cadence, locale))
        return self.lease


class FakeFrontend:
    def __init__(self, encoding: str, rate: int) -> None:
        self.encoding = encoding
        self.rate = rate

    def push(self, data: bytes) -> list[float]:
        return [0.0] if data else []

    def flush(self) -> list[float]:
        return []


def _recognition() -> rasr.RecognitionConfig:
    return rasr.RecognitionConfig(
        encoding=raud.LINEAR_PCM,
        sample_rate_hertz=16000,
        language_code="en-US",
    )


def _servicer(factory: FakeFactory, **overrides: object) -> RivaServicer:
    config = overrides.pop("config", _config())
    admission = overrides.pop("admission", FakeAdmission())
    frontend_factory = overrides.pop("frontend_factory", FakeFrontend)
    return RivaServicer(
        factory=factory,
        config=config,
        model_name="nemotron",
        locales=frozenset({"auto", "en-US"}),
        frontend_factory=frontend_factory,
        admission=admission,
        **overrides,
    )


async def _requests(
    requests: list[rasr.StreamingRecognizeRequest],
) -> AsyncIterator[rasr.StreamingRecognizeRequest]:
    for request in requests:
        yield request


async def _collect_stream(
    stream: AsyncIterator[rasr.StreamingRecognizeResponse],
) -> list[rasr.StreamingRecognizeResponse]:
    return [response async for response in stream]


# @spec ING-GRPC-001, ING-GRPC-002, ING-CORE-007
def test_streaming_uses_real_config_first_proto_and_one_direct_lease() -> None:
    factory = FakeFactory()
    events: list[str] = []
    admission = FakeAdmission(events=events)
    config = rasr.StreamingRecognitionConfig(
        config=_recognition(), interim_results=True
    )
    first = rasr.StreamingRecognizeRequest(streaming_config=config)
    audio = rasr.StreamingRecognizeRequest(audio_content=b"\x00\x00")

    async def exercise() -> list[rasr.StreamingRecognizeResponse]:
        return [
            response
            async for response in _servicer(
                factory, admission=admission
            ).StreamingRecognize(_requests([first, audio]), FakeContext())
        ]

    responses = asyncio.run(exercise())
    # @spec ING-GRPC-002: streaming opens on the shared 560ms streaming arm.
    assert factory.opened == [("560ms", "en-US")]
    assert [item.results[0].is_final for item in responses] == [False, True]
    assert [
        item.results[0].alternatives[0].transcript for item in responses
    ] == [
        "partial",
        "final",
    ]
    assert factory.lease.calls == ["feed", "flush", "finish", "release"]
    assert events == ["admission:try", "admission:release"]
    assert len(admission.leases) == 1
    assert admission.leases[0].released


# @spec ING-GRPC-002, ING-CORE-005
def test_streaming_emits_each_changed_hypothesis_and_skips_duplicates() -> None:
    factory = FakeFactory()
    factory.lease = MultiHypothesisLease()
    config = rasr.StreamingRecognitionConfig(
        config=_recognition(), interim_results=True
    )
    requests = [
        rasr.StreamingRecognizeRequest(streaming_config=config),
        rasr.StreamingRecognizeRequest(audio_content=b"\x00\x00"),
    ]
    responses = asyncio.run(
        _collect_stream(
            _servicer(factory).StreamingRecognize(
                _requests(requests), FakeContext()
            )
        )
    )
    assert [
        response.results[0].alternatives[0].transcript for response in responses
    ] == ["one", "one two", "final"]


# @spec ING-GRPC-002, ING-LIFE-001
def test_audio_before_config_is_code_first_failed_precondition() -> None:
    request = rasr.StreamingRecognizeRequest(audio_content=b"audio")

    async def exercise() -> None:
        async for _ in _servicer(FakeFactory()).StreamingRecognize(
            _requests([request]), FakeContext()
        ):
            pass

    with pytest.raises(RpcAbort) as error:
        asyncio.run(exercise())
    assert error.value.code is grpc.StatusCode.FAILED_PRECONDITION
    assert error.value.details.startswith("protocol_order:")


# @spec ING-GRPC-005, ING-GRPC-006
def test_model_intrinsic_punctuation_and_verbatim_values_are_accepted() -> None:
    config = _recognition()
    config.enable_automatic_punctuation = True
    config.verbatim_transcripts = True
    stream = rasr.StreamingRecognizeRequest(
        streaming_config=rasr.StreamingRecognitionConfig(config=config)
    )

    async def exercise() -> None:
        async for _ in _servicer(FakeFactory()).StreamingRecognize(
            _requests([stream]), FakeContext()
        ):
            pass

    asyncio.run(exercise())


# @spec ING-VEH-013, ING-GRPC-005
def test_canonical_checkpoint_is_not_an_alias_when_public_name_is_set() -> None:
    factory = FakeFactory()
    config = _recognition()
    config.model = "nvidia/nemotron-asr"
    request = rasr.StreamingRecognizeRequest(
        streaming_config=rasr.StreamingRecognitionConfig(config=config)
    )

    async def exercise() -> None:
        async for _ in _servicer(factory).StreamingRecognize(
            _requests([request]), FakeContext()
        ):
            pass

    with pytest.raises(RpcAbort) as error:
        asyncio.run(exercise())
    assert error.value.code is grpc.StatusCode.INVALID_ARGUMENT
    assert error.value.details.startswith("invalid_config_field:")
    assert "model" in error.value.details
    assert factory.opened == []


# @spec ING-VEH-013, ING-GRPC-005, ING-GRPC-009
def test_accepted_alias_selector_still_opens_the_bound_lease() -> None:
    """An accepted alias must delegate, not vanish before the factory.

    Complements the rejection above: the exact-alias case must not be a
    silent no-op either -- it must reach the one bound session factory,
    the same as an unset (omitted) model does.
    """
    factory = FakeFactory()
    config = _recognition()
    config.model = "nemotron"
    request = rasr.RecognizeRequest(config=config, audio=b"\x00\x00")

    response = asyncio.run(_servicer(factory).Recognize(request, FakeContext()))

    assert factory.opened == [("1120ms", "en-US")]
    assert response.results[0].alternatives[0].transcript == "final"


# @spec ING-GRPC-003, ING-GRPC-009, ING-GRPC-010
def test_unary_real_proto_uses_1120ms_direct_lease_without_file_io() -> None:
    factory = FakeFactory()
    request = rasr.RecognizeRequest(config=_recognition(), audio=b"\x00\x00")
    response = asyncio.run(_servicer(factory).Recognize(request, FakeContext()))
    assert factory.opened == [("1120ms", "en-US")]
    assert response.results[0].alternatives[0].transcript == "final"
    assert factory.lease.calls == ["feed", "flush", "finish", "release"]


# @spec ING-GRPC-005, ING-GRPC-013, ING-ERR-003
def test_invalid_config_and_runtime_rider_are_named_before_open() -> None:
    factory = FakeFactory()
    events: list[str] = []
    gate = LoadShedGate(max_sessions=1)
    token = RecordingOwnerToken(events)

    async def owner_register(kind: str) -> RecordingOwnerToken:
        assert kind == "grpc_recognize"
        assert gate.active == 1
        events.append("owner:register")
        return token

    request = rasr.RecognizeRequest(
        config=rasr.RecognitionConfig(
            encoding=raud.LINEAR_PCM,
            sample_rate_hertz=16000,
            language_code="bad-locale",
        ),
        audio=b"x",
    )
    with pytest.raises(RpcAbort) as error:
        asyncio.run(
            _servicer(
                factory, load_shed=gate, owner_register=owner_register
            ).Recognize(request, FakeContext())
        )
    assert error.value.details.startswith("unknown_locale:")
    assert not factory.opened
    assert token.released
    assert events == ["owner:register", "owner:release"]
    assert gate.active == 0


# @spec ING-GRPC-005, ING-GRPC-006, ING-GRPC-013
def test_config_validation_collects_every_unsupported_field_family() -> None:
    config = rasr.RecognitionConfig(
        encoding=999,
        sample_rate_hertz=44100,
        language_code="bad",
        max_alternatives=2,
        audio_channel_count=2,
        model="other",
        profanity_filter=True,
        enable_word_time_offsets=True,
        enable_separate_recognition_per_channel=True,
    )
    config.speech_contexts.add().phrases.append("boost")
    config.diarization_config.enable_speaker_diarization = True
    config.custom_configuration["custom"] = "value"
    config.endpointing_config.start_history = 1

    rejected = grpc_frontend.validate_recognition_config(
        config,
        model_name="nemotron",
        locales=frozenset({"auto", "en-US"}),
    )

    assert ("unsupported_format", "encoding, sample_rate_hertz") in rejected
    assert ("unknown_locale", "language_code") in rejected
    assert ("invalid_config_field", "max_alternatives") in rejected
    assert ("invalid_config_field", "audio_channel_count") in rejected
    assert ("invalid_config_field", "model") in rejected
    assert ("invalid_config_field", "profanity_filter") in rejected
    assert ("unsupported_capability", "speech_contexts") in rejected
    assert (
        "invalid_config_field",
        "enable_word_time_offsets",
    ) in rejected


# @spec ING-ERR-001, ING-GRPC-008
def test_abort_and_deadline_helpers_require_terminal_context_behavior() -> None:
    servicer = _servicer(FakeFactory())
    with pytest.raises(TypeError, match="must expose abort"):
        asyncio.run(servicer._abort(object(), "internal", "bad"))

    class ReturningContext:
        def abort(self, code: grpc.StatusCode, details: str) -> None:
            del code, details

    with pytest.raises(RuntimeError, match="must terminate"):
        asyncio.run(servicer._abort(ReturningContext(), "internal", "bad"))

    with pytest.raises(RpcAbort) as error:
        asyncio.run(
            servicer._await_until(
                lambda: asyncio.sleep(0),
                deadline=-1.0,
                context=FakeContext(),
                timeout_code="configuration_timeout",
                timeout_fields="config",
            )
        )
    assert error.value.code is grpc.StatusCode.DEADLINE_EXCEEDED


# @spec ING-FE-001, ING-GRPC-009
@pytest.mark.parametrize(
    ("resolved", "recognition", "field"),
    [
        (
            FormatError("unsupported_format", ("encoding",)),
            _recognition(),
            "encoding",
        ),
        (
            RiffFormat("LINEAR_PCM", 8000, 1, 44, 2),
            rasr.RecognitionConfig(sample_rate_hertz=16000),
            "sample_rate_hertz",
        ),
        (
            RiffFormat("LINEAR_PCM", 16000, 2, 44, 2),
            rasr.RecognitionConfig(audio_channel_count=1),
            "audio_channel_count",
        ),
        (
            RiffFormat("MULAW", 16000, 1, 44, 2),
            rasr.RecognitionConfig(),
            "encoding",
        ),
    ],
)
def test_deferred_riff_rejections_name_the_conflicting_field(
    resolved: RiffFormat | FormatError,
    recognition: rasr.RecognitionConfig,
    field: str,
) -> None:
    servicer = _servicer(
        FakeFactory(),
        riff_resolver=lambda data, **kwargs: resolved,
    )
    with pytest.raises(RpcAbort) as error:
        asyncio.run(
            servicer._resolve_riff_or_abort(
                b"RIFF",
                recognition,
                FakeContext(),
            )
        )
    assert field in error.value.details


# @spec ING-GRPC-004, ING-GRPC-006
def test_static_config_rpc_uses_real_proto_and_honest_model_intrinsics() -> (
    None
):
    response = asyncio.run(
        _servicer(FakeFactory()).GetRivaSpeechRecognitionConfig(
            rasr.RivaSpeechRecognitionConfigRequest(), FakeContext()
        )
    )
    entry = response.model_config[0]
    assert entry.model_name == "nemotron"
    assert entry.parameters["supported_encodings"] == "LINEAR_PCM,MULAW,ALAW"
    assert entry.parameters["verbatim_transcripts"] == "model-intrinsic"


# @spec ING-GRPC-008, ING-GRPC-012
def test_unary_bound_and_cancelled_context_never_emit_fabricated_final() -> (
    None
):
    factory = FakeFactory()
    oversized = rasr.RecognizeRequest(config=_recognition(), audio=b"x" * 1025)
    with pytest.raises(RpcAbort) as error:
        asyncio.run(_servicer(factory).Recognize(oversized, FakeContext()))
    assert error.value.code is grpc.StatusCode.RESOURCE_EXHAUSTED
    cancelled = rasr.RecognizeRequest(config=_recognition(), audio=b"x")
    response = asyncio.run(
        _servicer(FakeFactory()).Recognize(
            cancelled, FakeContext(cancelled=True)
        )
    )
    assert not response.results


# @spec ING-FE-001, ING-GRPC-008, ING-GRPC-009
def test_unary_rejects_missing_truncated_and_overduration_audio() -> None:
    unspecified = rasr.RecognitionConfig(language_code="en-US")
    with pytest.raises(RpcAbort) as error:
        asyncio.run(
            _servicer(FakeFactory()).Recognize(
                rasr.RecognizeRequest(config=unspecified, audio=b"not-riff"),
                FakeContext(),
            )
        )
    assert error.value.details.startswith("unsupported_format:")

    truncated_servicer = _servicer(
        FakeFactory(),
        riff_resolver=lambda data, **kwargs: RiffFormat(
            "LINEAR_PCM", 16000, 1, 0, 100
        ),
    )
    with pytest.raises(RpcAbort) as error:
        asyncio.run(
            truncated_servicer.Recognize(
                rasr.RecognizeRequest(config=unspecified, audio=b"RIFF"),
                FakeContext(),
            )
        )
    assert error.value.details.startswith("invalid_audio:")

    short_limit = replace(_config(), unary_max_decoded_duration_seconds=0.00001)
    with pytest.raises(RpcAbort) as error:
        asyncio.run(
            _servicer(FakeFactory(), config=short_limit).Recognize(
                rasr.RecognizeRequest(config=_recognition(), audio=b"\x00\x00"),
                FakeContext(),
            )
        )
    assert error.value.details.startswith("request_too_large:")


# @spec ING-LIFE-001, ING-GRPC-001, ING-VEH-019
def test_closed_host_admission_does_not_consume_stream_or_register_owner() -> (
    None
):
    events: list[str] = []
    factory = FakeFactory()
    gate = LoadShedGate(max_sessions=1)

    async def owner_register(kind: str) -> RecordingOwnerToken:
        events.append(f"owner:register:{kind}")
        return RecordingOwnerToken(events)

    async def unread() -> AsyncIterator[rasr.StreamingRecognizeRequest]:
        events.append("stream:read")
        yield rasr.StreamingRecognizeRequest()

    async def exercise() -> None:
        admission = FakeAdmission(False, events=events)
        async for _ in _servicer(
            factory,
            admission=admission,
            load_shed=gate,
            owner_register=owner_register,
        ).StreamingRecognize(unread(), FakeContext()):
            pass

    with pytest.raises(RpcAbort) as error:
        asyncio.run(exercise())
    assert error.value.code is grpc.StatusCode.UNAVAILABLE
    assert events == ["admission:try"]
    assert gate.active == 0
    assert not factory.opened


# @spec ING-LIFE-001, ING-GRPC-003, ING-VEH-019
def test_closed_admission_does_not_register_unary_owner() -> None:
    factory = FakeFactory()
    events: list[str] = []

    async def owner_register(kind: str) -> RecordingOwnerToken:
        events.append(f"owner:register:{kind}")
        return RecordingOwnerToken(events)

    request = rasr.RecognizeRequest(config=_recognition(), audio=b"\x00\x00")
    admission = FakeAdmission(False, events=events)
    with pytest.raises(RpcAbort) as error:
        asyncio.run(
            _servicer(
                factory,
                admission=admission,
                owner_register=owner_register,
            ).Recognize(request, FakeContext())
        )
    assert error.value.code is grpc.StatusCode.UNAVAILABLE
    assert events == ["admission:try"]
    assert not factory.opened


# @spec ING-VEH-019, ING-GRPC-003, ING-LIFE-010
def test_unary_holds_host_admission_until_terminal_cleanup() -> None:
    events: list[str] = []
    admission = FakeAdmission(events=events)
    request = rasr.RecognizeRequest(
        config=_recognition(),
        audio=b"\x00\x00",
    )

    response = asyncio.run(
        _servicer(FakeFactory(), admission=admission).Recognize(
            request,
            FakeContext(),
        )
    )

    assert response.results[0].alternatives[0].transcript == "final"
    assert events == ["admission:try", "admission:release"]
    assert admission.leases[0].released


# @spec ING-VEH-019, ING-VEH-022
def test_owner_release_failure_cannot_leak_host_admission() -> None:
    events: list[str] = []
    admission = FakeAdmission(events=events)

    class FailingOwner:
        async def release(self) -> None:
            events.append("owner:release")
            raise RuntimeError("owner release failed")

    async def register(kind: str) -> FailingOwner:
        assert kind == "grpc_recognize"
        return FailingOwner()

    with pytest.raises(RuntimeError, match="owner release failed"):
        asyncio.run(
            _servicer(
                FakeFactory(),
                admission=admission,
                owner_register=register,
            ).Recognize(
                rasr.RecognizeRequest(config=_recognition(), audio=b"\x00\x00"),
                FakeContext(),
            )
        )

    assert events == [
        "admission:try",
        "owner:release",
        "admission:release",
    ]
    assert admission.leases[0].released


# @spec ING-LIFE-001, ING-GRPC-001, ING-ADM-006
def test_stream_owner_precedes_first_read_and_releases_with_gate() -> None:
    events: list[str] = []
    gate = LoadShedGate(max_sessions=1)
    token = RecordingOwnerToken(events)

    async def owner_register(kind: str) -> RecordingOwnerToken:
        assert kind == "grpc_streaming_recognize"
        assert gate.active == 1
        events.append("owner:register")
        return token

    async def requests() -> AsyncIterator[rasr.StreamingRecognizeRequest]:
        assert events == ["owner:register"]
        yield rasr.StreamingRecognizeRequest(
            streaming_config=rasr.StreamingRecognitionConfig(
                config=_recognition()
            )
        )
        yield rasr.StreamingRecognizeRequest(audio_content=b"\x00\x00")

    async def exercise() -> None:
        async for _ in _servicer(
            FakeFactory(), load_shed=gate, owner_register=owner_register
        ).StreamingRecognize(requests(), FakeContext()):
            pass

    asyncio.run(exercise())
    assert token.released
    assert events == ["owner:register", "owner:release"]
    assert gate.active == 0


# @spec ING-GRPC-007, ING-FE-001
def test_canonical_unspecified_riff_is_resolved_stripped_and_then_opened() -> (
    None
):
    factory = FakeFactory()
    request = rasr.RecognizeRequest(
        config=rasr.RecognitionConfig(), audio=b"RIFF\x00\x00"
    )

    def resolve(data: bytes, **_: object) -> RiffFormat:
        assert data.startswith(b"RIFF")
        return RiffFormat("LINEAR_PCM", 16000, 1, 4, 2)

    response = asyncio.run(
        _servicer(factory, riff_resolver=resolve).Recognize(
            request, FakeContext()
        )
    )
    assert response.results[0].alternatives[0].transcript == "final"
    assert factory.opened == [("1120ms", "auto")]


# @spec ING-GRPC-007, ING-CORE-007
def test_deferred_riff_feeds_only_after_resolution() -> None:
    factory = FakeFactory()
    calls = 0

    def resolve(data: bytes, **_: object) -> RiffFormat | None:
        nonlocal calls
        calls += 1
        if len(data) < 4:
            return None
        return RiffFormat("LINEAR_PCM", 16000, 1, 4, 2)

    config = rasr.StreamingRecognitionConfig(
        config=rasr.RecognitionConfig(), interim_results=False
    )
    requests = [
        rasr.StreamingRecognizeRequest(streaming_config=config),
        rasr.StreamingRecognizeRequest(audio_content=b"RI"),
        rasr.StreamingRecognizeRequest(audio_content=b"FF\x00\x00"),
    ]

    async def exercise() -> list[rasr.StreamingRecognizeResponse]:
        return [
            item
            async for item in _servicer(
                factory, riff_resolver=resolve
            ).StreamingRecognize(_requests(requests), FakeContext())
        ]

    responses = asyncio.run(exercise())
    # @spec ING-GRPC-002: streaming opens on the shared 560ms streaming arm.
    assert factory.opened == [("560ms", "auto")]
    assert calls == 2
    assert factory.lease.calls == ["feed", "flush", "finish", "release"]
    assert responses[0].results[0].is_final


# @spec ING-FE-005, ING-GRPC-008
def test_invalid_provider_credit_projects_internal_and_cleans_up() -> None:
    factory = FakeFactory()
    factory.lease = InvalidCreditLease()
    request = rasr.RecognizeRequest(config=_recognition(), audio=b"\x00\x00")
    with pytest.raises(RpcAbort, match="internal") as error:
        asyncio.run(_servicer(factory).Recognize(request, FakeContext()))
    assert error.value.code is grpc.StatusCode.INTERNAL
    assert factory.lease.calls == ["feed", "abort", "release"]


# @spec ING-ERR-001, ING-GRPC-012, ING-LIFE-013
def test_unexpected_provider_exception_projects_internal_after_cleanup() -> (
    None
):
    factory = FakeFactory()
    factory.lease = FailingFeedLease()
    request = rasr.RecognizeRequest(config=_recognition(), audio=b"\x00\x00")
    with pytest.raises(RpcAbort, match="internal") as error:
        asyncio.run(_servicer(factory).Recognize(request, FakeContext()))
    assert error.value.code is grpc.StatusCode.INTERNAL
    assert factory.lease.calls == ["feed", "abort", "release"]


# @spec ING-LIFE-003, ING-GRPC-001
def test_hung_stream_feed_is_bounded_by_accepted_audio_idle_timer() -> None:
    factory = FakeFactory()
    factory.lease = BlockingFeedLease()
    stream_config = rasr.StreamingRecognitionConfig(config=_recognition())
    requests = [
        rasr.StreamingRecognizeRequest(streaming_config=stream_config),
        rasr.StreamingRecognizeRequest(audio_content=b"\x00\x00"),
    ]
    config = replace(_config(), session_idle_timeout=0.01)

    async def exercise() -> None:
        async for _ in _servicer(factory, config=config).StreamingRecognize(
            _requests(requests), FakeContext()
        ):
            pass

    with pytest.raises(RpcAbort, match="idle_timeout"):
        asyncio.run(asyncio.wait_for(exercise(), timeout=0.2))
    assert factory.lease.calls == ["feed", "abort", "release"]


# @spec ING-FE-005, ING-GRPC-003
def test_long_unary_audio_and_tail_are_handed_off_in_credit_sized_pieces() -> (
    None
):
    class FiveSampleFrontend(FakeFrontend):
        def push(self, data: bytes) -> list[float]:
            del data
            return [0.0] * 5

        def flush(self) -> list[float]:
            return [0.0] * 5

    factory = FakeFactory()
    request = rasr.RecognizeRequest(config=_recognition(), audio=b"audio")
    response = asyncio.run(
        _servicer(
            factory,
            config=replace(_config(), pre_submit_max_samples=2),
            frontend_factory=FiveSampleFrontend,
        ).Recognize(request, FakeContext())
    )
    assert response.results[0].alternatives[0].transcript == "final"
    assert factory.lease.calls == [
        "feed",
        "feed",
        "feed",
        "feed",
        "feed",
        "feed",
        "flush",
        "finish",
        "release",
    ]


# @spec ING-FE-005, ING-GRPC-001
def test_long_stream_audio_and_tail_are_handed_off_in_credit_sized_pieces() -> (
    None
):
    class FiveSampleFrontend(FakeFrontend):
        def push(self, data: bytes) -> list[float]:
            del data
            return [0.0] * 5

        def flush(self) -> list[float]:
            return [0.0] * 5

    factory = FakeFactory()
    stream_config = rasr.StreamingRecognitionConfig(
        config=_recognition(), interim_results=False
    )
    requests = [
        rasr.StreamingRecognizeRequest(streaming_config=stream_config),
        rasr.StreamingRecognizeRequest(audio_content=b"audio"),
    ]
    responses = asyncio.run(
        _collect_stream(
            _servicer(
                factory,
                config=replace(_config(), pre_submit_max_samples=2),
                frontend_factory=FiveSampleFrontend,
            ).StreamingRecognize(_requests(requests), FakeContext())
        )
    )
    assert responses[-1].results[0].is_final
    assert factory.lease.calls == [
        "feed",
        "feed",
        "feed",
        "feed",
        "feed",
        "feed",
        "flush",
        "finish",
        "release",
    ]


# @spec ING-GRPC-009, ING-GRPC-012
def test_expired_deadline_rejects_before_open_and_never_fabricates_final() -> (
    None
):
    factory = FakeFactory()
    request = rasr.RecognizeRequest(config=_recognition(), audio=b"\x00\x00")
    with pytest.raises(RpcAbort) as error:
        asyncio.run(
            _servicer(factory).Recognize(request, FakeContext(time_remaining=0))
        )
    assert error.value.code is grpc.StatusCode.DEADLINE_EXCEEDED
    assert not factory.opened


# @spec ING-LIFE-002, ING-GRPC-001
def test_silent_preconfiguration_times_out_before_session_open() -> None:
    events: list[str] = []
    token = RecordingOwnerToken(events)
    gate = LoadShedGate(max_sessions=1)
    factory = FakeFactory()

    async def owner_register(_: str) -> RecordingOwnerToken:
        events.append("owner:register")
        return token

    async def silent() -> AsyncIterator[rasr.StreamingRecognizeRequest]:
        await asyncio.Event().wait()
        yield rasr.StreamingRecognizeRequest()

    config = replace(_config(), preconfiguration_timeout=0.01)

    async def exercise() -> None:
        async for _ in _servicer(
            factory,
            config=config,
            load_shed=gate,
            owner_register=owner_register,
        ).StreamingRecognize(silent(), FakeContext()):
            pass

    with pytest.raises(RpcAbort, match="configuration_timeout") as error:
        asyncio.run(asyncio.wait_for(exercise(), timeout=0.2))
    assert error.value.code is grpc.StatusCode.DEADLINE_EXCEEDED
    assert not factory.opened
    assert token.released
    assert events == ["owner:register", "owner:release"]
    assert gate.active == 0


# @spec ING-LIFE-003, ING-GRPC-001
def test_empty_audio_does_not_reset_accepted_audio_idle_timer() -> None:
    factory = FakeFactory()
    stream_config = rasr.StreamingRecognitionConfig(config=_recognition())

    async def requests() -> AsyncIterator[rasr.StreamingRecognizeRequest]:
        yield rasr.StreamingRecognizeRequest(streaming_config=stream_config)
        yield rasr.StreamingRecognizeRequest(audio_content=b"")
        await asyncio.Event().wait()
        yield rasr.StreamingRecognizeRequest(audio_content=b"late")

    config = replace(_config(), session_idle_timeout=0.01)

    async def exercise() -> None:
        async for _ in _servicer(factory, config=config).StreamingRecognize(
            requests(), FakeContext()
        ):
            pass

    with pytest.raises(RpcAbort, match="idle_timeout") as error:
        asyncio.run(asyncio.wait_for(exercise(), timeout=0.2))
    assert error.value.code is grpc.StatusCode.ABORTED
    assert factory.lease.calls == ["abort", "release"]


# @spec ING-LIFE-003, ING-GRPC-001
def test_max_session_duration_times_out_while_the_stream_is_blocked() -> None:
    factory = FakeFactory()
    stream_config = rasr.StreamingRecognitionConfig(config=_recognition())

    async def requests() -> AsyncIterator[rasr.StreamingRecognizeRequest]:
        yield rasr.StreamingRecognizeRequest(streaming_config=stream_config)
        yield rasr.StreamingRecognizeRequest(audio_content=b"audio")
        await asyncio.Event().wait()
        yield rasr.StreamingRecognizeRequest(audio_content=b"late")

    config = replace(
        _config(), session_idle_timeout=1.0, max_session_duration=0.01
    )

    async def exercise() -> None:
        async for _ in _servicer(factory, config=config).StreamingRecognize(
            requests(), FakeContext()
        ):
            pass

    with pytest.raises(RpcAbort, match="request_timeout"):
        asyncio.run(asyncio.wait_for(exercise(), timeout=0.2))
    assert factory.lease.calls == ["feed", "abort", "release"]


# @spec ING-LIFE-010, ING-GRPC-001
def test_stream_finalization_timeout_cancels_without_emitting_final() -> None:
    factory = FakeFactory()
    factory.lease = BlockingFinalizeLease()
    stream_config = rasr.StreamingRecognitionConfig(config=_recognition())
    requests = [
        rasr.StreamingRecognizeRequest(streaming_config=stream_config),
        rasr.StreamingRecognizeRequest(audio_content=b"audio"),
    ]
    config = replace(
        _config(),
        session_finalization_timeout=0.01,
        session_cleanup_timeout=0.01,
    )

    async def exercise() -> list[rasr.StreamingRecognizeResponse]:
        return [
            response
            async for response in _servicer(
                factory, config=config
            ).StreamingRecognize(_requests(requests), FakeContext())
        ]

    with pytest.raises(RpcAbort, match="finalization_timeout") as error:
        asyncio.run(asyncio.wait_for(exercise(), timeout=0.2))
    assert error.value.code is grpc.StatusCode.DEADLINE_EXCEEDED
    assert factory.lease.calls == ["feed", "flush", "abort", "release"]


# @spec ING-LIFE-010, ING-GRPC-003
def test_unary_finalization_respects_the_rpc_deadline_cap() -> None:
    factory = FakeFactory()
    factory.lease = BlockingFinalizeLease()
    request = rasr.RecognizeRequest(config=_recognition(), audio=b"audio")
    config = replace(
        _config(),
        session_finalization_timeout=1.0,
        session_cleanup_timeout=0.01,
    )

    with pytest.raises(RpcAbort, match="finalization_timeout") as error:
        asyncio.run(
            asyncio.wait_for(
                _servicer(factory, config=config).Recognize(
                    request, FakeContext(time_remaining=0.01)
                ),
                timeout=0.2,
            )
        )
    assert error.value.code is grpc.StatusCode.DEADLINE_EXCEEDED
    assert factory.lease.calls == ["feed", "flush", "abort", "release"]


# @spec ING-LIFE-013, ING-VEH-004
def test_builder_preserves_the_injected_owner_factory() -> None:
    def owner_factory(*_: object, **__: object) -> DirectLeaseOwner:
        raise AssertionError("test only verifies factory wiring")

    servicer = build_servicer(
        factory=FakeFactory(),
        config=_config(),
        model_name="nemotron",
        locales=frozenset({"auto", "en-US"}),
        owner_factory=owner_factory,
        admission=FakeAdmission(),
    )
    assert servicer._owner_factory is owner_factory


# @spec ING-GRPC-011, ING-VEH-002
def test_aio_registration_exposes_health_and_reflection() -> None:
    async def exercise() -> health.aio.HealthServicer:
        server = grpc.aio.server()
        result = register_aio_services(
            server=server, servicer=_servicer(FakeFactory())
        )
        await server.stop(None)
        return result

    health_servicer = asyncio.run(exercise())
    assert isinstance(health_servicer, health.aio.HealthServicer)
