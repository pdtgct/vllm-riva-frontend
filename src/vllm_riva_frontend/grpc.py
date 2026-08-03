"""Direct-lease Riva gRPC adapter for the public generated protocol types."""

import asyncio
import inspect
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from typing import NoReturn, Protocol, TypeVar, cast

import grpc
from grpc_health.v1 import health, health_pb2_grpc
from grpc_reflection.v1alpha import reflection
from riva.client.proto import riva_asr_pb2 as rasr
from riva.client.proto import riva_asr_pb2_grpc as rasr_grpc
from riva.client.proto import riva_audio_pb2 as raud

from vllm_riva_frontend.admission import (
    AdmissionLease,
    HostAdmission,
    LoadShedGate,
    LoadShedRegistration,
    LoadShedRejected,
    try_acquire_admission,
)
from vllm_riva_frontend.config import FrontendConfig, dispositioned_fields
from vllm_riva_frontend.errors import catalog
from vllm_riva_frontend.frontend import (
    FormatError,
    RiffFormat,
    sniff_riff,
    validate_format,
)
from vllm_riva_frontend.lease import DirectLeaseOwner, SessionFactory

ENCODING_NAMES = {
    raud.LINEAR_PCM: "LINEAR_PCM",
    raud.MULAW: "MULAW",
    raud.ALAW: "ALAW",
}
RIVA_SERVICE_NAME = rasr.DESCRIPTOR.services_by_name[
    "RivaSpeechRecognition"
].full_name


class AudioFrontend(Protocol):
    """The shared audio decoder/resampler needed by the Riva adapter."""

    def push(self, data: bytes) -> object:
        """Decode one input fragment into normalized samples."""
        ...

    def flush(self) -> object:
        """Drain decoder/resampler state into normalized samples."""
        ...


FrontendFactory = Callable[[str, int], AudioFrontend]
OwnerFactory = Callable[..., DirectLeaseOwner]
RiffResolver = Callable[..., RiffFormat | FormatError | None]


class OwnerToken(Protocol):
    """One lifecycle-wide host ownership registration."""

    async def release(self) -> None:
        """Release the host ownership registration exactly once."""
        ...


OwnerRegister = Callable[[str], Awaitable[OwnerToken]]
T = TypeVar("T")


class _InvalidAcceptanceCredit(Exception):
    """The lease violated the synchronous exact-credit handoff contract."""


def _default_frontend(encoding: str, sample_rate_hz: int) -> AudioFrontend:
    """Construct the shared frontend lazily to keep this adapter testable."""
    from vllm_riva_frontend.frontend import StreamingAudioFrontend

    return StreamingAudioFrontend(
        encoding=encoding, sample_rate_hz=sample_rate_hz
    )


def _context_cancelled(context: object) -> bool:
    """Read the grpc.aio cancellation predicate when the context exposes it."""
    cancelled = getattr(context, "cancelled", None)
    return bool(cancelled()) if callable(cancelled) else False


def _sample_count(samples: object) -> int:
    """Read a frontend piece's sample count structurally."""
    return len(samples)  # type: ignore[arg-type]


def _is_projected_abort(error: BaseException) -> bool:
    """Recognize real and test gRPC abort exceptions."""
    return isinstance(error, grpc.RpcError) or (
        hasattr(error, "code") and hasattr(error, "details")
    )


# @spec ING-GRPC-005, ING-GRPC-006, ING-GRPC-013
def validate_recognition_config(
    recognition: rasr.RecognitionConfig,
    *,
    model_name: str,
    locales: frozenset[str],
) -> list[tuple[str, str]]:
    """Return every named rejection; never silently discard a config field."""
    errors: list[tuple[str, str]] = []
    encoding = recognition.encoding
    if encoding != raud.ENCODING_UNSPECIFIED:
        name = ENCODING_NAMES.get(encoding)
        if name is None or (name, recognition.sample_rate_hertz) not in {
            ("LINEAR_PCM", 16000),
            ("LINEAR_PCM", 8000),
            ("MULAW", 8000),
            ("ALAW", 8000),
        }:
            errors.append(("unsupported_format", "encoding, sample_rate_hertz"))
    if recognition.language_code and recognition.language_code not in locales:
        errors.append(("unknown_locale", "language_code"))
    if recognition.max_alternatives > 1:
        errors.append(("invalid_config_field", "max_alternatives"))
    if recognition.audio_channel_count > 1:
        errors.append(("invalid_config_field", "audio_channel_count"))
    if recognition.model and recognition.model != model_name:
        errors.append(("invalid_config_field", "model"))
    fields = {
        "enable_automatic_punctuation": (
            recognition.enable_automatic_punctuation
        ),
        "verbatim_transcripts": recognition.verbatim_transcripts,
        "profanity_filter": recognition.profanity_filter,
        "speech_contexts": list(recognition.speech_contexts),
        "enable_word_time_offsets": recognition.enable_word_time_offsets,
        "enable_separate_recognition_per_channel": (
            recognition.enable_separate_recognition_per_channel
        ),
        "diarization_config": recognition.diarization_config.ByteSize() > 0,
        "custom_configuration": list(recognition.custom_configuration),
        "endpointing_config": recognition.endpointing_config.ByteSize() > 0,
    }
    for field, disposition in dispositioned_fields(
        fields, dialect="grpc"
    ).items():
        if not fields[field]:
            continue
        if disposition in {"honored", "model_intrinsic"}:
            continue
        code = (
            "unsupported_capability"
            if disposition == "unsupported_capability"
            else "invalid_config_field"
        )
        errors.append((code, field))
    return errors


class RivaServicer(
    rasr_grpc.RivaSpeechRecognitionServicer,  # type: ignore[misc]
):
    """Async public Riva API bound only through the RFC-1 lease owner."""

    def __init__(
        self,
        *,
        factory: SessionFactory,
        config: FrontendConfig,
        model_name: str,
        locales: frozenset[str],
        frontend_factory: FrontendFactory = _default_frontend,
        owner_factory: OwnerFactory = DirectLeaseOwner,
        load_shed: LoadShedGate | None = None,
        riff_resolver: RiffResolver = sniff_riff,
        admission: HostAdmission,
        owner_register: OwnerRegister | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Bind immutable settings and the host's one engine-local factory."""
        self._factory = factory
        self._config = config
        self._model_name = model_name
        self._locales = locales
        self._frontend_factory = frontend_factory
        self._owner_factory = owner_factory
        self._load_shed = load_shed
        self._riff_resolver = riff_resolver
        self._admission = admission
        self._owner_register = owner_register
        self._clock = clock

    async def _abort(self, context: object, code: str, fields: str) -> NoReturn:
        """Project a code-first gRPC status and then stop the RPC."""
        projection = catalog()[code]
        abort = getattr(context, "abort", None)
        if not callable(abort):
            raise TypeError("gRPC context must expose abort")
        result = abort(
            getattr(grpc.StatusCode, projection.grpc_status),
            f"{code}: {fields}",
        )
        if inspect.isawaitable(result):
            await result
        raise RuntimeError("grpc context.abort must terminate the RPC")

    async def _registration(
        self, kind: str, context: object
    ) -> LoadShedRegistration | None:
        """Atomically register only counted compatibility inference work."""
        if self._load_shed is None:
            return None
        result = await self._load_shed.register(kind)
        if isinstance(result, LoadShedRejected):
            await self._abort(
                context, result.code, f"authority: {result.authority}"
            )
        return result

    async def _admission_or_abort(self, context: object) -> AdmissionLease:
        """Acquire host ownership before consuming compatibility work."""
        lease = try_acquire_admission(self._admission)
        if lease is None:
            await self._abort(
                context, "service_unavailable", "application admission closed"
            )
        return lease

    async def _owner_token(self, kind: str) -> OwnerToken | None:
        """Register host ownership after shared load-shed admission succeeds."""
        if self._owner_register is None:
            return None
        return await self._owner_register(kind)

    async def _validate_or_abort(
        self, recognition: rasr.RecognitionConfig, context: object
    ) -> None:
        """Reject every invalid recognition field before opening a lease."""
        rejections = validate_recognition_config(
            recognition, model_name=self._model_name, locales=self._locales
        )
        if rejections:
            code, first = rejections[0]
            fields = ", ".join(field for _, field in rejections)
            await self._abort(context, code, fields or first)

    async def _deadline_or_abort(self, context: object) -> None:
        """Honor an already-expired gRPC deadline at every owner boundary."""
        time_remaining = getattr(context, "time_remaining", None)
        remaining = time_remaining() if callable(time_remaining) else None
        if remaining is not None and remaining <= 0:
            await self._abort(context, "request_timeout", "grpc deadline")

    @staticmethod
    def _rpc_remaining(context: object) -> float | None:
        """Return a positive gRPC deadline budget when the host supplied one."""
        time_remaining = getattr(context, "time_remaining", None)
        remaining = time_remaining() if callable(time_remaining) else None
        return remaining if remaining is None or remaining > 0 else 0.0

    async def _await_until(
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        deadline: float,
        context: object,
        timeout_code: str,
        timeout_fields: str,
    ) -> T:
        """Bound one blocking wait by its absolute and gRPC deadlines."""
        await self._deadline_or_abort(context)
        timeout = deadline - self._clock()
        rpc_remaining = self._rpc_remaining(context)
        if rpc_remaining is not None:
            if rpc_remaining <= 0:
                await self._abort(context, "request_timeout", "grpc deadline")
            if rpc_remaining < timeout:
                timeout = rpc_remaining
        if timeout <= 0:
            await self._abort(context, timeout_code, timeout_fields)
        try:
            return await asyncio.wait_for(operation(), timeout=timeout)
        except asyncio.TimeoutError:
            await self._abort(context, timeout_code, timeout_fields)
        raise AssertionError("context.abort must terminate the RPC")

    async def _await_stream_activity(
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        deadline: Callable[[], tuple[float, str, str]],
        accepted_event: asyncio.Event,
        context: object,
    ) -> T:
        """Recompute streaming lifetime bounds when receipt credit arrives."""
        task: asyncio.Future[T] = asyncio.ensure_future(operation())
        try:
            while True:
                absolute, code, fields = deadline()
                remaining = absolute - self._clock()
                rpc_remaining = self._rpc_remaining(context)
                if rpc_remaining is not None and rpc_remaining < remaining:
                    remaining = rpc_remaining
                    fields = "grpc deadline"
                if remaining <= 0:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    await self._abort(context, code, fields)
                accepted_wait = asyncio.create_task(accepted_event.wait())
                activity_tasks = {
                    cast(asyncio.Future[object], task),
                    cast(asyncio.Future[object], accepted_wait),
                }
                done, _ = await asyncio.wait(
                    activity_tasks,
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if task in done:
                    accepted_wait.cancel()
                    await asyncio.gather(accepted_wait, return_exceptions=True)
                    return await task
                if accepted_wait in done:
                    accepted_event.clear()
                    continue
                accepted_wait.cancel()
                await asyncio.gather(accepted_wait, return_exceptions=True)
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await self._abort(context, code, fields)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def _bounded_feed(
        self,
        owner: DirectLeaseOwner,
        samples: object,
        outstanding: list[int],
        context: object,
        on_audio_accepted: Callable[[int], None] | None = None,
        activity_deadline: (Callable[[], tuple[float, str, str]] | None) = None,
        accepted_event: asyncio.Event | None = None,
    ) -> list[str]:
        """Split normalized samples into credit-bounded lease handoffs."""
        count = _sample_count(samples)
        if not count:
            return []
        hypotheses: list[str] = []
        start = 0
        while start < count:
            available = self._config.pre_submit_max_samples - outstanding[0]
            if available <= 0:
                await self._abort(
                    context, "buffer_overflow", "pre_submit_max_samples"
                )
            piece_count = min(available, count - start)
            piece = samples[start : start + piece_count]  # type: ignore[index]
            credit: list[int | None] = [None]

            def accepted(
                accepted_count: int,
                *,
                expected_count: int = piece_count,
                piece_credit: list[int | None] = credit,
            ) -> None:
                if (
                    type(accepted_count) is not int
                    or accepted_count != expected_count
                    or piece_credit[0] is not None
                ):
                    raise _InvalidAcceptanceCredit
                piece_credit[0] = accepted_count
                outstanding[0] -= accepted_count
                if on_audio_accepted is not None:
                    on_audio_accepted(accepted_count)

            outstanding[0] += piece_count
            try:

                def operation(
                    current_piece: object = piece,
                    callback: Callable[[int], None] = accepted,
                ) -> Awaitable[list[str]]:
                    return owner.feed(current_piece, on_accepted=callback)

                if activity_deadline is None or accepted_event is None:
                    piece_hypotheses = await operation()
                else:
                    piece_hypotheses = await self._await_stream_activity(
                        operation,
                        deadline=activity_deadline,
                        accepted_event=accepted_event,
                        context=context,
                    )
                hypotheses.extend(piece_hypotheses)
            except _InvalidAcceptanceCredit:
                await self._abort(
                    context, "internal", "provider acceptance credit"
                )
            if credit[0] != piece_count:
                await self._abort(
                    context, "internal", "provider acceptance credit"
                )
            start += piece_count
        return hypotheses

    async def _resolve_riff_or_abort(
        self,
        data: bytes,
        recognition: rasr.RecognitionConfig,
        context: object,
    ) -> tuple[AudioFrontend, bytes, str, int, int] | None:
        """Resolve/validate deferred RIFF without decoding header metadata."""
        resolved = self._riff_resolver(
            data, max_header_bytes=self._config.max_riff_header_bytes
        )
        if resolved is None:
            return None
        if isinstance(resolved, FormatError):
            await self._abort(
                context, resolved.code, ", ".join(resolved.fields)
            )
        if recognition.sample_rate_hertz and (
            recognition.sample_rate_hertz != resolved.sample_rate_hz
        ):
            await self._abort(
                context, "unsupported_format", "sample_rate_hertz"
            )
        if recognition.audio_channel_count and (
            recognition.audio_channel_count != resolved.channels
        ):
            await self._abort(
                context, "unsupported_format", "audio_channel_count"
            )
        format_error = validate_format(
            resolved.encoding, resolved.sample_rate_hz, resolved.channels
        )
        if format_error is not None:
            await self._abort(
                context, format_error.code, ", ".join(format_error.fields)
            )
        return (
            self._frontend_factory(resolved.encoding, resolved.sample_rate_hz),
            data[resolved.data_offset :],
            resolved.encoding,
            resolved.sample_rate_hz,
            resolved.data_bytes,
        )

    @staticmethod
    def _duration_exceeds(
        data: bytes, *, encoding: str, sample_rate_hz: int, limit: float
    ) -> bool:
        """Derive decoded duration before frontend allocation."""
        bytes_per_sample = 2 if encoding == "LINEAR_PCM" else 1
        return len(data) / (bytes_per_sample * sample_rate_hz) > limit

    @staticmethod
    def _stream_response(
        transcript: str, *, final: bool
    ) -> rasr.StreamingRecognizeResponse:
        """Make one cumulative Riva streaming recognition result."""
        response = rasr.StreamingRecognizeResponse()
        result = response.results.add()
        result.is_final = final
        result.alternatives.add().transcript = transcript
        return response

    # @spec ING-GRPC-001, ING-GRPC-002, ING-CORE-007, ING-LIFE-001
    async def StreamingRecognize(
        self,
        request_iterator: AsyncIterator[rasr.StreamingRecognizeRequest],
        context: object,
    ) -> AsyncIterator[rasr.StreamingRecognizeResponse]:
        """Require config/open before further reads, then drive one lease."""
        admission_lease = await self._admission_or_abort(context)
        registration: LoadShedRegistration | None = None
        owner_token: OwnerToken | None = None
        owner: DirectLeaseOwner | None = None
        try:
            registration = await self._registration(
                "grpc_streaming_recognize", context
            )
            preconfiguration_deadline = (
                self._clock() + self._config.preconfiguration_timeout
            )
            owner_token = await self._await_until(
                lambda: self._owner_token("grpc_streaming_recognize"),
                deadline=preconfiguration_deadline,
                context=context,
                timeout_code="configuration_timeout",
                timeout_fields="owner registration",
            )
            try:
                first = await self._await_until(
                    lambda: anext(request_iterator),
                    deadline=preconfiguration_deadline,
                    context=context,
                    timeout_code="configuration_timeout",
                    timeout_fields="streaming_config",
                )
            except StopAsyncIteration:
                await self._abort(context, "protocol_order", "streaming_config")
                return
            if first.WhichOneof("streaming_request") != "streaming_config":
                await self._abort(context, "protocol_order", "streaming_config")
                return
            if first.runtime_config:
                await self._abort(
                    context, "invalid_config_field", "runtime_config"
                )
                return
            streaming = first.streaming_config
            recognition = streaming.config
            await self._validate_or_abort(recognition, context)
            owner = self._owner_factory(
                self._factory,
                cleanup_timeout=self._config.session_cleanup_timeout,
            )
            await self._await_until(
                lambda: owner.open(
                    cadence="1120ms", locale=recognition.language_code or "auto"
                ),
                deadline=preconfiguration_deadline,
                context=context,
                timeout_code="configuration_timeout",
                timeout_fields="session open",
            )
            opened_at = self._clock()
            last_accepted_audio = [opened_at]
            max_session_deadline = (
                opened_at + self._config.max_session_duration
                if self._config.max_session_duration is not None
                else None
            )
            frontend: AudioFrontend | None = None
            if recognition.encoding != raud.ENCODING_UNSPECIFIED:
                encoding = ENCODING_NAMES[recognition.encoding]
                frontend = self._frontend_factory(
                    encoding,
                    recognition.sample_rate_hertz,
                )
            riff_head = b""
            riff_remaining: int | None = None
            outstanding = [0]
            accepted_event = asyncio.Event()
            last_hypothesis = ""

            def note_accepted_audio(_: int) -> None:
                last_accepted_audio[0] = self._clock()
                accepted_event.set()

            def activity_deadline() -> tuple[float, str, str]:
                idle = (
                    last_accepted_audio[0] + self._config.session_idle_timeout
                )
                if (
                    max_session_deadline is not None
                    and max_session_deadline <= idle
                ):
                    return (
                        max_session_deadline,
                        "request_timeout",
                        "session duration",
                    )
                return idle, "idle_timeout", "streaming audio"

            while True:
                stream_deadline, timeout_code, timeout_fields = (
                    activity_deadline()
                )
                try:
                    request = await self._await_until(
                        lambda: anext(request_iterator),
                        deadline=stream_deadline,
                        context=context,
                        timeout_code=timeout_code,
                        timeout_fields=timeout_fields,
                    )
                except StopAsyncIteration:
                    break
                if _context_cancelled(context):
                    await owner.cancel()
                    return
                if request.runtime_config:
                    await self._abort(
                        context, "invalid_config_field", "runtime_config"
                    )
                if request.WhichOneof("streaming_request") != "audio_content":
                    await self._abort(
                        context, "protocol_order", "audio_content"
                    )
                if frontend is None:
                    riff_candidate = riff_head + request.audio_content
                    resolved = await self._resolve_riff_or_abort(
                        riff_candidate, recognition, context
                    )
                    if resolved is None:
                        riff_head = riff_candidate
                        continue
                    frontend, payload, _, _, riff_remaining = resolved
                    riff_head = b""
                else:
                    payload = request.audio_content
                if riff_remaining is not None:
                    payload = payload[:riff_remaining]
                    riff_remaining -= len(payload)
                hypotheses = await self._bounded_feed(
                    owner,
                    frontend.push(payload),
                    outstanding,
                    context,
                    on_audio_accepted=note_accepted_audio,
                    activity_deadline=activity_deadline,
                    accepted_event=accepted_event,
                )
                if streaming.interim_results:
                    for hypothesis in hypotheses:
                        if hypothesis == last_hypothesis:
                            continue
                        last_hypothesis = hypothesis
                        yield self._stream_response(hypothesis, final=False)
            if _context_cancelled(context):
                await owner.cancel()
                return
            finalization_deadline = (
                self._clock() + self._config.session_finalization_timeout
            )
            if frontend is not None:
                tail_samples = frontend.flush()
                await self._bounded_feed(
                    owner,
                    tail_samples,
                    outstanding,
                    context,
                    activity_deadline=lambda: (
                        finalization_deadline,
                        "finalization_timeout",
                        "session finalization",
                    ),
                    accepted_event=accepted_event,
                )
            elif riff_head:
                await self._abort(context, "unsupported_format", "RIFF header")
            if riff_remaining not in {None, 0}:
                await self._abort(
                    context, "invalid_audio", "truncated RIFF data"
                )
            tail = await self._await_until(
                owner.complete,
                deadline=finalization_deadline,
                context=context,
                timeout_code="finalization_timeout",
                timeout_fields="session finalization",
            )
            if tail is not None:
                yield self._stream_response(tail, final=True)
        except asyncio.CancelledError:
            if owner is not None:
                with suppress(BaseException):
                    await owner.cancel()
            raise
        except BaseException as error:
            if owner is not None:
                with suppress(BaseException):
                    await owner.cancel()
            if _is_projected_abort(error):
                raise
            if _context_cancelled(context):
                return
            await self._abort(context, "internal", "provider failure")
        finally:
            try:
                if owner_token is not None:
                    await owner_token.release()
            finally:
                try:
                    if registration is not None:
                        await registration.release()
                finally:
                    admission_lease.release()

    # @spec ING-GRPC-003, ING-GRPC-008, ING-GRPC-009, ING-GRPC-010
    async def Recognize(
        self, request: rasr.RecognizeRequest, context: object
    ) -> rasr.RecognizeResponse:
        """Run bounded unary recognition through the same 1120-ms lease path."""
        admission_lease = await self._admission_or_abort(context)
        registration: LoadShedRegistration | None = None
        owner_token: OwnerToken | None = None
        owner: DirectLeaseOwner | None = None
        try:
            registration = await self._registration("grpc_recognize", context)
            owner_token = await self._owner_token("grpc_recognize")
            await self._validate_or_abort(request.config, context)
            if getattr(request, "runtime_config", None):
                await self._abort(
                    context, "invalid_config_field", "runtime_config"
                )
            if len(request.audio) > self._config.unary_max_encoded_audio_bytes:
                await self._abort(
                    context, "request_too_large", "audio too long"
                )
            await self._deadline_or_abort(context)
            if _context_cancelled(context):
                return rasr.RecognizeResponse()
            if request.config.encoding == raud.ENCODING_UNSPECIFIED:
                resolved = await self._resolve_riff_or_abort(
                    request.audio, request.config, context
                )
                if resolved is None:
                    await self._abort(
                        context, "unsupported_format", "RIFF header"
                    )
                (
                    frontend,
                    payload,
                    duration_encoding,
                    duration_rate,
                    data_bytes,
                ) = resolved
                if len(payload) < data_bytes:
                    await self._abort(
                        context, "invalid_audio", "truncated RIFF data"
                    )
                payload = payload[:data_bytes]
            else:
                duration_encoding = ENCODING_NAMES[request.config.encoding]
                duration_rate = request.config.sample_rate_hertz
                frontend = self._frontend_factory(
                    duration_encoding, duration_rate
                )
                payload = request.audio
            if self._duration_exceeds(
                payload,
                encoding=duration_encoding,
                sample_rate_hz=duration_rate,
                limit=self._config.unary_max_decoded_duration_seconds,
            ):
                await self._abort(
                    context, "request_too_large", "audio too long"
                )
            owner = self._owner_factory(
                self._factory,
                cleanup_timeout=self._config.session_cleanup_timeout,
            )
            finalization_deadline = (
                self._clock() + self._config.session_finalization_timeout
            )
            await self._await_until(
                lambda: owner.open(
                    cadence="1120ms",
                    locale=request.config.language_code or "auto",
                ),
                deadline=finalization_deadline,
                context=context,
                timeout_code="finalization_timeout",
                timeout_fields="session open",
            )
            outstanding = [0]
            terminal_event = asyncio.Event()

            def terminal_deadline() -> tuple[float, str, str]:
                return (
                    finalization_deadline,
                    "finalization_timeout",
                    "session finalization",
                )

            samples = frontend.push(payload)
            await self._bounded_feed(
                owner,
                samples,
                outstanding,
                context,
                activity_deadline=terminal_deadline,
                accepted_event=terminal_event,
            )
            tail_samples = frontend.flush()
            await self._bounded_feed(
                owner,
                tail_samples,
                outstanding,
                context,
                activity_deadline=terminal_deadline,
                accepted_event=terminal_event,
            )
            await self._deadline_or_abort(context)
            if _context_cancelled(context):
                await owner.cancel()
                return rasr.RecognizeResponse()
            transcript = await self._await_until(
                owner.complete,
                deadline=finalization_deadline,
                context=context,
                timeout_code="finalization_timeout",
                timeout_fields="session finalization",
            )
            response = rasr.RecognizeResponse()
            if transcript is not None:
                response.results.add().alternatives.add().transcript = (
                    transcript
                )
            return response
        except asyncio.CancelledError:
            if owner is not None:
                with suppress(BaseException):
                    await owner.cancel()
            raise
        except BaseException as error:
            if owner is not None:
                with suppress(BaseException):
                    await owner.cancel()
            if _is_projected_abort(error):
                raise
            if _context_cancelled(context):
                return rasr.RecognizeResponse()
            await self._abort(context, "internal", "provider failure")
        finally:
            try:
                if owner_token is not None:
                    await owner_token.release()
            finally:
                try:
                    if registration is not None:
                        await registration.release()
                finally:
                    admission_lease.release()

    # @spec ING-GRPC-004, ING-GRPC-006
    async def GetRivaSpeechRecognitionConfig(
        self, request: rasr.RivaSpeechRecognitionConfigRequest, context: object
    ) -> rasr.RivaSpeechRecognitionConfigResponse:
        """Return the exact static model and accept-matrix capabilities."""
        del request, context
        response = rasr.RivaSpeechRecognitionConfigResponse()
        entry = response.model_config.add()
        entry.model_name = self._model_name
        entry.parameters["streaming"] = "true"
        entry.parameters["offline"] = "true"
        entry.parameters["supported_sample_rates"] = "8000,16000"
        entry.parameters["supported_encodings"] = "LINEAR_PCM,MULAW,ALAW"
        entry.parameters["language_code"] = ",".join(sorted(self._locales))
        entry.parameters["enable_automatic_punctuation"] = "model-intrinsic"
        entry.parameters["verbatim_transcripts"] = "model-intrinsic"
        return response


# @spec ING-GRPC-011, ING-VEH-002
def register_aio_services(
    *, server: grpc.aio.Server, servicer: RivaServicer
) -> health.aio.HealthServicer:
    """Register Riva, standard async health, and reflection with one server."""
    rasr_grpc.add_RivaSpeechRecognitionServicer_to_server(servicer, server)
    health_servicer = health.aio.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    reflection.enable_server_reflection(
        (
            RIVA_SERVICE_NAME,
            health.SERVICE_NAME,
            reflection.SERVICE_NAME,
        ),
        server,
    )
    return health_servicer


# @spec ING-GRPC-001, ING-GRPC-011
def build_servicer(
    *,
    factory: SessionFactory,
    config: FrontendConfig,
    model_name: str,
    locales: frozenset[str],
    frontend_factory: FrontendFactory = _default_frontend,
    owner_factory: OwnerFactory = DirectLeaseOwner,
    load_shed: LoadShedGate | None = None,
    admission: HostAdmission,
    owner_register: OwnerRegister | None = None,
) -> RivaServicer:
    """Build an unbound Riva servicer for the lifecycle owner to start."""
    return RivaServicer(
        factory=factory,
        config=config,
        model_name=model_name,
        locales=locales,
        frontend_factory=frontend_factory,
        owner_factory=owner_factory,
        load_shed=load_shed,
        admission=admission,
        owner_register=owner_register,
    )
