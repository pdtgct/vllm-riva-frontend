"""The Riva gRPC surface: the ``RivaSpeechRecognition`` servicer and listener.

The servicer speaks the public protocol with the generated stubs of the
``nvidia-riva-client`` package (``riva.client.proto``). ``StreamingRecognize``
drives one ``lease.StreamOwner`` per stream; ``Recognize`` validates and
converts a complete request here and hands it to the files leaf's execution
path; ``GetRivaSpeechRecognitionConfig`` is a static description.

The private helpers under "stream owner" are shared with ``nim_ws``: both
streaming surfaces read client input through the owner's bounded
``receive`` and hand every terminal cause to the owner.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar

import grpc
import grpc.aio
import numpy as np
from grpc_health.v1 import health, health_pb2_grpc
from grpc_reflection.v1alpha import reflection
from riva.client.proto import riva_asr_pb2 as rasr
from riva.client.proto import riva_asr_pb2_grpc as rasr_grpc
from riva.client.proto import riva_audio_pb2 as raudio

# Modules, not names: ``lifecycle`` imports this module while it loads.
from vllm_riva_frontend import admission, errors, frontend, lease, surfaces

if TYPE_CHECKING:
    from vllm_riva_frontend.config import FrontendConfig

T = TypeVar("T")

RIVA_SERVICE_NAME: str = rasr.DESCRIPTOR.services_by_name[
    "RivaSpeechRecognition"
].full_name

#: The proto encodings the accept-matrix serves, by the audio leaf's names.
ENCODING_NAMES: Mapping[int, str] = MappingProxyType(
    {
        raudio.AudioEncoding.Value("LINEAR_PCM"): "LINEAR_PCM",
        raudio.AudioEncoding.Value("MULAW"): "MULAW",
        raudio.AudioEncoding.Value("ALAW"): "ALAW",
    }
)
_UNSPECIFIED = raudio.AudioEncoding.Value("ENCODING_UNSPECIFIED")

#: The disposition of every ``RecognitionConfig`` field of the pinned stubs:
#: ``honoured``, ``model_intrinsic`` (accepted under either value, output
#: unchanged), ``rejected`` (``invalid_config_field`` when set) or
#: ``capability`` (``unsupported_capability`` when set).
FIELD_DISPOSITIONS: Mapping[str, str] = MappingProxyType(
    {
        "encoding": "honoured",
        "sample_rate_hertz": "honoured",
        "language_code": "honoured",
        "max_alternatives": "honoured",
        "profanity_filter": "rejected",
        "speech_contexts": "capability",
        "audio_channel_count": "honoured",
        "enable_word_time_offsets": "rejected",
        "enable_automatic_punctuation": "model_intrinsic",
        "enable_separate_recognition_per_channel": "rejected",
        "model": "honoured",
        "verbatim_transcripts": "model_intrinsic",
        "diarization_config": "capability",
        "custom_configuration": "rejected",
        "endpointing_config": "capability",
    }
)

#: The audio leaf's neutral format fields by their Riva names.
_RIVA_FORMAT_FIELDS = MappingProxyType(
    {
        "encoding": "encoding",
        "sample_rate_hz": "sample_rate_hertz",
        "channels": "audio_channel_count",
    }
)
_MODEL_INTRINSIC_NOTE = (
    "enable_automatic_punctuation and verbatim_transcripts are model-intrinsic:"
    " either value is accepted and neither changes the output; punctuation"
    " follows the checkpoint and transcripts are never re-normalized"
)


# ================================================================ validation


def _format_error(
    error: frontend.FormatError, names: Mapping[str, str]
) -> errors.FrontendError:
    return errors.FrontendError(
        "unsupported_format",
        error.detail,
        fields=tuple(dict.fromkeys(names[f] for f in error.fields)),
    )


def _primary(rejections: list[errors.FrontendError]) -> errors.FrontendError:
    """One gRPC status for every rejection of one message.

    The first rejection's code leads; every field rejected with that code is
    named, and the rest are listed in the detail, so nothing is silent.
    """
    first = rejections[0]
    same = [e for e in rejections if e.code == first.code]
    fields = tuple(dict.fromkeys(f for e in same for f in e.fields))
    detail = "; ".join(e.detail for e in same if e.detail)
    others = [e for e in rejections if e.code != first.code]
    if others:
        listed = ", ".join(
            f"{e.code} ({', '.join(e.fields)})" if e.fields else e.code
            for e in others
        )
        detail = f"{detail}; also rejected: {listed}"
    return errors.FrontendError(first.code, detail, fields=fields)


# @spec ING-GRPC-005, ING-GRPC-006, ING-GRPC-017, ING-GRPC-018, ING-GRPC-022
# @spec ING-GRPC-023, ING-GRPC-024, ING-EOU-003, ING-EP-026
def validate_recognition_config(
    config: rasr.RecognitionConfig,
    *,
    served_model: str,
    locales: frozenset[str],
    default_locale: str | None,
) -> list[errors.FrontendError]:
    """Every rejection of one ``RecognitionConfig``, each naming its field.

    Zero ``audio_channel_count`` and ``max_alternatives`` are unset (1).
    Under ``ENCODING_UNSPECIFIED`` a zero rate or channel count is
    unconstrained and a non-zero one is a constraint the header must match
    later; under a concrete encoding they are the cell itself.
    """
    rejections: list[errors.FrontendError] = []
    rate = config.sample_rate_hertz
    channels = config.audio_channel_count or 1
    if config.encoding == _UNSPECIFIED:
        if rate not in (0, 8_000, 16_000):
            rejections.append(
                errors.FrontendError(
                    "unsupported_format",
                    f"no accepted cell has a sample rate of {rate} Hz",
                    fields=("sample_rate_hertz",),
                )
            )
        if channels != 1:
            rejections.append(
                errors.FrontendError(
                    "unsupported_format",
                    "only mono audio is accepted",
                    fields=("audio_channel_count",),
                )
            )
    else:
        name = ENCODING_NAMES.get(config.encoding) or _encoding_label(
            config.encoding
        )
        problem = frontend.validate_format(name, rate, channels)
        if problem is not None:
            rejections.append(_format_error(problem, _RIVA_FORMAT_FIELDS))
    locale = surfaces.locale_error(
        config.language_code,
        field="language_code",
        locales=locales,
        default_locale=default_locale,
    )
    if locale is not None:
        rejections.append(locale)
    if config.max_alternatives > 1:
        rejections.append(
            _field_error(
                "invalid_config_field",
                "max_alternatives",
                "greedy decoding has one hypothesis",
            )
        )
    for name in (
        "profanity_filter",
        "enable_word_time_offsets",
        "enable_separate_recognition_per_channel",
    ):
        if getattr(config, name):
            rejections.append(
                _field_error(
                    "invalid_config_field", name, "is not supported when true"
                )
            )
    if config.model and config.model != served_model:
        rejections.append(
            _field_error(
                "invalid_config_field",
                "model",
                f"this server serves {served_model!r}",
            )
        )
    if len(config.speech_contexts):
        rejections.append(
            _field_error(
                "unsupported_capability", "speech_contexts", "word boosting"
            )
        )
    if config.diarization_config.ListFields():
        rejections.append(
            _field_error(
                "unsupported_capability",
                "diarization_config",
                "speaker diarization",
            )
        )
    if len(config.custom_configuration):
        rejections.append(
            _field_error(
                "invalid_config_field",
                "custom_configuration",
                "no custom configuration key is accepted",
            )
        )
    if config.endpointing_config.ListFields():
        rejections.append(
            _field_error(
                "unsupported_capability",
                "endpointing_config",
                "end-of-utterance endpointing",
            )
        )
    return rejections


def _field_error(code: str, field: str, detail: str) -> errors.FrontendError:
    return errors.FrontendError(code, f"{field}: {detail}", fields=(field,))


def _encoding_label(value: int) -> str:
    try:
        return str(raudio.AudioEncoding.Name(value))
    except ValueError:
        return f"encoding {value}"


# ============================================================== description


# @spec ING-GRPC-014, ING-GRPC-006
def config_response(plugin: Any) -> rasr.RivaSpeechRecognitionConfigResponse:
    """The static ``GetRivaSpeechRecognitionConfig`` description.

    One entry per supported type for the served identity: ``streaming``
    always, ``offline`` while complete files are enabled.
    """
    types = ["streaming"]
    if plugin.files_enabled:
        types.append("offline")
    locales = ",".join(sorted(plugin.factory.locales))
    response = rasr.RivaSpeechRecognitionConfigResponse()
    for kind in types:
        entry = response.model_config.add()
        entry.model_name = plugin.served_model
        entry.parameters["type"] = kind
        entry.parameters["sample_rates"] = "8000,16000"
        entry.parameters["encodings"] = "LINEAR_PCM,MULAW,ALAW"
        entry.parameters["language_codes"] = locales
        entry.parameters["notes"] = _MODEL_INTRINSIC_NOTE
    return response


# =============================================================== listener


# @spec ING-GRPC-001
def register_aio_services(
    server: grpc.aio.Server, servicer: RivaServicer
) -> health.aio.HealthServicer:
    """Register the Riva servicer, standard health and reflection."""
    rasr_grpc.add_RivaSpeechRecognitionServicer_to_server(servicer, server)
    health_servicer = health.aio.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    reflection.enable_server_reflection(
        (RIVA_SERVICE_NAME, health.SERVICE_NAME, reflection.SERVICE_NAME),
        server,
    )
    return health_servicer


# @spec ING-GRPC-031
def build_server(
    config: FrontendConfig, servicer: RivaServicer
) -> tuple[grpc.aio.Server, health.aio.HealthServicer]:
    """An unbound ``grpc.aio`` server carrying the frontend's services.

    The receive cap is ``grpc_receive_max_bytes`` for every request message
    of both RPCs; keepalive is on only when ``grpc_keepalive_seconds`` is
    set. The caller binds the address and starts the server.
    """
    options: list[tuple[str, Any]] = [
        ("grpc.max_receive_message_length", int(config.grpc_receive_max_bytes)),
    ]
    keepalive = getattr(config, "grpc_keepalive_seconds", None)
    if keepalive is not None:
        options.append(("grpc.keepalive_time_ms", int(keepalive * 1000)))
    server = grpc.aio.server(options=options)
    return server, register_aio_services(server, servicer)


def _rpc_deadline(clock: Any, context: Any) -> float | None:
    """The RPC's absolute deadline, read once at servicer entry."""
    remaining = context.time_remaining()
    if remaining is None:
        return None
    return float(clock.monotonic()) + max(0.0, float(remaining))


class _Abandoned(Exception):
    """A write or the status exceeded its send bound: nothing more is sent."""


async def _abort(
    plugin: Any,
    context: Any,
    error: errors.FrontendError,
    *,
    rpc_deadline: float | None,
    owner: Any = None,
    watch: surfaces.ShutdownWatch | None = None,
) -> NoReturn:
    """End the RPC with ``error``'s status under the send bound.

    Raises what ``context.abort`` raises; a status the transport does not
    take within the bound raises ``_Abandoned`` instead.
    """
    status = plugin.catalog.grpc_status(error.code)
    try:
        await surfaces.bounded_send(
            plugin,
            context.abort(status, error.grpc_details()),
            owner=owner,
            rpc_deadline=rpc_deadline,
            watch=watch,
        )
    except TimeoutError:
        raise _Abandoned from None
    raise RuntimeError("context.abort returned")  # pragma: no cover


_TRANSPORT_GONE = errors.FrontendError("internal", "the transport is gone")


# ================================================================ servicer

_HALF_CLOSE = object()  # what a read returns at the client's half-close
_TRIGGER = object()  # what a read returns once the shutdown trigger is seen


def _streaming_response(
    text: str, *, final: bool
) -> rasr.StreamingRecognizeResponse:
    result = rasr.StreamingRecognitionResult(
        alternatives=[rasr.SpeechRecognitionAlternative(transcript=text)]
    )
    if final:
        result.is_final = True
    return rasr.StreamingRecognizeResponse(results=[result])


def _recognize_response(text: str) -> rasr.RecognizeResponse:
    return rasr.RecognizeResponse(
        results=[
            rasr.SpeechRecognitionResult(
                alternatives=[
                    rasr.SpeechRecognitionAlternative(transcript=text)
                ]
            )
        ]
    )


class _ClientGone(Exception):
    """The client cancelled the RPC: no response is produced."""


class _RivaStream:
    """One ``StreamingRecognize`` call over its ``StreamOwner``.

    The handler reads the request iterator only between owner operations,
    one message at a time, through the owner's bounded ``receive``, and
    writes each response with ``context.write`` under the send bound. The
    client's cancellation reaches the owner at once through the context's
    done callback and ``fail``, which interrupts a running call.
    """

    def __init__(
        self,
        plugin: Any,
        owner: Any,
        record: Any,
        context: Any,
        requests: AsyncIterator[rasr.StreamingRecognizeRequest],
        watch: surfaces.ShutdownWatch,
    ) -> None:
        self._plugin = plugin
        self._config = plugin.config
        self._owner = owner
        self.record = record
        self._context = context
        self._watch = watch
        self._requests = requests.__aiter__()
        self._interim = True
        self._hypothesis = ""
        self._tasks: list[asyncio.Future[Any]] = []
        self.gone = False
        # Deferred RIFF header: the declared constraints and the bytes so far.
        self._riff: bytearray | None = None
        self._declared_rate = 0
        self._declared_channels = 0
        self._remaining: int | None = None

    # ------------------------------------------------------------ signals

    # @spec ING-LIFE-039
    def arm(self) -> None:
        self._context.add_done_callback(self._on_done)

    def _on_done(self, context: Any) -> None:
        if self.gone or not context.cancelled():
            return
        self.gone = True
        if self._owner.state != "terminal":
            cause = lease.Cause("internal", "transport", None)
            self._tasks.append(
                surfaces.child(self.record, self._owner.fail(cause))
            )

    async def close(self) -> None:
        await surfaces.join(
            self._tasks, stopped=lambda: surfaces.torn_down(self._plugin)
        )

    # ------------------------------------------------------------- writes

    async def _write(self, response: rasr.StreamingRecognizeResponse) -> None:
        """One response under the send bound; past it, the transport path."""
        if self.gone:
            raise _ClientGone
        try:
            await surfaces.bounded_send(
                self._plugin,
                self._context.write(response),
                owner=self._owner,
                rpc_deadline=self._owner.rpc_deadline,
                watch=self._watch,
            )
        except Exception as failure:  # beyond the bound, or the write failed
            self.gone = True
            raise _Abandoned from failure

    async def abort(self, error: errors.FrontendError) -> NoReturn:
        """Project the owner's terminal error as the RPC's status."""
        await _abort(
            self._plugin,
            self._context,
            error,
            rpc_deadline=self._owner.rpc_deadline,
            owner=self._owner,
            watch=self._watch,
        )

    # -------------------------------------------------------------- reads

    async def _next_request(self) -> Any:
        try:
            return await self._requests.__anext__()
        except StopAsyncIteration:
            return _HALF_CLOSE

    async def _read(self) -> Any:
        """The next request, ``_HALF_CLOSE``, or ``_TRIGGER``."""
        if self.gone:
            raise _ClientGone
        try:
            return await self._call(self._owner.receive(self._next_request()))
        except lease.ShutdownObserved:
            return _TRIGGER

    async def _terminal(
        self, error: errors.FrontendError, phase: str
    ) -> errors.FrontendError:
        await surfaces.settle(self._owner, error, phase)
        return surfaces.selected_error(self._owner, error)

    async def _call(self, operation: Awaitable[T]) -> T:
        """One owner operation; a failure has already been cleaned up."""
        try:
            return await operation
        except errors.FrontendError:
            if self.gone:
                raise _ClientGone from None
            raise

    # --------------------------------------------------------------- flow

    # @spec ING-GRPC-002, ING-GRPC-016, ING-GRPC-019, ING-GRPC-020
    # @spec ING-GRPC-026, ING-GRPC-027, ING-GRPC-028, ING-GRPC-033
    # @spec ING-CORE-021, ING-LIFE-001, ING-LIFE-008, ING-LIFE-024
    # @spec ING-ERR-016, ING-VEH-007
    async def run(self) -> None:
        """Configure, stream and finalize; raise the terminal error."""
        owner = self._owner
        first = await self._read()
        if first is _TRIGGER:  # pragma: no cover - the owner ends it itself
            raise await self._terminal(
                errors.FrontendError("service_unavailable"), "shutdown"
            )
        if first is _HALF_CLOSE or not first.HasField("streaming_config"):
            raise await self._terminal(
                errors.FrontendError(
                    "protocol_order",
                    "the first message must carry streaming_config",
                ),
                "protocol",
            )
        locale, audio = await self._configure(first)
        await self._call(
            owner.open(cadence=lease.STREAM_CADENCE, locale=locale)
        )
        if audio is not None:
            owner.attach_frontend(audio)
        while True:
            message = await self._read()
            if message is _HALF_CLOSE or message is _TRIGGER:
                break
            if message.HasField("streaming_config"):
                raise await self._terminal(
                    errors.FrontendError(
                        "protocol_order",
                        "streaming_config is allowed only as the first message",
                    ),
                    "protocol",
                )
            if len(message.runtime_config):
                raise await self._terminal(
                    _field_error(
                        "invalid_config_field",
                        "runtime_config",
                        "the request-level rider is not supported",
                    ),
                    "protocol",
                )
            await self._audio(message.audio_content)
        if self._riff:
            raise await self._terminal(
                errors.FrontendError(
                    "unsupported_format",
                    "the stream ended inside its RIFF/WAVE header",
                    fields=("encoding",),
                ),
                "protocol",
            )
        text = await self._call(owner.finalize())
        await self._write(_streaming_response(text, final=True))

    async def _configure(
        self, message: rasr.StreamingRecognizeRequest
    ) -> tuple[str, frontend.StreamingAudioFrontend | None]:
        """Validate the first message completely.

        Returns the locale and, for a concrete cell, the stream's audio
        frontend (``None`` while a deferred header is still to resolve).
        """
        plugin = self._plugin
        factory = plugin.factory
        streaming = message.streaming_config
        config = streaming.config
        rejections = validate_recognition_config(
            config,
            served_model=plugin.served_model,
            locales=frozenset(factory.locales),
            default_locale=factory.default_locale,
        )
        if len(message.runtime_config):
            rejections.append(
                _field_error(
                    "invalid_config_field",
                    "runtime_config",
                    "the request-level rider is not supported",
                )
            )
        if rejections:
            raise await self._terminal(_primary(rejections), "protocol")
        self._interim = bool(streaming.interim_results)
        locale = config.language_code or str(factory.default_locale)
        if config.encoding == _UNSPECIFIED:
            self._riff = bytearray()
            self._declared_rate = config.sample_rate_hertz
            self._declared_channels = config.audio_channel_count
            return locale, None
        return locale, frontend.StreamingAudioFrontend(
            encoding=ENCODING_NAMES[config.encoding],
            sample_rate_hz=config.sample_rate_hertz,
        )

    # @spec ING-CORE-019
    async def _audio(self, data: bytes) -> None:
        """Decode one message in bounded slices and feed each in turn.

        Reads stay paused while a slice is fed. After the shutdown trigger
        no further slice is handed over: the stream is finalizing.
        """
        owner = self._owner
        if self._riff is not None:
            resolved = await self._resolve_header(data)
            if resolved is None:
                return
            data = resolved
        if self._remaining is not None:
            data = data[: self._remaining]
            self._remaining -= len(data)
        audio = owner.frontend
        if audio is None or not data:
            return
        bound = int(self._config.pre_submit_max_samples)
        for piece in audio.decode(data, max_samples=bound):
            if not len(piece):
                continue
            hypotheses = await self._call(owner.feed(piece))
            for text in hypotheses:
                if text == self._hypothesis:
                    continue
                self._hypothesis = text
                if self._interim:
                    await self._write(_streaming_response(text, final=False))
            if owner.shutdown_observed:
                return

    async def _resolve_header(self, data: bytes) -> bytes | None:
        """Buffer deferred-header bytes; return the audio once resolved."""
        assert self._riff is not None
        bound = int(self._config.max_riff_header_bytes)
        buffered = bytes(self._riff) + bytes(data)
        header = frontend.sniff_riff(buffered, max_header_bytes=bound)
        if header is None:
            self._riff = bytearray(buffered)
            return None
        problem = surfaces.header_problem(
            header, self._declared_rate, self._declared_channels
        )
        if problem is not None:
            raise await self._terminal(
                _format_error(problem, _RIVA_FORMAT_FIELDS), "protocol"
            )
        assert isinstance(header, frontend.RiffFormat)
        self._riff = None
        self._remaining = header.data_length
        self._owner.attach_frontend(
            frontend.StreamingAudioFrontend(
                encoding=header.encoding, sample_rate_hz=header.sample_rate_hz
            )
        )
        return buffered[header.data_offset :]


class RivaServicer(rasr_grpc.RivaSpeechRecognitionServicer):  # type: ignore[misc]
    """``nvidia.riva.asr.RivaSpeechRecognition`` over the plugin's seams."""

    def __init__(self, plugin: Any) -> None:
        """Serve ``plugin``'s factory, gate, registry, clock and catalog."""
        self._plugin = plugin

    # @spec ING-GRPC-001, ING-GRPC-015, ING-ERR-029, ING-ERR-042
    async def StreamingRecognize(  # noqa: N802 - the generated stub's name
        self,
        request_iterator: AsyncIterator[rasr.StreamingRecognizeRequest],
        context: Any,
    ) -> None:
        """One stream: admission, its owner, then the wire projection.

        Every terminal outcome is the owner's one selected cause, projected
        with ``context.abort``; a client cancellation produces no response,
        and a final is sent only after ``finish`` and ``release`` succeeded.
        A write or status beyond its send bound ends the call cancelled, for
        which ``grpc.aio`` sends no status: nothing more reaches the client.
        """
        try:
            await self._streaming(request_iterator, context)
        except _Abandoned:
            raise asyncio.CancelledError from None

    async def _streaming(
        self,
        request_iterator: AsyncIterator[rasr.StreamingRecognizeRequest],
        context: Any,
    ) -> None:
        plugin = self._plugin
        clock = plugin.clock
        rpc_deadline = _rpc_deadline(clock, context)
        if context.cancelled():
            return
        admitted = plugin.ingress.try_admit(admission.OwnerKind.stream)
        if admitted is None:
            await _abort(
                plugin,
                context,
                plugin.ingress.refusal(admission.OwnerKind.stream),
                rpc_deadline=rpc_deadline,
            )
        registry = plugin.registry
        record = registry.register(
            admission.OwnerKind.stream, asyncio.current_task()
        )
        stream: _RivaStream | None = None
        try:
            owner = lease.StreamOwner(
                lease_opener=plugin.factory.open,
                surface="riva",
                config=plugin.config,
                clock=clock,
                registry=registry,
                record=record,
                catalog=plugin.catalog,
            )
            owner.rpc_deadline = rpc_deadline
            watch = surfaces.watch_shutdown(registry, record, owner)
            stream = _RivaStream(
                plugin, owner, record, context, request_iterator, watch
            )
            stream.arm()
            try:
                await self._serve(stream, owner, context)
            except _Abandoned:
                # Cleaned up as for a disconnect; nothing further is sent.
                await surfaces.settle_shielded(
                    plugin, record, owner, _TRANSPORT_GONE, "transport"
                )
                raise
        finally:
            try:
                if stream is not None:
                    await stream.close()
            finally:
                surfaces.release_owner(registry, record, admitted)

    async def _serve(
        self, stream: _RivaStream, owner: Any, context: Any
    ) -> None:
        """Run the stream, then project its one terminal error, if any."""
        plugin = self._plugin
        error: errors.FrontendError | None = None
        try:
            await stream.run()
        except errors.FrontendError as raised:
            error = raised
        except _ClientGone:
            return
        except _Abandoned:
            raise
        except asyncio.CancelledError:
            error = await self._cancelled(stream, owner, context)
        except Exception as unexpected:  # a bug here, not the client's
            error = errors.FrontendError(
                "internal", f"unexpected failure: {unexpected!r}"
            )
            await surfaces.settle_shielded(
                plugin, stream.record, owner, error, "transport"
            )
            error = surfaces.selected_error(owner, error)
        if error is not None and not stream.gone:
            await stream.abort(error)

    async def _cancelled(
        self, stream: _RivaStream, owner: Any, context: Any
    ) -> errors.FrontendError | None:
        """The handler's task was cancelled: clean up, then decide the end.

        Cleanup is joined through any further cancellation, so the record
        and admission outlive it (unless ``stop`` tore the plugin down). A
        client cancellation re-raises with no response. Otherwise the plugin
        cancelled the owner (``drain`` within the engine reserve, or
        ``stop``): the stream ends with the shutdown abort row.
        """
        shutdown = errors.FrontendError(
            "service_unavailable", "the server is shutting down"
        )
        await surfaces.settle_shielded(
            self._plugin, stream.record, owner, shutdown, "shutdown"
        )
        if context.cancelled() or stream.gone:
            raise asyncio.CancelledError
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            task.uncancel()
        return surfaces.selected_error(owner, shutdown)

    # @spec ING-GRPC-001, ING-GRPC-010, ING-GRPC-029, ING-GRPC-021
    # @spec ING-GRPC-032, ING-FILE-014, ING-ADM-017, ING-ADM-011
    async def Recognize(  # noqa: N802 - the generated stub's name
        self, request: rasr.RecognizeRequest, context: Any
    ) -> rasr.RecognizeResponse:
        """Validate and convert a complete request, then run it as a file.

        Every validation, the encoded-size and decoded-duration bounds and
        header inference finish before a file-gate permit is taken; at each
        boundary a cancellation or an already expired RPC deadline wins. A
        status beyond its send bound ends the call cancelled, with none.
        """
        try:
            return await self._recognize(request, context)
        except _Abandoned:
            raise asyncio.CancelledError from None

    async def _recognize(
        self, request: rasr.RecognizeRequest, context: Any
    ) -> rasr.RecognizeResponse:
        plugin = self._plugin
        clock = plugin.clock
        deadline = _rpc_deadline(clock, context)
        await self._unary_boundary(context, deadline)
        if not plugin.files_enabled:
            await _abort(
                plugin,
                context,
                errors.FrontendError(
                    "unsupported_capability",
                    "complete files are not served by this deployment",
                ),
                rpc_deadline=deadline,
            )
        kind = admission.OwnerKind.recognize
        admitted = plugin.ingress.try_admit(kind)
        if admitted is None:
            await _abort(
                plugin,
                context,
                plugin.ingress.refusal(kind),
                rpc_deadline=deadline,
            )
        registry = plugin.registry
        record = registry.register(kind, asyncio.current_task())
        handed_over = False
        try:
            try:
                locale, wav = self._prepare(request)
            except errors.FrontendError as rejected:
                await self._unary_boundary(context, deadline)
                await _abort(plugin, context, rejected, rpc_deadline=deadline)
            await self._unary_boundary(context, deadline)
            from vllm_riva_frontend import files  # the files leaf's path

            # From here the file owner releases the admission and the record
            # exactly once, from its cleanup continuation if that outlasts
            # the call.
            handed_over = True
            try:
                result = await files.recognize_file(
                    plugin,
                    wav,
                    locale,
                    context=context,
                    deadline=deadline,
                    admission=admitted,
                    record=record,
                )
            except errors.FrontendError as failed:
                await _abort(plugin, context, failed, rpc_deadline=deadline)
        finally:
            if not handed_over:
                surfaces.unregister(registry, record)
                admitted.release()
        if isinstance(result, rasr.RecognizeResponse):
            return result
        return _recognize_response(str(result))

    async def _unary_boundary(
        self, context: Any, deadline: float | None
    ) -> None:
        if context.cancelled():
            raise asyncio.CancelledError
        if deadline is not None and self._plugin.clock.monotonic() >= deadline:
            await _abort(
                self._plugin,
                context,
                errors.FrontendError(
                    "request_timeout", "the RPC deadline expired"
                ),
                rpc_deadline=deadline,
            )

    def _prepare(self, request: rasr.RecognizeRequest) -> tuple[str, bytes]:
        """The locale and the 16 kHz mono WAV of a valid request.

        Sizes are checked on the encoded bytes and the fixed-width cell
        before anything is decoded, so conversion never makes an unbounded
        second copy; nothing touches the filesystem.
        """
        plugin = self._plugin
        config = request.config
        settings = plugin.config
        factory = plugin.factory
        rejections = validate_recognition_config(
            config,
            served_model=plugin.served_model,
            locales=frozenset(factory.locales),
            default_locale=factory.default_locale,
        )
        if rejections:
            raise _primary(rejections)
        locale = config.language_code or str(factory.default_locale)
        audio = memoryview(request.audio)
        limit = int(settings.unary_max_encoded_audio_bytes)
        if len(audio) > limit:
            raise errors.FrontendError(
                "request_too_large",
                f"the encoded audio is {len(audio)} bytes; limit {limit}",
                fields=("audio",),
            )
        if config.encoding == _UNSPECIFIED:
            encoding, rate, audio = _unary_header(
                audio,
                bound=int(settings.max_riff_header_bytes),
                declared_rate=config.sample_rate_hertz,
                declared_channels=config.audio_channel_count,
            )
        else:
            encoding = ENCODING_NAMES[config.encoding]
            rate = config.sample_rate_hertz
        seconds = frontend.duration_from_bytes(encoding, rate, len(audio))
        maximum = float(settings.unary_max_decoded_duration_seconds)
        if seconds > maximum:
            raise errors.FrontendError(
                "request_too_large",
                f"the audio lasts {seconds:.3f} s; the limit is {maximum:g} s",
                fields=("audio",),
            )
        samples = _decode_complete(
            audio, encoding, rate, int(settings.pre_submit_max_samples)
        )
        # The accessor's copy of the encoded audio is not needed once it is
        # decoded; dropping it keeps it and the WAV from being held together.
        del audio
        return locale, frontend.encode_wav16(samples)

    # @spec ING-GRPC-001, ING-GRPC-014
    async def GetRivaSpeechRecognitionConfig(  # noqa: N802 - the stub's name
        self, request: rasr.RivaSpeechRecognitionConfigRequest, context: Any
    ) -> rasr.RivaSpeechRecognitionConfigResponse:
        """The static description of the served model (``config_response``)."""
        del request, context
        return config_response(self._plugin)


def _decode_complete(
    audio: memoryview, encoding: str, rate: int, slice_samples: int
) -> np.ndarray:
    """Decode a complete request into one preallocated 16 kHz array.

    The audio frontend decodes in bounded slices, each copied into place,
    so the decoded audio exists once (no list of slices joined afterwards).
    Resampling 8 kHz doubles the sample count; the drained tail is part of
    that count.
    """
    decoder = frontend.StreamingAudioFrontend(
        encoding=encoding, sample_rate_hz=rate
    )
    count = frontend.duration_from_bytes(encoding, rate, len(audio)) * rate
    capacity = int(round(count)) * (16_000 // rate) + 64
    samples = np.empty(capacity, dtype=np.float32)
    filled = 0

    def place(piece: np.ndarray) -> None:
        nonlocal samples, filled
        end = filled + len(piece)
        if end > samples.size:  # pragma: no cover - the count is exact
            samples = np.concatenate([samples[:filled], piece])
        else:
            samples[filled:end] = piece
        filled = end

    for piece in decoder.decode(audio, max_samples=slice_samples):
        place(piece)
    place(decoder.drain())
    return samples[:filled]


def _unary_header(
    audio: memoryview,
    *,
    bound: int,
    declared_rate: int,
    declared_channels: int,
) -> tuple[str, int, memoryview]:
    """Resolve, validate and strip a complete request's RIFF/WAVE header."""
    header = frontend.sniff_riff(audio, max_header_bytes=bound)
    if header is None:
        header = frontend.FormatError(
            ("encoding",), "the request ends inside its RIFF/WAVE header"
        )
    problem = surfaces.header_problem(header, declared_rate, declared_channels)
    if problem is not None:
        raise _format_error(problem, _RIVA_FORMAT_FIELDS)
    assert isinstance(header, frontend.RiffFormat)
    payload = audio[header.data_offset :]
    if header.data_length is not None:
        if len(payload) < header.data_length:
            raise errors.FrontendError(
                "invalid_audio",
                f"the RIFF data chunk declares {header.data_length} bytes and "
                f"the request carries {len(payload)}",
                fields=("audio",),
            )
        payload = payload[: header.data_length]
    return header.encoding, header.sample_rate_hz, payload


__all__ = [
    "ENCODING_NAMES",
    "FIELD_DISPOSITIONS",
    "RIVA_SERVICE_NAME",
    "RivaServicer",
    "build_server",
    "config_response",
    "register_aio_services",
    "validate_recognition_config",
]
