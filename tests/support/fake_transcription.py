"""A stand-in for core's transcription serving class.

Core's route (``vllm/entrypoints/speech_to_text/transcription/api_router.py``
lines 42-61) reads ``request.app.state.openai_serving_transcription``, reads
the upload with ``read_upload_with_limit(request.file)`` and calls
``handler.create_transcription(audio_data, request, raw_request)``; it returns
a JSON response for a ``TranscriptionResponseVariant``, an error response with
``error.code`` as status for an ``ErrorResponse``, and a ``StreamingResponse``
(``text/event-stream``) for an async generator.

``FakeTranscriptionServing.create_transcription`` has the signature of
``OpenAIServingTranscription.create_transcription``
(``.../transcription/serving.py:49-61``):
``(audio_data: bytes, request: TranscriptionRequest, raw_request: Request |
None = None)``. It derives the engine request ID as core's
``_create_speech_to_text`` does (``.../base/serving.py:485``):
``f"{task_type}-{_base_request_id(raw_request)}"`` with ``task_type``
``"transcribe"`` and core's own ``_base_request_id``
(``vllm/entrypoints/serve/engine/serving.py:133-142``), which takes the
``X-Request-Id`` header when present. It records the call, then returns,
blocks or raises as scripted. The other attributes are the ones the frontend
reads: ``models`` (``model_name()``), ``model_cls.supported_languages``
(``vllm/model_executor/models/interfaces.py:1343``), ``asr_config``
(``max_audio_clip_s``, ``sample_rate``), ``max_audio_decode_duration_s`` and
``default_sampling_params`` (``.../base/serving.py:114``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from http import HTTPStatus
from types import SimpleNamespace
from typing import Any

from fastapi import Request
from vllm import envs
from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.serve.engine.protocol import ErrorInfo, ErrorResponse
from vllm.entrypoints.serve.engine.serving import BaseServing
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionUsageAudio,
)

from support.fake_session import Behaviour, Block, Raise, Return, _perform

TASK_TYPE = "transcribe"


class FakeServingModels:
    """The part of core's ``OpenAIServingModels`` the file path uses.

    ``model_name()`` returns the first base model's name
    (``vllm/entrypoints/openai/models/serving.py:144-147``).
    """

    def __init__(self, *names: str) -> None:
        if not names:
            names = ("nvidia/nemotron-asr-streaming-test",)
        self.base_model_paths = [
            SimpleNamespace(name=name, model_path=names[0]) for name in names
        ]
        self.model_name_calls = 0

    def model_name(self, lora_request: Any = None) -> str:
        self.model_name_calls += 1
        return self.base_model_paths[0].name

    def is_base_model(self, model_name: str) -> bool:
        return any(m.name == model_name for m in self.base_model_paths)

    async def check_model(self, model_name: str | None) -> ErrorResponse | None:
        if not model_name or self.is_base_model(model_name):
            return None
        return error_response(
            f"The model `{model_name}` does not exist.",
            HTTPStatus.NOT_FOUND,
            err_type="NotFoundError",
        )


def transcription_response(
    text: str, *, seconds: int = 0
) -> TranscriptionResponse:
    """A complete response in core's shape (``text`` and audio ``usage``)."""
    return TranscriptionResponse(
        text=text, usage=TranscriptionUsageAudio(seconds=seconds)
    )


def error_response(
    message: str,
    status: HTTPStatus | int = HTTPStatus.BAD_REQUEST,
    *,
    param: str | None = None,
    err_type: str = "BadRequestError",
) -> ErrorResponse:
    """An ``ErrorResponse`` in core's shape.

    See ``vllm/entrypoints/serve/engine/protocol.py:79-87``.
    """
    return ErrorResponse(
        error=ErrorInfo(
            message=message, type=err_type, param=param, code=int(status)
        )
    )


class EventStream:
    """A scriptable ``text/event-stream`` generator.

    Yields ``events`` in order; an event at an index in ``blocks`` waits for
    that ``Block`` to be released first (``then`` may be a ``Raise``).
    Records every pull on ``pulls`` and whether ``aclose`` was called.
    """

    def __init__(
        self,
        events: Iterable[str],
        *,
        blocks: Mapping[int, Block] | None = None,
    ) -> None:
        self.events = list(events)
        self.blocks = dict(blocks or {})
        self.pulls = 0
        self.closed = False
        self._index = 0

    def __aiter__(self) -> EventStream:
        return self

    async def __anext__(self) -> str:
        self.pulls += 1
        if self.closed or self._index >= len(self.events):
            raise StopAsyncIteration
        block = self.blocks.get(self._index)
        if block is not None:
            await _perform(block, None)
        event = self.events[self._index]
        self._index += 1
        return event

    async def aclose(self) -> None:
        self.closed = True


def event_stream(*events: str, blocks: Mapping[int, Block] | None = None):
    return EventStream(events, blocks=blocks)


def sse(payload: str | Mapping[str, Any]) -> str:
    """One server-sent event line pair as core's generator emits them.

    A mapping is serialized as JSON first, as core does with its models.
    """
    if not isinstance(payload, str):
        payload = json.dumps(dict(payload))
    return f"data: {payload}\n\n"


@dataclass
class TranscriptionCall:
    """One recorded ``create_transcription`` call."""

    audio_data: bytes
    request: TranscriptionRequest
    raw_request: Request | None
    request_id: str
    headers: dict[str, str] = field(default_factory=dict)


class FakeTranscriptionServing:
    """See the module docstring."""

    task_type = TASK_TYPE

    def __init__(
        self,
        models: FakeServingModels | None = None,
        *,
        supported_languages: Mapping[str, str] | None = None,
        max_audio_clip_s: int | None = None,
        sample_rate: int = 16_000,
        default_text: str = "",
        default_sampling_params: Mapping[str, Any] | None = None,
    ) -> None:
        self.models = models or FakeServingModels()
        #: The served model's generation settings, as core's serving keeps
        #: them (``model_config.get_diff_sampling_param()`` at
        #: ``.../speech_to_text/base/serving.py:114``). Empty by default.
        self.default_sampling_params: dict[str, Any] = dict(
            default_sampling_params or {}
        )
        self.model_cls = SimpleNamespace(
            supported_languages=dict(
                supported_languages
                if supported_languages is not None
                else {"en": "english", "de": "german"}
            ),
            supports_explicit_language_detection=False,
        )
        self.asr_config = SimpleNamespace(
            max_audio_clip_s=max_audio_clip_s, sample_rate=sample_rate
        )
        self.max_audio_decode_duration_s = envs.VLLM_MAX_AUDIO_DECODE_DURATION_S
        self.default_text = default_text
        self.calls: list[TranscriptionCall] = []
        self.shutdown_calls = 0
        self._behaviours: list[Behaviour] = []

    def script(self, *behaviours: Behaviour) -> None:
        """Queue behaviours for the next calls.

        ``Return(x)`` returns ``x`` (a response, an ``ErrorResponse`` or an
        ``EventStream``); ``Block()`` parks the call; ``Raise(exc)`` raises.
        """
        self._behaviours.extend(behaviours)

    @property
    def request_ids(self) -> list[str]:
        return [call.request_id for call in self.calls]

    async def create_transcription(
        self,
        audio_data: bytes,
        request: TranscriptionRequest,
        raw_request: Request | None = None,
    ) -> Any:
        base = BaseServing._base_request_id(raw_request)
        request_id = f"{self.task_type}-{base}"
        headers = dict(raw_request.headers) if raw_request is not None else {}
        if raw_request is not None:
            raw_request.state.request_metadata = RequestResponseMetadata(
                request_id=request_id
            )
        self.calls.append(
            TranscriptionCall(
                audio_data, request, raw_request, request_id, headers
            )
        )
        behaviour = self._behaviours.pop(0) if self._behaviours else Return()
        return await _perform(
            behaviour, transcription_response(self.default_text)
        )

    def shutdown(self) -> None:
        self.shutdown_calls += 1


__all__ = [
    "Block",
    "EventStream",
    "FakeServingModels",
    "FakeTranscriptionServing",
    "Raise",
    "Return",
    "TranscriptionCall",
    "error_response",
    "event_stream",
    "sse",
    "transcription_response",
]
