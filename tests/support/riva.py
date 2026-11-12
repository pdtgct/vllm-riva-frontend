"""The Riva gRPC wire, driven directly on the servicer.

Message builders over the pinned public stubs (``riva.client.proto``) and
``StreamingCall``, the one driver of ``StreamingRecognize`` every leaf uses.
The handler is a coroutine that writes each response with ``context.write``
(implementation layout, Shapes), so the driver awaits it in a task and
records what it wrote and how it ended: ``RpcAbort`` from
``FakeServicerContext.abort`` is the projected gRPC status; a handler that
ends cancelled on its own is ``abandoned`` (``grpc.aio`` sends no status for
it); anything else the handler raises is kept on ``error`` and surfaced by
``wait``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import grpc
from riva.client.proto import riva_asr_pb2 as rasr
from riva.client.proto import riva_audio_pb2 as raudio

from support.fake_engine import DEFAULT_MODEL
from support.grpc_harness import FakeServicerContext, RequestStream, RpcAbort

LINEAR_PCM = raudio.AudioEncoding.Value("LINEAR_PCM")
MULAW = raudio.AudioEncoding.Value("MULAW")
ALAW = raudio.AudioEncoding.Value("ALAW")
UNSPECIFIED = raudio.AudioEncoding.Value("ENCODING_UNSPECIFIED")
FLAC = raudio.AudioEncoding.Value("FLAC")
OGGOPUS = raudio.AudioEncoding.Value("OGGOPUS")

#: Loop turns that let a cancel, its settlement and the cleanup calls run.
SETTLE = 60


def encoding_value(encoding: str | int) -> int:
    """The proto enum value for an encoding name (or a value, unchanged)."""
    if isinstance(encoding, str):
        return raudio.AudioEncoding.Value(encoding)
    return encoding


# ----------------------------------------------------------------- messages


def recognition_config(
    *,
    encoding: str | int = "LINEAR_PCM",
    sample_rate_hertz: int = 16_000,
    language_code: str = "en-US",
    audio_channel_count: int = 1,
    max_alternatives: int = 1,
    **fields: Any,
) -> rasr.RecognitionConfig:
    """A concrete ``RecognitionConfig``: PCM16, 16 kHz, mono, ``en-US``.

    Keyword names are the proto field names; ``encoding`` may be the enum
    name or its value. ``model`` defaults to empty (the served identity is
    also accepted).
    """
    return rasr.RecognitionConfig(
        encoding=encoding_value(encoding),
        sample_rate_hertz=sample_rate_hertz,
        language_code=language_code,
        audio_channel_count=audio_channel_count,
        max_alternatives=max_alternatives,
        **fields,
    )


def config_request(
    config: rasr.RecognitionConfig | None = None,
    *,
    interim_results: bool = True,
    runtime_config: dict[str, str] | None = None,
    **fields: Any,
) -> rasr.StreamingRecognizeRequest:
    """The first message of a stream: ``streaming_config``.

    ``fields`` build the ``RecognitionConfig`` when ``config`` is not given;
    ``runtime_config`` fills the request-level rider.
    """
    request = rasr.StreamingRecognizeRequest(
        streaming_config=rasr.StreamingRecognitionConfig(
            config=config
            if config is not None
            else recognition_config(**fields),
            interim_results=interim_results,
        )
    )
    if runtime_config:
        request.runtime_config.update(runtime_config)
    return request


def audio_request(data: bytes) -> rasr.StreamingRecognizeRequest:
    return rasr.StreamingRecognizeRequest(audio_content=data)


def recognize_request(
    audio: bytes, config: rasr.RecognitionConfig | None = None, **fields: Any
) -> rasr.RecognizeRequest:
    """A unary ``RecognizeRequest`` over ``audio``."""
    return rasr.RecognizeRequest(
        config=config if config is not None else recognition_config(**fields),
        audio=audio,
    )


def config_rpc_request() -> rasr.RivaSpeechRecognitionConfigRequest:
    return rasr.RivaSpeechRecognitionConfigRequest()


def catalog_code(details: str | None) -> str | None:
    """The catalog code leading a status-details string, or ``None``."""
    if not details:
        return None
    return details.split(":", 1)[0].strip()


def transcripts(responses: Iterable[Any], *, final: bool) -> list[str]:
    """The transcript of every result with the given ``is_final``."""
    return [
        result.alternatives[0].transcript if result.alternatives else ""
        for response in responses
        for result in response.results
        if bool(result.is_final) is final
    ]


# ------------------------------------------------------------------ driver


def _cancel_requested() -> bool:
    """Whether the running task was cancelled (not merely ended so)."""
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


class StreamingCall:
    """One ``StreamingRecognize`` call on a servicer, driven by the test.

    ``start()`` runs the handler in a task over a ``RequestStream`` and a
    ``FakeServicerContext``; the constructor starts it unless
    ``start=False``. ``configure``, ``audio`` and ``push`` queue request
    messages, ``half_close`` ends the request stream, ``cancel`` is the
    client's cancellation. ``wait`` awaits the handler and raises if it
    ended with anything but a normal return or an ``RpcAbort``; ``finish``
    is ``half_close`` then ``wait``.

    ``status`` is the abort's status, else the code the handler set, else
    ``OK``; ``code`` is the catalog code leading ``details``; ``responses``
    are what the handler wrote (the context's ``writes``); ``finals`` and
    ``interims`` are transcripts. ``abandoned`` is set when the handler
    ended cancelled without the test cancelling it: no status at all.
    ``settle`` and ``advance`` need ``clock``, which is also the clock of the
    context's RPC deadline (``time_remaining`` seconds from the call's
    construction).
    """

    def __init__(
        self,
        servicer: Any,
        clock: Any = None,
        *,
        messages: Iterable[Any] = (),
        time_remaining: float | None = None,
        metadata: Iterable[tuple[str, str]] = (),
        start: bool = True,
    ) -> None:
        self.servicer = servicer
        self.clock = clock
        self.requests = RequestStream(messages)
        self.context = FakeServicerContext(
            time_remaining=time_remaining, clock=clock, metadata=metadata
        )
        self.responses: list[Any] = self.context.writes
        self.abort: RpcAbort | None = None
        self.abandoned = False
        self.error: BaseException | None = None
        self.task: asyncio.Task[None] | None = None
        if start:
            self.start()

    # -- driving --------------------------------------------------------------
    def start(self) -> StreamingCall:
        if self.task is None:
            self.task = asyncio.create_task(self._run())
        return self

    async def _run(self) -> None:
        try:
            await self.servicer.StreamingRecognize(self.requests, self.context)
        except RpcAbort as exc:
            self.abort = exc
        except asyncio.CancelledError:
            if _cancel_requested():
                raise
            self.abandoned = True
        except BaseException as exc:  # kept for ``wait`` to report
            self.error = exc

    def push(self, message: Any) -> None:
        self.requests.push(message)

    def configure(
        self,
        config: rasr.RecognitionConfig | None = None,
        *,
        interim_results: bool = True,
        runtime_config: dict[str, str] | None = None,
        **fields: Any,
    ) -> None:
        """Queue the ``streaming_config`` message (see ``config_request``)."""
        self.push(
            config_request(
                config,
                interim_results=interim_results,
                runtime_config=runtime_config,
                **fields,
            )
        )

    def audio(self, data: bytes) -> None:
        self.push(audio_request(data))

    def half_close(self) -> None:
        self.requests.close()

    def cancel(self) -> None:
        """The client cancelled the RPC (fires the context's callbacks)."""
        self.context.cancel()

    async def settle(self, rounds: int = SETTLE) -> None:
        if self.clock is not None:
            await self.clock.settle(rounds)
        else:
            for _ in range(rounds):
                await asyncio.sleep(0)

    async def advance(self, seconds: float) -> None:
        assert self.clock is not None, "advance needs the plugin's clock"
        await self.clock.advance(seconds, settle=SETTLE)

    async def wait(self, timeout: float = 5.0) -> StreamingCall:
        """Await the handler's end; raise if it raised something unexpected."""
        assert self.task is not None, "the call was not started"
        await asyncio.wait_for(asyncio.shield(self.task), timeout)
        if self.error is not None:
            raise AssertionError(
                f"handler raised {self.error!r}"
            ) from self.error
        return self

    async def finish(self, timeout: float = 5.0) -> StreamingCall:
        """Half-close the request stream, then ``wait``."""
        self.half_close()
        return await self.wait(timeout)

    async def until(
        self,
        predicate: Any,
        *,
        timeout: float = 2.0,
        what: str = "",
    ) -> None:
        """Yield to the loop until ``predicate()`` holds or the call ends."""
        loop = asyncio.get_running_loop()
        end = loop.time() + timeout
        while not predicate() and not self.done:
            if loop.time() > end:
                raise AssertionError(
                    f"timed out waiting for {what or predicate}"
                )
            await asyncio.sleep(0.001)

    # -- reading --------------------------------------------------------------
    @property
    def done(self) -> bool:
        return self.task is not None and self.task.done()

    @property
    def status(self) -> grpc.StatusCode:
        if self.abort is not None:
            return self.abort.code
        if self.context.code is not None:
            return self.context.code
        return grpc.StatusCode.OK

    @property
    def details(self) -> str:
        if self.abort is not None:
            return self.abort.details
        return self.context.details or ""

    @property
    def code(self) -> str | None:
        """The catalog code the client saw, or ``None`` on a normal end."""
        return catalog_code(self.details)

    @property
    def finals(self) -> list[str]:
        return transcripts(self.responses, final=True)

    @property
    def interims(self) -> list[str]:
        return transcripts(self.responses, final=False)

    @property
    def final_responses(self) -> list[Any]:
        return [r for r in self.responses if any(x.is_final for x in r.results)]

    @property
    def interim_responses(self) -> list[Any]:
        return [
            r
            for r in self.responses
            if r.results and not any(x.is_final for x in r.results)
        ]

    # -- asserting ------------------------------------------------------------
    def assert_ok(self) -> None:
        assert self.abort is None, (self.status, self.details)
        assert self.status is grpc.StatusCode.OK, (self.status, self.details)

    def assert_status(
        self, status: grpc.StatusCode, code: str | None = None, *names: str
    ) -> None:
        """Assert the status, the leading catalog code and named fields."""
        assert self.status is status, (self.status, self.details)
        if code is not None:
            assert self.details.startswith(f"{code}:"), self.details
        for name in names:
            assert name in self.details, (name, self.details)

    def assert_rejected(
        self, status: grpc.StatusCode, code: str | None = None, *names: str
    ) -> None:
        """``assert_status`` and no final was emitted."""
        self.assert_status(status, code, *names)
        assert not self.finals, self.finals


# ------------------------------------------------------------- unary calls


@dataclass
class RecognizeOutcome:
    """What one ``Recognize`` call ended with."""

    response: Any = None
    abort: RpcAbort | None = None
    error: BaseException | None = None
    context: FakeServicerContext | None = None
    #: The handler ended cancelled on its own: no response and no status.
    abandoned: bool = False

    @property
    def status(self) -> grpc.StatusCode:
        if self.abort is not None:
            return self.abort.code
        if self.context is not None and self.context.code is not None:
            return self.context.code
        return grpc.StatusCode.OK

    @property
    def details(self) -> str:
        return "" if self.abort is None else self.abort.details

    @property
    def code(self) -> str | None:
        return catalog_code(self.details)

    @property
    def transcript(self) -> str | None:
        """The one alternative's transcript, or ``None`` without a response."""
        if self.response is None:
            return None
        (result,) = self.response.results
        (alternative,) = result.alternatives
        return alternative.transcript


async def recognize(
    servicer: Any,
    request: rasr.RecognizeRequest,
    context: FakeServicerContext | None = None,
) -> RecognizeOutcome:
    """Call ``Recognize`` and classify how it ended."""
    context = context if context is not None else FakeServicerContext()
    outcome = RecognizeOutcome(context=context)
    try:
        outcome.response = await servicer.Recognize(request, context)
    except RpcAbort as abort:
        outcome.abort = abort
    except asyncio.CancelledError:
        if _cancel_requested():
            raise
        outcome.abandoned = True
    except BaseException as exc:  # a bug in the handler, for the test
        outcome.error = exc
    return outcome


__all__ = [
    "ALAW",
    "DEFAULT_MODEL",
    "FLAC",
    "LINEAR_PCM",
    "MULAW",
    "OGGOPUS",
    "SETTLE",
    "UNSPECIFIED",
    "RecognizeOutcome",
    "StreamingCall",
    "audio_request",
    "catalog_code",
    "config_request",
    "config_rpc_request",
    "encoding_value",
    "rasr",
    "raudio",
    "recognition_config",
    "recognize",
    "recognize_request",
    "transcripts",
]
