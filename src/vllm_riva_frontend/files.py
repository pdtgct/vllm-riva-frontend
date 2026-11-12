"""Complete files: the gate, the guard and the owner of a file request.

Both complete-file entry points, the Riva ``Recognize`` RPC and the
frontend's ``POST /plugins/riva/v1/audio/transcriptions``, which sits beside
core's own route and leaves it unchanged, run on core's transcription service
(``state.openai_serving_transcription``) behind one in-flight file gate and
one request guard. This module is what the frontend adds in front of that
service:

* ``FileGate``: the counting gate of ``max_concurrent_files``;
* ``RequestGuard``: refuses what the model's single-slot file path cannot
  honour (beam search, completion limits, non-default sampling, ignored
  fields, foreign ``vllm_xargs`` keys) before a permit is taken;
* ``FileRequestOwner``: the owner of one file request. It takes the permit,
  calls ``create_transcription`` in a child task it never cancels, advances
  core's event generator one pull at a time, records one terminal cause,
  aborts the fixed engine request ID itself until the call has settled and
  releases the permit, the file-owner slot and the server-load count exactly
  once from a cleanup continuation registered with its record;
* ``file_route``: the handler of the frontend's file route;
* ``recognize_file``: the ``Recognize`` execution path.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import io
import json
import logging
import typing
import uuid
import wave
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import grpc
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError
from riva.client.proto import riva_asr_pb2 as rasr
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.types import Message, Receive, Scope, Send

from vllm_riva_frontend import surfaces
from vllm_riva_frontend.admission import OwnerKind
from vllm_riva_frontend.errors import (
    FrontendError,
    classify_core_exception,
    project_error_response,
)
from vllm_riva_frontend.errors import catalog as default_catalog
from vllm_riva_frontend.multipart import (
    MultipartLimits,
    _memory_upload,
    make_upload,
    parse_content_length,
    parse_content_type,
    parse_multipart,
)

__all__ = [
    "FORWARDED_SAMPLING_FIELDS",
    "IGNORED_FIELDS",
    "REQUEST_FIELD_ALLOWLIST",
    "FileCause",
    "FileGate",
    "FileRequestOwner",
    "Permit",
    "RequestGuard",
    "TranscriptionCall",
    "build_raw_request",
    "file_route",
    "locale_language",
    "recognize_file",
    "recognize_language",
    "wav_seconds",
]

_LOG = logging.getLogger(__name__)

#: Seconds between the repeated aborts of an unsettled engine request.
ABORT_PERIOD = 0.25

#: The names and aliases of core's ``TranscriptionRequest`` fields. An
#: explicit list, so a field core adds is refused until it is reviewed.
REQUEST_FIELD_ALLOWLIST = frozenset(
    {
        "file",
        "model",
        "language",
        "hotwords",
        "prompt",
        "response_format",
        "timestamp_granularities",
        "timestamp_granularities[]",
        "stream",
        "stream_include_usage",
        "stream_continuous_usage_stats",
        "vllm_xargs",
        "to_language",
        "use_beam_search",
        "n",
        "length_penalty",
        "include_stop_str_in_output",
        "temperature",
        "watermarking",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "frequency_penalty",
        "repetition_penalty",
        "presence_penalty",
        "max_completion_tokens",
    }
)

#: The request fields core forwards into the engine's sampling parameters
#: (``TranscriptionRequest.to_sampling_params``), apart from the completion
#: limit and ``vllm_xargs``, which the guard handles on their own.
FORWARDED_SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "seed",
    "watermarking",
)

#: Fields core accepts on the non-beam path but never forwards.
IGNORED_FIELDS = ("n", "length_penalty", "include_stop_str_in_output")

_ABSENT = object()

#: Core's field defaults: an explicit value equal to one of these is
#: accepted; ``_ABSENT`` means any explicit value is non-default.
_FIELD_DEFAULTS: dict[str, object] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "seed": _ABSENT,
    "watermarking": _ABSENT,
    "n": 1,
    "length_penalty": 1.0,
    "include_stop_str_in_output": False,
}

_XARGS = "vllm_xargs"
_LOCALE_KEY = "locale"
_REQUEST_ID_HEADER = b"x-request-id"
_TASK_TYPE = "transcribe"
_EVENT_STREAM = "text/event-stream"


# ------------------------------------------------------------------ gate


class Permit:
    """One slot of the file gate, released exactly once."""

    def __init__(self, gate: FileGate) -> None:
        """Hold one slot of ``gate``."""
        self._gate: FileGate | None = gate

    @property
    def released(self) -> bool:
        """Whether the slot was given back."""
        return self._gate is None

    def release(self) -> None:
        """Give the slot back; a repeated call does nothing."""
        gate, self._gate = self._gate, None
        if gate is not None:
            gate._release_slot()


# @spec ING-FILE-080, ING-FILE-019, ING-FILE-053, ING-EP-082
class FileGate:
    """The counting gate of complete files in flight.

    One gate covers ``Recognize`` and the file route. Waiters wait here, in
    the frontend, as owner tasks, in arrival order; a released slot passes
    straight to the oldest waiter. The gate caps files in flight and
    reserves nothing for streams. It refuses new waiters once it is closed
    (the plugin's ``drain`` or a recorded fault) and, read directly, while
    ``serving()`` is false (core has set ``state.draining``).
    """

    def __init__(
        self,
        size: int,
        *,
        clock: Any,
        serving: Callable[[], bool] | None = None,
    ) -> None:
        """Create an open gate of ``size`` slots on ``clock``.

        ``serving`` reports whether core still serves (``state.draining``
        unset); without it only ``close`` closes the gate.
        """
        self._size = size
        self._clock = clock
        self._serving = serving
        self._in_flight = 0
        self._closed = False
        self._waiters: deque[asyncio.Future[None]] = deque()

    @property
    def size(self) -> int:
        """The number of slots."""
        return self._size

    @property
    def in_flight(self) -> int:
        """Slots held now."""
        return self._in_flight

    @property
    def waiting(self) -> int:
        """Waiters not yet granted a slot."""
        return sum(1 for waiter in self._waiters if not waiter.done())

    @property
    def closed(self) -> bool:
        """Whether the gate refuses new waiters (ING-FILE-053, ING-EP-082)."""
        serving = self._serving
        return self._closed or (serving is not None and not serving())

    def close(self) -> None:
        """Refuse new waiters; waiters already queued keep their place."""
        self._closed = True

    async def acquire(
        self,
        deadline: float,
        *,
        spawn: Callable[[Coroutine[Any, Any, Any]], asyncio.Future[Any]]
        | None = None,
    ) -> Permit:
        """Wait for a slot until the absolute ``deadline``.

        The wait's timer is a ``clock.sleep`` task started by ``spawn``, so
        an owner keeps it under its registry record (ING-EP-106).

        Raises:
            FrontendError: ``service_unavailable`` on a closed gate; ``busy``
                (``authority: load_shed``) when ``deadline`` passes first.
        """
        if self.closed:
            raise FrontendError(
                "service_unavailable", "the server is shutting down"
            )
        if self._in_flight < self._size and not self.waiting:
            self._in_flight += 1
            return Permit(self)
        waiter: asyncio.Future[None] = (
            asyncio.get_running_loop().create_future()
        )
        self._waiters.append(waiter)
        try:
            remaining = deadline - self._clock.monotonic()
            if remaining > 0:
                timer = (spawn or asyncio.ensure_future)(
                    self._clock.sleep(remaining)
                )
                try:
                    await asyncio.wait(
                        {waiter, timer}, return_when=asyncio.FIRST_COMPLETED
                    )
                finally:
                    timer.cancel()
            if not waiter.done():
                waiter.cancel()
                raise FrontendError(
                    "busy",
                    "no complete-file slot freed within file_gate_wait_timeout",
                    authority="load_shed",
                )
        except BaseException:
            if waiter.done() and not waiter.cancelled():
                # Granted as the wait was abandoned: pass the slot on.
                self._release_slot()
            raise
        finally:
            with contextlib.suppress(ValueError):
                self._waiters.remove(waiter)
        return Permit(self)

    def _release_slot(self) -> None:
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.set_result(None)
                return
        self._in_flight -= 1


# ----------------------------------------------------------------- guard


def _refusal(name: str, detail: str) -> FrontendError:
    return FrontendError("invalid_config_field", detail, (name,))


def _xargs_error(request: Any, locales: Iterable[str]) -> FrontendError | None:
    """Check a client ``vllm_xargs``: only the key ``locale``, a served one."""
    if _XARGS not in request.model_fields_set:
        return None
    xargs = getattr(request, _XARGS, None)
    if xargs is None:
        return None
    if set(xargs) != {_LOCALE_KEY}:
        return _refusal(_XARGS, "vllm_xargs accepts only the key 'locale'")
    locale = xargs[_LOCALE_KEY]
    if not isinstance(locale, str) or locale not in set(locales):
        return FrontendError(
            "unknown_locale", f"locale {locale!r} is not served", (_XARGS,)
        )
    return None


# @spec ING-FILE-010, ING-FILE-020, ING-FILE-021, ING-FILE-022, ING-FILE-024
class RequestGuard:
    """Refuses request features the single-slot file path cannot honour.

    It distinguishes an omitted field (accepted), an explicit value equal to
    core's default (accepted) and an explicit non-default value (refused);
    the value a field inherits from the served model's generation settings
    is operator configuration and is not inspected.
    """

    def check(
        self, request: Any, *, locales: Iterable[str]
    ) -> FrontendError | None:
        """Return the first refusal for ``request``, or ``None``.

        ``request`` is core's validated ``TranscriptionRequest``; ``locales``
        are the session factory's served locales.
        """
        explicit = request.model_fields_set
        if "use_beam_search" in explicit and request.use_beam_search:
            return _refusal(
                "use_beam_search",
                "beam search is not supported for complete files",
            )
        if (
            "max_completion_tokens" in explicit
            and request.max_completion_tokens is not None
        ):
            return _refusal(
                "max_completion_tokens",
                "a completion limit is not supported for complete files",
            )
        for name in (*FORWARDED_SAMPLING_FIELDS, *IGNORED_FIELDS):
            if name not in explicit:
                continue
            value = getattr(request, name, None)
            if value is None:
                continue
            default = _FIELD_DEFAULTS[name]
            if default is _ABSENT or value != default:
                return _refusal(
                    name,
                    f"{name} must keep its default; the model decodes greedily",
                )
        return _xargs_error(request, locales)


# ---------------------------------------------------------------- locale


# @spec ING-FILE-013
def locale_language(locale: str) -> str | None:
    """Return a locale's language, or ``None`` when it has none.

    The language is the lowercase letters before the first ``-`` or
    uppercase letter, when there are two or three of them: ``en-US`` and
    ``enGB`` are ``en``, ``haw-US`` is ``haw``; ``auto`` has none.
    """
    end = len(locale)
    for index, char in enumerate(locale):
        if char == "-" or "A" <= char <= "Z":
            end = index
            break
    prefix = locale[:end]
    if 2 <= len(prefix) <= 3 and all("a" <= c <= "z" for c in prefix):
        return prefix
    return None


# @spec ING-FILE-013
def recognize_language(
    locale: str, supported_languages: Mapping[str, Any] | Iterable[str]
) -> str | None:
    """Return the ``language`` ``Recognize`` sends core for ``locale``.

    The locale's language when the model supports it, else ``None``.
    """
    language = locale_language(locale)
    if language is None or language not in supported_languages:
        return None
    return language


# ------------------------------------------------------------ request ID


# @spec ING-FILE-026, ING-FILE-027
def build_raw_request(scope: Mapping[str, Any] | None = None) -> Request:
    """Return the ``Request`` core derives the engine request ID from.

    Core takes the engine request ID from the ``X-Request-Id`` header of the
    request it is given, so the frontend gives it a new ``Request`` carrying
    exactly one lowercase ``x-request-id`` header with a fresh UUID. Over
    ``scope`` (the file route) the scope is copied, every client
    ``x-request-id`` entry is dropped and the frontend's added; the original
    scope is not changed. Without ``scope`` (``Recognize``) the request is a
    minimal HTTP scope with that one header.
    """
    entry = (_REQUEST_ID_HEADER, str(uuid.uuid4()).encode("ascii"))
    if scope is None:
        return Request({"type": "http", "headers": [entry]})
    copied = dict(scope)
    copied["headers"] = [
        (name, value)
        for name, value in scope.get("headers", ())
        if bytes(name).lower() != _REQUEST_ID_HEADER
    ] + [entry]
    return Request(copied)


def _engine_request_id(service: Any, raw_request: Request) -> str:
    base = raw_request.headers.get("x-request-id", "")
    return f"{getattr(service, 'task_type', _TASK_TYPE)}-{base}"


# ----------------------------------------------------------- the core call


class _Exhausted:
    """The end of core's event generator, as a pull's result."""


_EXHAUSTED = _Exhausted()


async def _pull_one(generator: Any) -> Any:
    try:
        return await generator.__anext__()
    except StopAsyncIteration:
        return _EXHAUSTED


def _exhausted(pull: asyncio.Future[Any]) -> bool:
    """Whether a finished pull found the end of the event generator."""
    return (
        not pull.cancelled()
        and pull.exception() is None
        and pull.result() is _EXHAUSTED
    )


def _is_event_stream(result: Any) -> bool:
    return hasattr(result, "__anext__")


class TranscriptionCall:
    """The child tasks of one call of core's transcription service.

    ``start`` runs ``create_transcription`` in a child task; ``pull`` runs
    one pull of the event generator it returned (``stream=true``) in
    another. Pulls never overlap. Nothing in the frontend cancels either
    task, in ``drain`` and ``stop`` alike: the engine request ends only
    through ``abort`` (ING-FILE-077).
    """

    def __init__(
        self,
        request_id: str,
        *,
        service: Any,
        audio: bytes,
        request: Any,
        raw_request: Request,
        spawn: Callable[[Coroutine[Any, Any, Any]], asyncio.Task[Any]],
    ) -> None:
        """Prepare the call; nothing runs until ``start``."""
        self.request_id = request_id
        self._service = service
        self._args: tuple[bytes, Any, Request] | None = (
            audio,
            request,
            raw_request,
        )
        self._spawn = spawn
        self.task: asyncio.Task[Any] | None = None
        self.pull_task: asyncio.Task[Any] | None = None
        self.exhausted = False
        self.closed = False

    def start(self) -> asyncio.Task[Any]:
        """Call ``create_transcription(audio, request, raw_request)``."""
        assert self._args is not None and self.task is None
        audio, request, raw_request = self._args
        self._args = None
        self.task = self._spawn(
            self._service.create_transcription(audio, request, raw_request)
        )
        return self.task

    @property
    def returned(self) -> bool:
        """Whether core's service call has returned or raised."""
        return self.task is not None and self.task.done()

    @property
    def result(self) -> Any:
        """What the call returned (``None`` until it returned normally)."""
        task = self.task
        if task is None or not task.done() or task.cancelled():
            return None
        if task.exception() is not None:
            return None
        return task.result()

    @property
    def generator(self) -> Any:
        """The event generator the call returned, or ``None``."""
        result = self.result
        return result if _is_event_stream(result) else None

    @property
    def pulling(self) -> bool:
        """Whether a pull of the generator is outstanding."""
        return self.pull_task is not None and not self.pull_task.done()

    @property
    def settled(self) -> bool:
        """The call has returned and no pull is outstanding."""
        return self.returned and not self.pulling

    def pending(self) -> set[asyncio.Task[Any]]:
        """The call and pull tasks still running."""
        tasks = {self.task, self.pull_task}
        return {t for t in tasks if t is not None and not t.done()}

    def pull(self) -> asyncio.Task[Any]:
        """Start the next pull of the event generator."""
        generator = self.generator
        assert generator is not None and not self.pulling
        self.pull_task = self._spawn(_pull_one(generator))
        return self.pull_task

    async def aclose(self) -> None:
        """Close the event generator once no pull is outstanding."""
        generator = self.generator
        if generator is None or self.closed:
            return
        if self.pull_task is not None and not self.pull_task.done():
            await asyncio.wait({self.pull_task})
        self.closed = True
        close = getattr(generator, "aclose", None)
        if close is None:
            return
        try:
            await close()
        except Exception:
            _LOG.warning(
                "closing the event stream of %s failed",
                self.request_id,
                exc_info=True,
            )


# ------------------------------------------------------------ accounting


class _Accounting:
    """What one file request holds, each released exactly once.

    The permit, the file-owner slot (the ingress admission), the server-load
    count and, when this module registered it, the registry record.
    """

    def __init__(
        self,
        *,
        registry: Any,
        record: Any,
        admission: Any = None,
        load_state: Any = None,
        owns_record: bool = False,
    ) -> None:
        self.registry = registry
        self.record = record
        self.admission = admission
        self.load_state = load_state
        self.owns_record = owns_record
        self.permit: Permit | None = None
        self.released = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        for step in (
            self._permit,
            self._admission,
            self._load,
            self._record,
        ):
            try:
                step()
            except Exception:
                _LOG.exception("releasing a file request's accounting failed")

    def _permit(self) -> None:
        if self.permit is not None:
            self.permit.release()

    def _admission(self) -> None:
        if self.admission is not None:
            self.admission.release()

    def _load(self) -> None:
        if self.load_state is not None:
            self.load_state.server_load_metrics -= 1

    def _record(self) -> None:
        if not self.owns_record:
            return
        records = getattr(self.registry, "records", {})
        values = records.values() if isinstance(records, Mapping) else records
        if any(record is self.record for record in values):
            self.registry.unregister(self.record)


# ----------------------------------------------------------------- owner


@dataclass(frozen=True)
class FileCause:
    """The one terminal cause of a file request.

    Attributes:
        kind: ``transport`` (the client left: no response), ``shutdown``,
            ``deadline``, ``send`` (a response send exceeded its bound: no
            further send) or ``failure``.
        error: The catalog error the surface projects, if the frontend
            decides it.
        status: The HTTP status of the route's own error when it is not the
            catalog's (504 for the execution deadline).
        exception: Core's raised exception (``failure``), which the route
            re-raises for core's handlers and ``Recognize`` classifies.
        deadline: The absolute deadline of a ``deadline`` cause.
    """

    kind: str
    error: FrontendError | None = None
    status: int | None = None
    exception: BaseException | None = None
    deadline: float | None = None


_DONE = "done"
_GONE = "gone"
_TIMEOUT = "timeout"
_CANCELLED = "cancelled"
_NOTICE = "notice"


def _resolve(future: asyncio.Future[None]) -> None:
    if not future.done():
        future.set_result(None)


def _start(status: int, headers: list[tuple[bytes, bytes]]) -> Message:
    return {"type": "http.response.start", "status": status, "headers": headers}


def _body(chunk: str | bytes | memoryview, *, more: bool = True) -> Message:
    data = chunk.encode("utf-8") if isinstance(chunk, str) else bytes(chunk)
    return {"type": "http.response.body", "body": data, "more_body": more}


@functools.cache
def _event_stream_headers() -> tuple[tuple[bytes, bytes], ...]:
    # Core's route answers a returned generator with this response class;
    # its headers are taken from it so both routes send the same ones.
    response = StreamingResponse(iter(()), media_type=_EVENT_STREAM)
    return tuple(response.raw_headers)


# @spec ING-FILE-007, ING-FILE-077, ING-FILE-035, ING-FILE-064, ING-FILE-089
# @spec ING-FILE-068, ING-FILE-069, ING-FILE-078, ING-FILE-079, ING-FILE-091
# @spec ING-FILE-080, ING-FILE-087, ING-ERR-056
class FileRequestOwner:
    """The owner of one complete-file request, on either surface.

    The owner takes a gate permit, calls core's service in a child task
    (``TranscriptionCall``) that it never cancels, and, on the route,
    delivers core's response or event stream with each send bounded by
    ``response_send_timeout``. It records exactly one terminal cause at its
    operation boundaries (the permit, the call's return, each pull, each
    send). A cause selected after the permit while core still works aborts
    the engine request ID every ``ABORT_PERIOD`` until the call has returned
    and no pull is outstanding, then once more, awaited, before the
    generator is closed and anything is released. That cleanup is a
    continuation registered with the request's record; when it outlasts the
    cleanup bound the handler answers and leaves it running, still holding
    the permit, the file-owner slot and the load count.

    The owner withdraws every cancellation of its task and ends by itself.
    A cancellation by the plugin (the settle point of ``drain``, the
    vehicle's settlement, ``stop``, or for ``Recognize`` an RPC termination
    after the listener stop began) selects shutdown when no cause stands
    and makes the handler answer at once (ING-FILE-082, ING-FILE-086).

    ``run`` returns ``None`` on the route (the owner has answered, or the
    client left) and core's result for ``Recognize`` (``send`` is
    ``None``), whose caller projects ``cause``. On the route a raised core
    exception that decides the response is re-raised for core's exception
    handlers.
    """

    def __init__(
        self,
        *,
        service: Any,
        engine_client: Any,
        config: Any,
        clock: Any,
        registry: Any,
        record: Any,
        catalog: Any,
        gate: FileGate | None = None,
        accounting: _Accounting | None = None,
        gone: asyncio.Future[Any] | None = None,
        lifecycle: Any = None,
        state: Any = None,
    ) -> None:
        """Bind the owner to its service, engine, record and accounting.

        ``gate`` is the file gate, ``accounting`` what the request holds,
        ``gone`` a future that completes when the client leaves (the route's
        disconnect watcher or the RPC's cancellation), ``lifecycle`` the
        plugin (its drain deadline and teardown state) and ``state`` core's
        app state.
        """
        self._service = service
        self._engine = engine_client
        self._config = config
        self._clock = clock
        self._registry = registry
        self._record = record
        self._catalog = catalog
        self._gate = gate
        self._accounting = accounting or _Accounting(
            registry=registry, record=record
        )
        self._gone = gone
        self._lifecycle = lifecycle
        self._state = state
        self._asgi_send: Send | None = None
        self._rpc_deadline: float | None = None
        self._request_id = ""
        self._call: TranscriptionCall | None = None
        self._execution_deadline = 0.0
        self._cause: FileCause | None = None
        self._cleanup_by = 0.0
        self._plugin_cancelled = False
        self._abort_needed = False
        self._suppressed = False
        self._gone_seen = False
        self._watch_stopped = False
        self._task_cancelled = False
        self._headers_sent = False
        self._broken = False
        self._result: Any = None
        self._aborted: list[str] = []
        self._tasks: set[asyncio.Task[Any]] = set()
        loop = asyncio.get_running_loop()
        self._settled: asyncio.Future[None] = loop.create_future()
        self._delivered: asyncio.Future[None] = loop.create_future()
        self.continuation: asyncio.Task[None] | None = None

    # -- what callers read ---------------------------------------------------

    @property
    def cause(self) -> FileCause | None:
        """The one terminal cause, or ``None`` after a normal completion."""
        return self._cause

    @property
    def suppressed(self) -> bool:
        """Whether a client that left after the cause suppressed the answer."""
        return self._suppressed

    @property
    def cancelled_by_task(self) -> bool:
        """Whether the client's leaving was a cancellation of the task."""
        return self._task_cancelled

    @property
    def request_id(self) -> str:
        """The engine request ID core derives from the fixed header."""
        return self._request_id

    @property
    def abort_ids(self) -> tuple[str, ...]:
        """The engine request IDs passed to ``abort``, one per call."""
        return tuple(self._aborted)

    @property
    def result(self) -> Any:
        """Core's result for ``Recognize`` after a normal completion."""
        return self._result

    # -- the request ---------------------------------------------------------

    async def run(
        self,
        audio: bytes,
        request: Any,
        raw_request: Request,
        *,
        send: Send | None = None,
        deadline: float | None = None,
    ) -> Any:
        """Run the request from the permit to its last send.

        ``send`` is the route's ASGI send (``None`` for ``Recognize``);
        ``deadline`` is the RPC's absolute deadline, if any.
        """
        self._asgi_send = send
        self._rpc_deadline = deadline
        self._request_id = _engine_request_id(self._service, raw_request)
        try:
            if not self._boundary():
                permit = await self._take_permit()
                if permit is not None:
                    self._accounting.permit = permit
                    if not self._boundary():
                        await self._execute(audio, request, raw_request)
            return await self._conclude()
        except BaseException:
            _resolve(self._delivered)
            if self.continuation is None:
                self._stop_watch()
                self._accounting.release()
            raise

    async def _execute(
        self, audio: bytes, request: Any, raw_request: Request
    ) -> None:
        self._execution_deadline = (
            self._clock.monotonic() + self._config.file_execution_timeout
        )
        self._call = TranscriptionCall(
            self._request_id,
            service=self._service,
            audio=audio,
            request=request,
            raw_request=raw_request,
            spawn=self._spawn,
        )
        self._call.start()
        await self._await_call()
        if self._cause is not None:
            return
        result = self._call.result
        if self._asgi_send is None:
            if _is_event_stream(result):
                self._select(
                    FileCause(
                        "failure",
                        FrontendError(
                            "internal",
                            "core returned an event stream for a "
                            "non-streaming request",
                        ),
                    )
                )
                return
            self._result = result
            return
        if _is_event_stream(result):
            await self._stream()
        else:
            await self._deliver(_complete_response(result))

    # -- causes ----------------------------------------------------------------

    # @spec ING-FILE-089, ING-FILE-078
    def _select(self, cause: FileCause) -> bool:
        """Record ``cause`` if none is recorded; return whether it was.

        The cleanup deadline is fixed here (ING-FILE-078):
        ``session_cleanup_timeout`` from now, clipped to ``D`` only while a
        ``drain`` call runs its graceful work with ``D`` still ahead. No
        later change of the drain state or cancellation moves it.
        """
        if self._cause is not None:
            return False
        self._cause = cause
        now = self._clock.monotonic()
        bound: float = now + self._config.session_cleanup_timeout
        drain = self._drain_deadline()
        if drain is not None and drain > now:
            bound = min(bound, drain)
        self._cleanup_by = bound
        call = self._call
        self._abort_needed = (
            self._accounting.permit is not None
            and call is not None
            and (
                not call.returned
                or (call.generator is not None and not call.exhausted)
            )
        )
        return True

    def _shutdown(self) -> None:
        """The plugin cancelled the request: shutdown, unless a cause stands."""
        self._plugin_cancelled = True
        self._select(
            FileCause(
                "shutdown",
                FrontendError(
                    "service_unavailable", "the server is shutting down"
                ),
            )
        )

    # @spec ING-FILE-086
    def _client_gone(self) -> None:
        self._gone_seen = True
        if self._asgi_send is None and self._listener_stopping():
            # An RPC terminated after the listener stop began is the
            # plugin's cancellation, never the client's.
            self._shutdown()
            return
        if not self._select(FileCause("transport")):
            self._suppressed = True

    def _listener_stopping(self) -> bool:
        return getattr(self._lifecycle, "listener_stopping", False) is True

    def _drain_deadline(self) -> float | None:
        value = getattr(self._lifecycle, "drain_deadline", None)
        return value if isinstance(value, (int, float)) else None

    def _bounds_passed(self) -> bool:
        """Whether every send bound counts as passed (ING-EP-110)."""
        lifecycle = self._lifecycle
        if getattr(lifecycle, "send_bounds_passed", False) is True:
            return True
        if getattr(lifecycle, "torn_down", False) is True:
            return True
        drain = self._drain_deadline()
        return drain is not None and drain <= self._clock.monotonic()

    # @spec ING-FILE-089, ING-FILE-086
    def _plugin_cancelling(self) -> bool:
        """Whether a cancellation of the owner task is the plugin's own.

        Read from the plugin's own signals only (``_cancelled_by_plugin``),
        never from core's ``state.draining``, the plugin's drain flag or
        closed admission: a cancellation in those windows is the client's.
        """
        return _cancelled_by_plugin(
            self._lifecycle, self._record, rpc=self._asgi_send is None
        )

    # @spec ING-FILE-082, ING-FILE-086
    def _interrupted(self) -> None:
        """The owner task was cancelled, by the plugin or by its transport.

        The cancellation is withdrawn: the owner ends by itself. The
        plugin's cancellation selects shutdown when no cause stands and
        makes the handler answer at once; a transport's is the client
        leaving.
        """
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            task.uncancel()
        if self._plugin_cancelling():
            self._shutdown()
            return
        self._task_cancelled = True
        if not self._select(FileCause("transport")):
            self._suppressed = True

    def _rpc_timeout(self, deadline: float) -> FileCause:
        return FileCause(
            "deadline",
            FrontendError("request_timeout", "the RPC deadline passed"),
            deadline=deadline,
        )

    def _execution_timeout(self) -> FileCause:
        return FileCause(
            "deadline",
            FrontendError(
                "request_timeout",
                "core did not finish within file_execution_timeout",
            ),
            status=504,
            deadline=self._execution_deadline,
        )

    def _boundary(
        self, phase: float | None = None, expiry: FileCause | None = None
    ) -> bool:
        """Arbitrate the client and expired deadlines at an operation boundary.

        Runs before the operation's own outcome (a permit, core's result or
        failure, a pull) is consumed. The client's leaving wins; otherwise
        the earliest expired deadline among the RPC's and the operation's
        own bound ``phase`` (whose cause is ``expiry``), the phase winning a
        tie. Returns whether a cause is selected.
        """
        if self._cause is not None:
            return True
        gone = self._gone
        if (
            gone is not None
            and not self._watch_stopped
            and gone.done()
            and not gone.cancelled()
        ):
            self._client_gone()
            return True
        now = self._clock.monotonic()
        rpc = self._rpc_deadline
        if (
            phase is not None
            and expiry is not None
            and phase <= now
            and (rpc is None or phase <= rpc)
        ):
            self._select(expiry)
            return True
        if rpc is not None and rpc <= now:
            self._select(self._rpc_timeout(rpc))
            return True
        return False

    # -- waiting ---------------------------------------------------------------

    # @spec ING-EP-106
    def _spawn(
        self, coro: Coroutine[Any, Any, Any], *, timer: bool = False
    ) -> asyncio.Task[Any]:
        """Start a child task under the request's registry record.

        A ``timer`` (a ``clock.sleep`` bounding a wait) is a child of the
        record too, but not one the cleanup waits for.
        """
        task = asyncio.ensure_future(coro)
        if not timer:
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        children = getattr(self._record, "children", None)
        if children is not None:
            children.add(task)
            task.add_done_callback(children.discard)
        return task

    def _spawn_timer(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        return self._spawn(coro, timer=True)

    def _timer(self, deadline: float) -> asyncio.Task[Any]:
        """A registered child that ends at the absolute ``deadline``."""
        remaining = max(0.0, deadline - self._clock.monotonic())
        return self._spawn_timer(self._clock.sleep(remaining))

    def _notice(self) -> asyncio.Future[None] | None:
        """The plugin's next change of the send bounds, if one is pending."""
        announced = getattr(self._lifecycle, "drain_announced", None)
        if not callable(announced):
            return None
        notice = announced()
        if not isinstance(notice, asyncio.Future) or notice.done():
            return None
        return notice

    def _watching(self) -> bool:
        gone = self._gone
        if gone is None or self._gone_seen or self._watch_stopped:
            return False
        return not (gone.done() and gone.cancelled())

    async def _wait(
        self,
        target: asyncio.Future[Any],
        deadline: float | None,
        *,
        watch: bool = True,
        notice: bool = False,
    ) -> str:
        """Wait for ``target``, which is never cancelled here.

        Returns ``_DONE``, ``_GONE`` (the client left), ``_TIMEOUT``
        (``deadline`` passed), ``_NOTICE`` (with ``notice``: the plugin
        changed the send bounds) or ``_CANCELLED`` (the owner task was
        cancelled, by the plugin or its transport; the cause is recorded
        and the cancellation withdrawn). The deadline is a registered
        timer, never a task outside the record.
        """
        while True:
            if target.done():
                return _DONE
            gone = self._gone if watch and self._watching() else None
            if gone is not None and gone.done():
                self._client_gone()
                return _GONE
            if deadline is not None and deadline <= self._clock.monotonic():
                return _TIMEOUT
            waiters: set[asyncio.Future[Any]] = {target}
            if gone is not None:
                waiters.add(gone)
            change = self._notice() if notice else None
            if change is not None:
                waiters.add(change)
            timer = None if deadline is None else self._timer(deadline)
            if timer is not None:
                waiters.add(timer)
            try:
                await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            except asyncio.CancelledError:
                self._interrupted()
                return _CANCELLED
            finally:
                if timer is not None:
                    timer.cancel()
                if change is not None and not change.done():
                    change.cancel()
            if change is not None and change.done() and not target.done():
                return _NOTICE

    async def _wait_until_done(
        self,
        target: asyncio.Future[Any],
        bound: float,
        *,
        watch: bool,
    ) -> bool:
        """Wait for ``target`` up to ``bound``; ``True`` if it finished.

        Returns ``False`` at the bound and when the plugin cancels the
        owner, which then answers at once (ING-FILE-082).
        """
        while True:
            outcome = await self._wait(target, bound, watch=watch)
            if outcome == _DONE:
                return True
            if outcome == _TIMEOUT:
                return False
            if outcome == _CANCELLED and self._plugin_cancelled:
                return False

    async def _join(self, task: asyncio.Future[Any]) -> None:
        """Await ``task`` to its end, absorbing cancellations of the owner."""
        while not task.done():
            try:
                await asyncio.wait({task})
            except asyncio.CancelledError:
                self._interrupted()

    # -- permit and call -------------------------------------------------------

    async def _take_permit(self) -> Permit | None:
        gate = self._gate
        if gate is None:
            self._select(
                FileCause(
                    "failure",
                    FrontendError(
                        "unsupported_capability",
                        "complete files are not served",
                    ),
                )
            )
            return None
        gate_deadline = (
            self._clock.monotonic() + self._config.file_gate_wait_timeout
        )
        rpc = self._rpc_deadline
        effective = gate_deadline if rpc is None else min(gate_deadline, rpc)
        acquire = self._spawn(gate.acquire(effective, spawn=self._spawn_timer))
        outcome: str | None = None
        try:
            outcome = await self._wait(acquire, None)
            if outcome != _DONE:
                # The client left or the owner was cancelled while waiting.
                acquire.cancel()
                await self._join(acquire)
        finally:
            if outcome != _DONE:
                if not acquire.done():
                    acquire.cancel()
                elif not acquire.cancelled() and acquire.exception() is None:
                    acquire.result().release()
        if outcome != _DONE:
            return None
        error = acquire.exception()
        if error is None:
            # ``run`` arbitrates this boundary with the permit held.
            return typing.cast(Permit, acquire.result())
        if not isinstance(error, FrontendError):
            raise error
        refusal = FileCause("failure", error)
        if error.code != "busy":
            if not self._boundary():
                self._select(refusal)
        elif not self._boundary(gate_deadline, refusal):
            if rpc is not None and rpc < gate_deadline:
                # The wait was clipped to the RPC deadline and expired as it.
                self._select(self._rpc_timeout(rpc))
            else:
                self._select(refusal)
        return None

    async def _await_call(self) -> None:
        call = self._call
        assert call is not None and call.task is not None
        execution = self._execution_deadline
        rpc = self._rpc_deadline
        deadline = execution if rpc is None else min(execution, rpc)
        outcome = await self._wait(call.task, deadline)
        if outcome not in (_DONE, _TIMEOUT):
            return  # the client left or the owner was cancelled: recorded
        # A return or failure that is visible together with the client's
        # leaving or an expired deadline yields to them.
        if self._boundary(execution, self._execution_timeout()):
            return
        if outcome == _DONE:
            task = call.task
            if task.cancelled():
                self._select(
                    FileCause(
                        "failure",
                        FrontendError(
                            "internal", "core's service call was cancelled"
                        ),
                    )
                )
            elif task.exception() is not None:
                self._select(FileCause("failure", exception=task.exception()))
        elif outcome == _TIMEOUT:
            if rpc is not None and rpc < execution:
                self._select(self._rpc_timeout(rpc))
            else:
                self._select(self._execution_timeout())

    # -- delivery (the route) --------------------------------------------------

    # @spec ING-FILE-091, ING-EP-110
    async def _send(
        self,
        messages: list[Message],
        *,
        phase: bool,
        terminal: bool = False,
    ) -> bool:
        """Send ``messages`` in order, each within ``response_send_timeout``.

        One child task sends them back to back, so a response's start and
        body leave together; each message's bound runs from the completion
        of the one before and is clipped to ``D`` while a ``drain`` call
        runs its graceful work. ``phase`` marks sends before the event
        generator is exhausted, which ``file_execution_timeout`` also
        bounds. Each message's completion is judged against its own bound
        as it completes, before the next starts: one that completes at or
        after its bound, or once the send bounds count as passed, has
        failed and no further message starts (ING-FILE-091,
        ING-LIFE-061). Once the send bounds count as passed
        (ING-EP-110) no send starts unless it is ``terminal`` (the
        response, or the termination of the event stream), which gets one
        attempt; a send in progress then is cut. Returns whether every
        message was delivered; a failure records its cause and ends
        delivery.
        """
        send = self._asgi_send
        if send is None or self._broken:
            return False
        timeout = float(self._config.response_send_timeout)
        progress: list[float] = [self._clock.monotonic()]
        late: list[bool] = [False]

        def bound() -> float:
            deadline = progress[0] + timeout
            if phase:
                deadline = min(deadline, self._execution_deadline)
            drain = self._drain_deadline()
            return deadline if drain is None else min(deadline, drain)

        async def deliver(*, judged: bool = True) -> None:
            for message in messages:
                await send(message)
                if message["type"] == "http.response.start":
                    self._headers_sent = True
                if not judged:
                    continue  # the one terminal attempt: no bound applies
                now = self._clock.monotonic()
                if now >= bound() or self._bounds_passed():
                    # Completed after its bound, or after every bound came
                    # to count as passed (the settlement began or the
                    # plugin was torn down in the same turn): this send has
                    # failed and no further message starts (ING-EP-110).
                    late[0] = True
                    return
                progress[0] = now

        if self._bounds_passed():
            if terminal:
                return await self._last_attempt(deliver(judged=False))
            self._broken = True
            self._select(self._send_timeout())
            return False

        task = self._spawn(deliver())
        while True:
            if not task.done() and self._bounds_passed():
                # The send in progress was its one attempt: it is cut.
                outcome = _TIMEOUT
                break
            outcome = await self._wait(task, bound(), notice=True)
            if outcome in (_DONE, _GONE):
                break
            if outcome in (_CANCELLED, _NOTICE):
                # The bounds may have changed; a shutdown or transport
                # cause is recorded and the send keeps its own bound.
                continue
            if self._clock.monotonic() >= bound():
                break
            # A message completed; the next has its own bound.
        if outcome == _DONE and late[0]:
            outcome = _TIMEOUT
        if outcome == _DONE:
            if task.cancelled() or task.exception() is not None:
                self._broken = True
                if not self._select(FileCause("transport")):
                    self._suppressed = True
                return False
            return True
        task.cancel()
        self._broken = True
        if outcome == _TIMEOUT:
            now = self._clock.monotonic()
            if (
                phase
                and self._execution_deadline <= now
                and self._execution_deadline <= progress[0] + timeout
            ):
                self._select(self._execution_timeout())
            else:
                self._select(self._send_timeout())
        return False

    async def _last_attempt(self, deliver: Coroutine[Any, Any, None]) -> bool:
        """A terminal send past its bound: one attempt that adds no wait.

        The send runs to its first suspension and succeeds only if it is
        complete then; otherwise it is cancelled and stays under the
        record until it ends (ING-EP-110, ING-FILE-091).
        """
        task = self._spawn(deliver)
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            self._interrupted()
        if task.done() and not task.cancelled() and task.exception() is None:
            return True
        self._broken = True
        if task.done():
            if not self._select(FileCause("transport")):
                self._suppressed = True
            return False
        task.cancel()
        self._select(self._send_timeout())
        return False

    def _send_timeout(self) -> FileCause:
        return FileCause(
            "send",
            FrontendError(
                "request_timeout",
                "a response send exceeded response_send_timeout",
            ),
        )

    async def _deliver(self, response: Response) -> None:
        start = _start(response.status_code, list(response.raw_headers))
        body = _body(response.body, more=False)
        await self._send([start, body], phase=False, terminal=True)

    async def _stream(self) -> None:
        call = self._call
        assert call is not None
        start = _start(200, list(_event_stream_headers()))
        if not await self._send([start], phase=True):
            return
        while self._cause is None:
            pull = call.pull()
            execution = self._execution_deadline
            outcome = await self._wait(pull, execution)
            if outcome not in (_DONE, _TIMEOUT):
                return  # the client left or the owner was cancelled
            if outcome == _DONE and _exhausted(pull):
                # The engine request has ended: a cause selected here
                # needs no abort.
                call.exhausted = True
            if self._boundary(execution, self._execution_timeout()):
                return
            if outcome == _TIMEOUT:
                self._select(self._execution_timeout())
                return
            if pull.cancelled():
                self._select(
                    FileCause(
                        "failure",
                        FrontendError(
                            "internal", "an event pull was cancelled"
                        ),
                    )
                )
                return
            error = pull.exception()
            if error is not None:
                self._select(FileCause("failure", exception=error))
                return
            event = pull.result()
            if event is _EXHAUSTED:
                break
            if not await self._send([_body(event)], phase=True):
                return
        if self._cause is None:
            await self._send(
                [_body(b"", more=False)], phase=False, terminal=True
            )

    async def _respond(self) -> None:
        """Answer the route's cause, if it has an answer."""
        cause = self._cause
        if (
            self._asgi_send is None
            or cause is None
            or cause.kind in ("transport", "send")
            or self._suppressed
            or self._broken
        ):
            return
        if not self._headers_sent:
            if cause.exception is not None:
                return  # re-raised for core's exception handlers
            error = cause.error or _internal()
            status = cause.status or self._catalog.http_status(error.code)
            await self._deliver(
                JSONResponse(error.http_body(status), status_code=status)
            )
            return
        # After headers no status can follow: end the event stream as core's
        # own generator ends a failed one.
        payload = self._stream_error(cause)
        await self._send(
            [
                _body(f"data: {payload}\n\n"),
                _body("data: [DONE]\n\n"),
                _body(b"", more=False),
            ],
            phase=False,
            terminal=True,
        )

    def _stream_error(self, cause: FileCause) -> str:
        if cause.exception is not None:
            from vllm.entrypoints.serve.exception_handling import (
                error_response,
            )

            exception = cause.exception
            if not isinstance(exception, Exception):
                exception = RuntimeError(str(exception))
            response = error_response.create_error_response(exception)
            return json.dumps(response.model_dump())
        error = cause.error or _internal()
        status = cause.status or self._catalog.http_status(error.code)
        return json.dumps(error.http_body(status))

    # -- conclusion and cleanup ------------------------------------------------

    # @spec ING-FILE-078, ING-FILE-079, ING-FILE-082, ING-FILE-080
    async def _conclude(self) -> Any:
        """Hand the cleanup to the continuation, answer, then return.

        With a cause, the handler waits for the settlement until the
        cleanup deadline fixed at the cause, then answers; a shutdown or
        any cancellation by the plugin answers at once. After a normal
        completion it waits for the continuation within
        ``session_cleanup_timeout``. Whatever is left stays with the
        registered continuation.
        """
        # Before the permit nothing reached core, but the disconnect watcher
        # still settles under the same continuation and bound.
        continuation = asyncio.ensure_future(self._cleanup())
        self.continuation = continuation
        self._registry.attach_continuation(self._record, continuation)
        normal_bound: float = (
            self._clock.monotonic() + self._config.session_cleanup_timeout
        )

        def limit() -> float:
            return normal_bound if self._cause is None else self._cleanup_by

        def at_once() -> bool:
            cause = self._cause
            return self._plugin_cancelled or (
                cause is not None and cause.kind == "shutdown"
            )

        if self._cause is not None and not at_once():
            await self._wait_until_done(self._settled, limit(), watch=True)
        await self._respond()
        _resolve(self._delivered)
        if not at_once():
            await self._wait_until_done(continuation, limit(), watch=False)
        return self._outcome()

    def _outcome(self) -> Any:
        cause = self._cause
        if self._asgi_send is None:
            return self._result
        if (
            cause is not None
            and cause.kind == "failure"
            and cause.exception is not None
            and not self._headers_sent
            and not self._suppressed
        ):
            raise cause.exception
        return None

    def _stop_watch(self) -> None:
        # From here the watcher's end is the owner's doing, not the client's.
        self._watch_stopped = True
        if self._gone is not None and not self._gone.done():
            self._gone.cancel()

    async def _abort(self) -> None:
        self._aborted.append(self._request_id)
        try:
            await self._engine.abort(self._request_id)
        except Exception:
            _LOG.warning("abort of %s failed", self._request_id, exc_info=True)

    async def _abort_until_settled(self) -> None:
        """Abort every ``ABORT_PERIOD`` until the call settles, then once more.

        Each period is a registered timer child (ING-EP-106).
        """
        call = self._call
        assert call is not None
        await self._abort()
        while not call.settled:
            timer = self._timer(self._clock.monotonic() + ABORT_PERIOD)
            try:
                await asyncio.wait(
                    {*call.pending(), timer},
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                timer.cancel()
            if not call.settled and timer.done() and not timer.cancelled():
                await self._abort()
        # Core submits lazily, so earlier aborts may have preceded it.
        await self._abort()

    async def _cleanup(self) -> None:
        """The cleanup continuation: settle, then release exactly once.

        The children it settles include the disconnect watcher, whose
        cancellation can outlast the handler's cleanup bound; until it has
        settled the accounting and the record stay held; the vehicle's
        settlement awaits the continuation, it never cancels it.
        """
        try:
            if self._abort_needed:
                await self._abort_until_settled()
            self._stop_watch()
            if self._call is not None:
                await self._call.aclose()
            pending: set[asyncio.Future[Any]] = {
                task for task in self._tasks if not task.done()
            }
            watcher = self._gone
            if watcher is not None and not watcher.done():
                pending.add(watcher)
            if pending:
                await asyncio.wait(pending)
            _resolve(self._settled)
            await self._delivered
            # The answer's own sends, a cut one included, end before the
            # record goes (ING-EP-110).
            while True:
                pending = {task for task in self._tasks if not task.done()}
                if not pending:
                    break
                await asyncio.wait(pending)
        finally:
            self._accounting.release()


def _internal() -> FrontendError:
    return FrontendError("internal", "unexpected failure of the file request")


def _complete_response(result: Any) -> Response:
    """Core's route's response for a returned result."""
    from vllm.entrypoints.serve.engine.protocol import ErrorResponse

    if isinstance(result, ErrorResponse):
        return JSONResponse(
            content=result.model_dump(), status_code=result.error.code
        )
    if isinstance(result, BaseModel):
        return JSONResponse(content=result.model_dump())
    raise TypeError(f"unexpected result from core: {type(result).__name__}")


# ------------------------------------------------------------- the route


def _catalog(plugin: Any) -> Any:
    catalog = getattr(plugin, "catalog", None)
    return catalog if catalog is not None else default_catalog()


def _limits(config: Any) -> MultipartLimits:
    return MultipartLimits(
        envelope_max_bytes=config.http_multipart_envelope_max_bytes,
        content_type_max_bytes=config.http_content_type_max_bytes,
        boundary_max_bytes=config.http_multipart_boundary_max_bytes,
        max_parts=config.http_multipart_max_parts,
        max_header_bytes=config.http_multipart_max_header_bytes,
        text_field_max_bytes=config.http_text_field_max_bytes,
    )


def _locales(plugin: Any) -> tuple[str, ...]:
    locales = getattr(getattr(plugin, "factory", None), "locales", None)
    return tuple(locales or ())


# @spec ING-FILE-089, ING-FILE-086
def _cancelled_by_plugin(plugin: Any, record: Any, *, rpc: bool) -> bool:
    """Whether a cancellation of a file request's task is the plugin's own.

    The plugin's own signals only: it marked the request's record as it
    cancelled the task (the settle point of ``drain``, the vehicle's
    settlement, ``stop``), it is torn down, or every send bound counts as
    passed (its settlement began or it skipped its graceful steps); and
    for ``Recognize`` (``rpc``), it began stopping the gRPC listener, whose
    RPC terminations are its cancellations. Core's ``state.draining``, the
    plugin's drain flag, ``D`` alone and closed admission (a recorded
    fault) are not: a cancellation in those windows is the client's.
    """
    if getattr(record, "cancelled_by_plugin", False) is True:
        return True
    if (
        getattr(plugin, "torn_down", False) is True
        or getattr(plugin, "send_bounds_passed", False) is True
    ):
        return True
    return rpc and getattr(plugin, "listener_stopping", False) is True


# @spec ING-EP-095, ING-EP-110, ING-FILE-083
async def _refuse(
    plugin: Any,
    send: Send,
    error: FrontendError,
    *,
    record: Any = None,
) -> None:
    """Send the route's own error in core's shape as one bounded send.

    The start and the body leave in sequence from one send, which is one
    terminal attempt once the send bounds count as passed (ING-EP-110).
    Without ``record`` (a refusal that creates no owner) the handler
    awaits the send inline and owns it until it ends or is registered as a
    send without an owner (ING-EP-095); with one, the send is a child of
    the request's record. A send past its bound or a failed transport ends
    delivery; a cancellation of the handler propagates.
    """
    status = _catalog(plugin).http_status(error.code)
    response = JSONResponse(error.http_body(status), status_code=status)
    messages: list[Message] = [
        _start(status, list(response.raw_headers)),
        _body(response.body, more=False),
    ]

    async def deliver() -> None:
        for message in messages:
            await send(message)

    with contextlib.suppress(Exception):
        await surfaces.bounded_send(
            plugin, deliver(), record=record, terminal=True
        )


def _release_when_settled(registry: Any, record: Any, accounting: Any) -> None:
    """Release a request that ended before its owner, once its children end.

    A child still running (a cut send, the body reader) keeps the record
    and the accounting under a registered continuation (ING-EP-106).
    """
    held = (
        *getattr(record, "children", ()),
        *getattr(record, "unsettled_sends", ()),
    )
    pending = {task for task in held if not task.done()}
    if not pending:
        accounting.release()
        return

    async def settle() -> None:
        try:
            while pending:
                await asyncio.wait(pending)
                pending.difference_update(
                    {task for task in pending if task.done()}
                )
        finally:
            accounting.release()

    registry.attach_continuation(record, asyncio.ensure_future(settle()))


@functools.cache
def _list_fields() -> frozenset[str]:
    """Names and aliases of core's request fields declared as lists."""
    from vllm.entrypoints.speech_to_text.transcription.protocol import (
        TranscriptionRequest,
    )

    names: set[str] = set()
    for name, info in TranscriptionRequest.model_fields.items():
        if typing.get_origin(info.annotation) is list:
            names.add(name)
            if info.alias:
                names.add(info.alias)
    return frozenset(names)


def _form_data(fields: Iterable[tuple[str, str]]) -> dict[str, Any]:
    """Map text parts to core's request input, refusing foreign names.

    A repeated name is allowed only for a field core declares as a list.
    """
    data: dict[str, Any] = {}
    lists = _list_fields()
    for name, value in fields:
        if name not in REQUEST_FIELD_ALLOWLIST or name == "file":
            raise _refusal(
                name, f"{name!r} is not a field of the transcription request"
            )
        if name in lists:
            data.setdefault(name, []).append(value)
            continue
        if name in data:
            raise FrontendError(
                "malformed_request", f"the field {name!r} is repeated", (name,)
            )
        data[name] = value
    return data


def _validate_request(data: dict[str, Any]) -> Any:
    """Build core's ``TranscriptionRequest`` with core's own validation.

    A rejected value raises ``RequestValidationError`` (core's handler
    answers 400 in core's shape); an exception core's model raises itself
    propagates for core's handlers.
    """
    from vllm.entrypoints.speech_to_text.transcription.protocol import (
        TranscriptionRequest,
    )

    try:
        return TranscriptionRequest.model_validate(data)
    except ValidationError as exc:
        errors = [
            {**error, "loc": ("body", *error.get("loc", ()))}
            for error in exc.errors(include_url=False)
        ]
        raise RequestValidationError(errors) from None


_GUARD = RequestGuard()


async def _ingest(
    plugin: Any, raw_request: Request, receive: Receive, record: Any
) -> tuple[Any, bytes]:
    """Steps 2 to 5 of the route: content type to the guard.

    The body read and its timer are children of the request's ``record``.
    """
    from vllm.entrypoints.speech_to_text.base.utils import (
        read_upload_with_limit,
    )

    config = plugin.config
    limits = _limits(config)
    values = raw_request.headers.getlist("content-type")
    if len(values) != 1:
        raise FrontendError(
            "malformed_request",
            "the request needs one multipart/form-data Content-Type",
        )
    boundary = parse_content_type(values[0], limits)
    parse_content_length(raw_request.headers, limits)
    form = await parse_multipart(
        receive,
        boundary,
        limits,
        deadline=plugin.clock.monotonic() + config.http_body_read_timeout,
        clock=plugin.clock,
        spawn=functools.partial(surfaces.child, record),
    )
    data = _form_data(form.fields)
    data["file"] = make_upload(form)
    request = _validate_request(data)
    locales = _locales(plugin)
    error = _xargs_error(request, locales)
    if error is not None:
        raise error
    audio = await read_upload_with_limit(request.file)
    error = _GUARD.check(request, locales=locales)
    if error is not None:
        raise error
    return request, audio


async def _watch_disconnect(receive: Receive) -> None:
    """Complete when the client disconnects (after the body was read)."""
    while True:
        message = await receive()
        if message.get("type") == "http.disconnect":
            return


# @spec ING-FILE-083, ING-FILE-082, ING-FILE-086, ING-FILE-090, ING-EP-106
async def _serve_file_request(
    plugin: Any, raw_request: Request, receive: Receive, send: Send
) -> None:
    """The file route, in the order the files design fixes.

    Files-enabled check, ingress cap, content type, body read, parse,
    validation and guard, gate permit, call to core. A refusal before the
    ingress cap creates no owner (ING-EP-095); from the cap on, the
    request has one registered owner, this task.
    """
    enabled = getattr(plugin, "files_enabled", None)
    if enabled is False:
        await _refuse(
            plugin,
            send,
            FrontendError(
                "unsupported_capability",
                "complete files are disabled on this server",
            ),
        )
        return
    ingress = getattr(plugin, "ingress", None)
    registry = getattr(plugin, "registry", None)
    gate = getattr(plugin, "file_gate", None)
    if not enabled or ingress is None or registry is None or gate is None:
        await _refuse(
            plugin,
            send,
            FrontendError("service_unavailable", "the frontend is not ready"),
        )
        return
    admission = ingress.try_admit(OwnerKind.file_route)
    if admission is None:
        await _refuse(plugin, send, ingress.refusal(OwnerKind.file_route))
        return
    task = asyncio.current_task()
    assert task is not None
    record = registry.register(OwnerKind.file_route, task)
    state = raw_request.app.state
    tracking = bool(getattr(state, "enable_server_load_tracking", False))
    if tracking:
        state.server_load_metrics += 1
    accounting = _Accounting(
        registry=registry,
        record=record,
        admission=admission,
        load_state=state if tracking else None,
        owns_record=True,
    )
    owner: FileRequestOwner | None = None
    watcher: asyncio.Future[None] | None = None
    try:
        try:
            request, audio = await _ingest(plugin, raw_request, receive, record)
        except FrontendError as error:
            await _refuse(plugin, send, error, record=record)
            return
        except ClientDisconnect:
            return  # only bounded multipart memory is dropped
        except asyncio.CancelledError:
            if not _cancelled_by_plugin(plugin, record, rpc=False):
                raise
            # The plugin's cancellation: answer at once (ING-FILE-082).
            if task.cancelling():
                task.uncancel()
            await _refuse(
                plugin,
                send,
                FrontendError(
                    "service_unavailable", "the server is shutting down"
                ),
                record=record,
            )
            return
        watcher = surfaces.child(record, _watch_disconnect(receive))
        owner = FileRequestOwner(
            service=state.openai_serving_transcription,
            engine_client=state.engine_client,
            config=plugin.config,
            clock=plugin.clock,
            registry=registry,
            record=record,
            catalog=_catalog(plugin),
            gate=gate,
            accounting=accounting,
            gone=watcher,
            lifecycle=plugin,
            state=state,
        )
        await owner.run(
            audio, request, build_raw_request(raw_request.scope), send=send
        )
    finally:
        if owner is None:
            if watcher is not None:
                watcher.cancel()
            _release_when_settled(registry, record, accounting)


class _FileRouteResponse(Response):
    """The route's response object: the whole request runs in ``__call__``.

    Returning it from the endpoint gives the owner the request's ASGI
    ``receive`` and ``send`` inside core's exception handling, so the owner
    reads the body itself, bounds every send and can re-raise core's own
    exceptions for core's handlers.
    """

    def __init__(self, plugin: Any, raw_request: Request) -> None:
        super().__init__()
        self._plugin = plugin
        self._raw_request = raw_request

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        await _serve_file_request(
            self._plugin, self._raw_request, receive, send
        )


# @spec ING-FILE-075, ING-FILE-055, ING-FILE-083, ING-FILE-005, ING-FILE-011
# @spec ING-FILE-023, ING-FILE-025, ING-FILE-041, ING-FILE-042, ING-FILE-086
# @spec ING-FILE-066, ING-FILE-090, ING-ERR-009, ING-ERR-055
def file_route(plugin: Any) -> Callable[[Request], Awaitable[Response]]:
    """Return the handler of ``POST /plugins/riva/v1/audio/transcriptions``.

    ``attach_router`` registers it beside core's route, only when core
    serves that route.

    The handler reads the plugin's state at request time, so it can be
    created in ``attach_router`` before ``init_state`` runs.
    """

    async def create_transcriptions(raw_request: Request) -> Response:
        """Transcribe a complete file through core's transcription service."""
        return _FileRouteResponse(plugin, raw_request)

    return create_transcriptions


# ------------------------------------------------------------- Recognize


def _rpc_cancellation(context: Any) -> asyncio.Future[None]:
    """A future that completes when the client cancels the RPC."""
    gone: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    def cancelled() -> bool:
        check = getattr(context, "cancelled", None)
        return bool(check()) if callable(check) else False

    def on_done(_: Any) -> None:
        if cancelled():
            _resolve(gone)

    add = getattr(context, "add_done_callback", None)
    if callable(add):
        add(on_done)
    if cancelled():
        _resolve(gone)
    return gone


def _recognize_request(wav: bytes, locale: str, service: Any) -> Any:
    """Core's request for a ``Recognize`` file: plain, with the exact locale.

    ``response_format`` is ``json``, there is no streaming, and beam search
    and the completion limit are never set.
    """
    from vllm.entrypoints.speech_to_text.transcription.protocol import (
        TranscriptionRequest,
    )

    model_cls = getattr(service, "model_cls", None)
    supported = getattr(model_cls, "supported_languages", None) or {}
    upload = _memory_upload(
        io.BytesIO(wav), filename="audio.wav", content_type="audio/wav"
    )
    return TranscriptionRequest.model_validate(
        {
            "file": upload,
            "response_format": "json",
            "stream": False,
            "language": recognize_language(locale, supported),
            _XARGS: {_LOCALE_KEY: locale},
        }
    )


def _recognize_response(result: Any, audio_seconds: float) -> Any:
    from vllm.entrypoints.serve.engine.protocol import ErrorResponse

    if isinstance(result, ErrorResponse):
        raise project_error_response(result)
    text = getattr(result, "text", None)
    if not isinstance(text, str):
        raise FrontendError("internal", "core's response carries no transcript")
    response = rasr.RecognizeResponse()
    result_message = response.results.add()
    result_message.alternatives.add().transcript = text
    result_message.audio_processed = audio_seconds
    return response


def wav_seconds(audio_wav: bytes) -> float:
    """The duration of the in-memory WAV handed core, from its header."""
    with wave.open(io.BytesIO(audio_wav), "rb") as reader:
        return reader.getnframes() / reader.getframerate()


def _resolve_locale(plugin: Any, locale: str) -> str:
    factory = getattr(plugin, "factory", None)
    if locale and locale in _locales(plugin):
        return locale
    from vllm_riva_frontend.lease import resolve_locale

    return resolve_locale(locale, factory, field="language_code")


# @spec ING-FILE-001, ING-FILE-002, ING-FILE-013, ING-FILE-027, ING-FILE-043
# @spec ING-FILE-044, ING-FILE-045, ING-FILE-081, ING-FILE-086, ING-FILE-049
# @spec ING-FILE-050, ING-FILE-065, ING-FILE-085, ING-FILE-088
async def recognize_file(
    plugin: Any,
    audio_wav: bytes,
    locale: str,
    *,
    context: Any,
    deadline: float | None,
    admission: Any,
    record: Any,
) -> Any:
    """Run one ``Recognize`` file on core's transcription service.

    ``audio_wav`` is the request's audio as an in-memory 16 kHz mono 16-bit
    WAV and ``locale`` the validated client locale. The call takes a gate
    permit, sends core the WAV with a plain request (``response_format``
    ``json``, no streaming, no beam search, no completion limit), the exact
    locale in ``vllm_xargs`` and ``language`` only for a supported language,
    over a minimal request that fixes the engine request ID, and returns one
    ``RecognizeResponse`` with one result and one alternative only after
    core completed normally. It is outside the server-load counter.

    ``deadline`` is the RPC's absolute deadline, read at servicer entry
    (``None`` without one). ``admission`` and ``record`` are the ingress
    admission and the owner record the servicer took at entry, before any
    decode; this call takes them over and releases them exactly once, from
    the cleanup continuation when that outlasts the call.

    Raises:
        FrontendError: The projected failure: core's returned
            ``ErrorResponse`` and raised exceptions are projected separately
            (``project_error_response``, ``classify_core_exception``);
            ``request_timeout``, ``busy``, ``service_unavailable`` for the
            frontend's own causes.
        asyncio.CancelledError: The task was cancelled by its transport.
    """
    registry = plugin.registry
    accounting = _Accounting(
        registry=registry, record=record, admission=admission, owns_record=True
    )
    try:
        service = plugin.transcription
        gate = plugin.file_gate
        if not plugin.files_enabled or service is None or gate is None:
            raise FrontendError(
                "unsupported_capability", "complete files are not served"
            )
        locale = _resolve_locale(plugin, locale)
        request = _recognize_request(audio_wav, locale, service)
        owner = FileRequestOwner(
            service=service,
            engine_client=plugin.engine_client,
            config=plugin.config,
            clock=plugin.clock,
            registry=registry,
            record=record,
            catalog=_catalog(plugin),
            gate=gate,
            accounting=accounting,
            gone=_rpc_cancellation(context),
            lifecycle=plugin,
            state=plugin.state,
        )
    except BaseException:
        accounting.release()
        raise
    result = await owner.run(
        audio_wav, request, build_raw_request(), deadline=deadline
    )
    cause = owner.cause
    if cause is None:
        return _recognize_response(result, wav_seconds(audio_wav))
    if cause.kind == "transport":
        if owner.cancelled_by_task:
            raise asyncio.CancelledError
        await context.abort(
            grpc.StatusCode.CANCELLED, "cancelled: the client cancelled the RPC"
        )
        raise asyncio.CancelledError
    if cause.exception is not None:
        raise classify_core_exception(cause.exception)
    raise cause.error or _internal()
