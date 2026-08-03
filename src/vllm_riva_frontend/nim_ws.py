"""NIM Realtime WebSocket compatibility over the direct RFC-1 lease.

This is raw ASGI middleware: it claims only NIM's exact transcription intent
and passes every other websocket scope to the host's native realtime route.
It never creates a second HTTP server or a transport-owned inference engine.
"""

import asyncio
import base64
import binascii
import contextlib
import json
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from enum import Enum
from typing import Any, TypeAlias
from urllib.parse import unquote_to_bytes

from vllm_riva_frontend.admission import (
    AdmissionLease,
    HostAdmission,
    LoadShedGate,
    LoadShedRegistration,
    LoadShedRejected,
    try_acquire_admission,
)
from vllm_riva_frontend.errors import catalog
from vllm_riva_frontend.frontend import (
    FormatError,
    RiffFormat,
    StreamingAudioFrontend,
    sniff_riff,
    validate_format,
)
from vllm_riva_frontend.lease import DirectLeaseOwner, SessionFactory

AsgiApp: TypeAlias = Callable[[object, object, object], Awaitable[None]]
AsgiReceive: TypeAlias = Callable[[], Awaitable[dict[str, object]]]
AsgiSend: TypeAlias = Callable[[dict[str, object]], Awaitable[None]]
OwnerRegister: TypeAlias = Callable[[], Awaitable[object]]
LeaseOwnerFactory: TypeAlias = Callable[..., DirectLeaseOwner]

_FORMAT_NAMES = {
    "pcm16": "LINEAR_PCM",
    "g711_ulaw": "MULAW",
    "g711_alaw": "ALAW",
}
_DEFERRED_FORMAT = "none"
_DELTA_EVENT = "conversation.item.input_audio_transcription.delta"


class DispatchDecision(Enum):
    """Exclusive scope outcome before either realtime dialect handles it."""

    CLAIM = "claim"
    PASS = "pass"
    DENY = "deny"


class _LifecycleTimeout(Exception):
    """One absolute session timer won while provider work was pending."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _event_id() -> str:
    """Return an opaque event identifier for one server event."""
    return f"event_{uuid.uuid4().hex}"


def _error(
    code: str, message: str, fields: list[str] | None = None
) -> dict[str, Any]:
    """Build the stable NIM protocol-error event."""
    projection = catalog().get(code)
    body: dict[str, Any] = {
        "type": projection.nim_event if projection is not None else "error",
        "error": {"code": code, "message": message},
    }
    if fields:
        body["error"]["param"] = ",".join(fields)
    return body


def _suffix(previous: str, current: str) -> str:
    """Project the true incremental suffix of cumulative hypotheses."""
    if current.startswith(previous):
        return current[len(previous) :]
    return current


def _structural_session_fields(session: Mapping[str, object]) -> list[str]:
    """Return every field whose JSON type cannot be interpreted safely."""
    invalid: list[str] = []
    scalar_strings = ("id", "object")
    invalid.extend(
        name
        for name in scalar_strings
        if name in session and not isinstance(session[name], str)
    )
    if "provenance" in session and not isinstance(session["provenance"], dict):
        invalid.append("provenance")
    modalities = session.get("modalities", ["text"])
    if not isinstance(modalities, list) or not all(
        isinstance(value, str) for value in modalities
    ):
        invalid.append("modalities")
    if not isinstance(session.get("input_audio_format", "pcm16"), str):
        invalid.append("input_audio_format")
    params = session.get("input_audio_params", {})
    if not isinstance(params, dict):
        invalid.append("input_audio_params")
    else:
        for name in ("sample_rate_hz", "num_channels"):
            value = params.get(name, 16000 if name == "sample_rate_hz" else 1)
            if type(value) is not int:
                invalid.append(f"input_audio_params.{name}")
    transcription = session.get("input_audio_transcription", {})
    if not isinstance(transcription, dict):
        invalid.append("input_audio_transcription")
    else:
        for name in ("language", "model", "prompt"):
            if name in transcription and not isinstance(
                transcription[name], str
            ):
                invalid.append(f"input_audio_transcription.{name}")
    recognition = session.get("recognition_config", {})
    if not isinstance(recognition, dict):
        invalid.append("recognition_config")
    else:
        for name, value in recognition.items():
            if name == "max_alternatives":
                if type(value) is not int:
                    invalid.append(f"recognition_config.{name}")
            elif type(value) is not bool:
                invalid.append(f"recognition_config.{name}")
    for name in ("speaker_diarization", "word_boosting", "endpointing_config"):
        value = session.get(name, {})
        if not isinstance(value, dict):
            invalid.append(name)
    word_boosting = session.get("word_boosting", {})
    if isinstance(word_boosting, dict):
        enabled = word_boosting.get("enable_word_boosting", False)
        words = word_boosting.get("word_boosting_list", [])
        if type(enabled) is not bool:
            invalid.append("word_boosting.enable_word_boosting")
        if not isinstance(words, list) or not all(
            isinstance(word, str) for word in words
        ):
            invalid.append("word_boosting.word_boosting_list")
    diarization = session.get("speaker_diarization", {})
    if isinstance(diarization, dict):
        enabled = diarization.get("enable_speaker_diarization", False)
        if type(enabled) is not bool:
            invalid.append("speaker_diarization.enable_speaker_diarization")
    return list(dict.fromkeys(invalid))


def _session_rejections(
    session: Mapping[str, object],
) -> list[tuple[str, str]]:
    """Return every semantic config rejection in dialect-native spelling."""
    allowed = {
        "id",
        "object",
        "client_secret",
        "provenance",
        "modalities",
        "input_audio_format",
        "input_audio_params",
        "input_audio_transcription",
        "recognition_config",
        "speaker_diarization",
        "word_boosting",
        "endpointing_config",
    }
    rejected: list[tuple[str, str]] = [
        ("invalid_config_field", key) for key in session if key not in allowed
    ]
    if session.get("modalities", ["text"]) != ["text"]:
        rejected.append(("invalid_config_field", "modalities"))
    params = session.get("input_audio_params", {})
    assert isinstance(params, dict)
    rejected.extend(
        ("invalid_config_field", f"input_audio_params.{key}")
        for key in params
        if key not in {"sample_rate_hz", "num_channels"}
    )
    transcription = session.get("input_audio_transcription", {})
    assert isinstance(transcription, dict)
    rejected.extend(
        ("invalid_config_field", f"input_audio_transcription.{key}")
        for key in transcription
        if key not in {"language", "model", "prompt"}
    )
    if transcription.get("prompt"):
        rejected.append(
            ("unsupported_capability", "input_audio_transcription.prompt")
        )
    for name in ("speaker_diarization", "word_boosting", "endpointing_config"):
        value = session.get(name, {})
        assert isinstance(value, dict)
        if any(bool(item) for item in value.values()):
            rejected.append(("unsupported_capability", name))
    recognition = session.get("recognition_config", {})
    assert isinstance(recognition, dict)
    supported = {
        "max_alternatives",
        "enable_automatic_punctuation",
        "enable_verbatim_transcripts",
        "enable_word_time_offsets",
        "enable_profanity_filter",
    }
    rejected.extend(
        ("invalid_config_field", f"recognition_config.{key}")
        for key in recognition
        if key not in supported
    )
    if recognition.get("max_alternatives", 1) > 1:
        rejected.append(
            ("invalid_config_field", "recognition_config.max_alternatives")
        )
    if recognition.get("enable_word_time_offsets"):
        rejected.append(
            (
                "unsupported_capability",
                "recognition_config.enable_word_time_offsets",
            )
        )
    if recognition.get("enable_profanity_filter"):
        rejected.append(
            (
                "invalid_config_field",
                "recognition_config.enable_profanity_filter",
            )
        )
    return list(dict.fromkeys(rejected))


def _decode_query(query: bytes) -> list[tuple[str, str]] | None:
    """Strictly percent/UTF-8 decode a raw ASGI query string."""
    pairs: list[tuple[str, str]] = []
    for part in query.split(b"&"):
        key, separator, value = part.partition(b"=")
        if not separator:
            value = b""
        try:
            decoded_key = unquote_to_bytes(key).decode("utf-8", "strict")
            decoded_value = unquote_to_bytes(value).decode("utf-8", "strict")
        except UnicodeDecodeError:
            return None
        if b"%" in key or b"%" in value:
            # urllib leaves malformed percent escapes unchanged; reject rather
            # than accidentally falling through to the RFC-1 dialect.
            for raw in (key, value):
                index = 0
                while index < len(raw):
                    if raw[index : index + 1] == b"%":
                        if index + 2 >= len(raw) or any(
                            char not in b"0123456789abcdefABCDEF"
                            for char in raw[index + 1 : index + 3]
                        ):
                            return None
                        index += 3
                    else:
                        index += 1
        pairs.append((decoded_key, decoded_value))
    return pairs


# @spec ING-NIMWS-002
def dispatch_decision(scope: Mapping[str, object]) -> DispatchDecision:
    """Claim only `/v1/realtime?intent=transcription` exactly once."""
    if scope.get("type") != "websocket" or scope.get("path") != "/v1/realtime":
        return DispatchDecision.PASS
    query = scope.get("query_string", b"")
    if not isinstance(query, bytes):
        return DispatchDecision.DENY
    decoded = _decode_query(query)
    if decoded is None:
        return DispatchDecision.DENY
    if decoded == [("intent", "transcription")]:
        return DispatchDecision.CLAIM
    return DispatchDecision.PASS


# @spec ING-NIMWS-001
def bootstrap_session(model_name: str) -> dict[str, object]:
    """Return a canonical client-echoable NIM transcription-session object."""
    if not model_name:
        raise ValueError("model_name must be non-empty")
    return {
        "id": f"sess_{uuid.uuid4().hex}",
        "object": "realtime.transcription_session",
        "modalities": ["text"],
        "input_audio_format": "pcm16",
        "input_audio_transcription": {
            "language": "auto",
            "model": model_name,
        },
        "input_audio_params": {"sample_rate_hz": 16000, "num_channels": 1},
        "recognition_config": {"max_alternatives": 1},
        "speaker_diarization": {"enable_speaker_diarization": False},
        "word_boosting": {
            "enable_word_boosting": False,
            "word_boosting_list": [],
        },
        "endpointing_config": {},
        "client_secret": None,
    }


# @spec ING-CORE-005, ING-NIMWS-004, ING-NIMWS-005, ING-NIMWS-006
def project_event(event: Mapping[str, object]) -> dict[str, object]:
    """Project one small sans-I/O event for unit-level wire assertions."""
    event_type = event.get("type")
    if event_type == "connect":
        return {
            "type": "conversation.created",
            "conversation": {"object": "realtime.conversation"},
        }
    if event_type == "hypothesis":
        return {
            "type": "conversation.item.input_audio_transcription.delta",
            "delta": _suffix(
                str(event.get("previous", "")), str(event.get("current", ""))
            ),
        }
    if event_type == "input_audio_buffer.done":
        return {
            "type": "conversation.item.input_audio_transcription.completed",
            "is_last_result": True,
        }
    if event_type == "input_audio_buffer.clear":
        return {"type": "input_audio_buffer.cleared", "rewound": False}
    if event_type == "transcription_session.update":
        fmt = str(event.get("format", "pcm16"))
        rate = event.get("rate", 16000)
        if type(rate) is not int:
            return _error(
                "invalid_event",
                "rate must be an integer",
                ["rate"],
            )
        mapped = _FORMAT_NAMES.get(fmt, fmt)
        rejection = validate_format(mapped, rate)
        if rejection is not None:
            return {"code": rejection.code, "fields": list(rejection.fields)}
        return {
            "type": "transcription_session.updated",
            "locale": str(event.get("locale", "auto")),
        }
    if event_type == "idle_timeout":
        return {"code": "idle_timeout", "aborted": True}
    return _error("invalid_event", f"unknown event type {event_type!r}")


class _Connection:
    """One raw-ASGI NIM session, serialized through one direct lease owner."""

    def __init__(
        self,
        factory: SessionFactory,
        config: object,
        gate: LoadShedGate,
        owner_register: OwnerRegister | None,
        owner_factory: LeaseOwnerFactory,
        admission_lease: AdmissionLease,
    ) -> None:
        """Capture shared factory/config and initialize local buffer state."""
        self._factory = factory
        self._config = config
        self._gate = gate
        self._owner_register = owner_register
        self._owner_factory = owner_factory
        self._admission_lease: AdmissionLease | None = admission_lease
        self._registration: LoadShedRegistration | None = None
        self._tracked_owner: object | None = None
        self._owner: DirectLeaseOwner | None = None
        self._frontend: StreamingAudioFrontend | None = None
        self._deferred = False
        self._riff_bytes = b""
        self._riff_remaining: int | None = None
        self._configured = False
        self._terminal = False
        self._last_hypothesis = ""
        self._configured_at: float | None = None
        self._preconfiguration_deadline = time.monotonic() + float(
            getattr(config, "preconfiguration_timeout", 30.0)
        )
        self._last_accepted_audio_at: float | None = None
        self._format_spec: tuple[str, int] | None = None
        self._declared_rate = 16000
        self._declared_channels = 1
        self._initial_session: dict[str, object] | None = None
        self._item_id = f"item_{uuid.uuid4().hex}"

    def _activity_deadline(self) -> tuple[float, str]:
        """Return the current accepted-audio idle/max-duration deadline."""
        assert self._configured_at is not None
        accepted_at = self._last_accepted_audio_at or self._configured_at
        idle = accepted_at + float(
            getattr(self._config, "session_idle_timeout", 30.0)
        )
        maximum = getattr(self._config, "max_session_duration", None)
        if maximum is not None:
            duration = self._configured_at + float(maximum)
            if duration <= idle:
                return duration, "request_timeout"
        return idle, "idle_timeout"

    async def _await_active(
        self,
        operation: Awaitable[Any],
        *,
        accepted_event: asyncio.Event | None = None,
    ) -> Any:
        """Bound provider work and recompute after exact receipt credit."""
        task = asyncio.ensure_future(operation)
        try:
            while True:
                deadline, code = self._activity_deadline()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _LifecycleTimeout(code)
                if accepted_event is None:
                    try:
                        return await asyncio.wait_for(
                            asyncio.shield(task), remaining
                        )
                    except asyncio.TimeoutError as error:
                        raise _LifecycleTimeout(code) from error
                accepted_wait = asyncio.create_task(accepted_event.wait())
                done, _ = await asyncio.wait(
                    {task, accepted_wait},
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
                raise _LifecycleTimeout(code)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def run(self, receive: AsgiReceive, send: AsgiSend) -> None:
        """Accept, configure once, process FIFO events, and clean up on drop."""
        await send({"type": "websocket.accept"})
        registration = await self._gate.register("nim_realtime_transcription")
        if isinstance(registration, LoadShedRejected):
            await self._send_json(
                send,
                _error(
                    registration.code,
                    f"authority: {registration.authority}",
                ),
            )
            await send({"type": "websocket.close", "code": 1013})
            await self._release_registration()
            return
        if registration is None:
            raise RuntimeError("realtime inference must consume one owner slot")
        assert isinstance(registration, LoadShedRegistration)
        self._registration = registration
        if self._owner_register is not None:
            self._tracked_owner = await self._owner_register()
        await self._send_json(
            send,
            {
                "type": "conversation.created",
                "event_id": _event_id(),
                "conversation": {
                    "id": f"conv_{uuid.uuid4().hex}",
                    "object": "realtime.conversation",
                },
            },
        )
        timeout_code = "configuration_timeout"
        try:
            while not self._terminal:
                now = time.monotonic()
                max_duration = getattr(
                    self._config, "max_session_duration", None
                )
                if (
                    self._configured_at is not None
                    and max_duration is not None
                    and now - self._configured_at >= float(max_duration)
                ):
                    await self._fail(
                        send,
                        "request_timeout",
                        "session duration exceeded",
                        close=1008,
                    )
                    return
                if self._configured:
                    accepted_at = self._last_accepted_audio_at
                    if accepted_at is None:
                        accepted_at = self._configured_at
                    assert accepted_at is not None
                    deadline = accepted_at + float(
                        getattr(self._config, "session_idle_timeout", 30.0)
                    )
                    timeout_code = "idle_timeout"
                    max_duration = getattr(
                        self._config, "max_session_duration", None
                    )
                    if max_duration is not None:
                        assert self._configured_at is not None
                        duration_deadline = self._configured_at + float(
                            max_duration
                        )
                        if duration_deadline < deadline:
                            deadline = duration_deadline
                            timeout_code = "request_timeout"
                else:
                    deadline = self._preconfiguration_deadline
                    timeout_code = "configuration_timeout"
                timeout = deadline - now
                if timeout <= 0:
                    await self._fail(
                        send,
                        timeout_code,
                        "session receive timeout",
                        close=1008,
                    )
                    return
                message = await asyncio.wait_for(receive(), timeout)
                message_type = message.get("type")
                if message_type == "websocket.disconnect":
                    await self._abort()
                    return
                if message_type != "websocket.receive":
                    continue
                text = message.get("text")
                if not isinstance(text, str):
                    await self._fail(
                        send,
                        "invalid_event",
                        "text JSON event required",
                        close=1008,
                    )
                    return
                if len(text.encode()) > int(
                    getattr(
                        self._config, "ws_event_envelope_max_bytes", 1 << 20
                    )
                ):
                    await self._fail(
                        send,
                        "request_too_large",
                        "event envelope exceeds limit",
                        close=1009,
                    )
                    return
                try:
                    event = json.loads(text)
                except json.JSONDecodeError:
                    await self._fail(
                        send,
                        "invalid_event",
                        "event is not valid JSON",
                        close=1008,
                    )
                    return
                if not isinstance(event, dict):
                    await self._fail(
                        send,
                        "invalid_event",
                        "event must be an object",
                        close=1008,
                    )
                    return
                await self._handle(event, send)
        except asyncio.CancelledError:
            # Plugin shutdown cancels the ASGI owner task. Preserve that
            # cancellation as the primary cause, but do not release owner
            # tracking until the lease has attempted abort → release.
            with contextlib.suppress(BaseException):
                await self._abort()
            raise
        except asyncio.TimeoutError:
            await self._fail(
                send,
                timeout_code,
                "session receive timeout",
                close=1008,
            )
        except Exception:
            await self._abort()
            await self._fail(send, "internal", "adapter failure", close=1011)
        finally:
            await self._release_registration()

    async def _handle(self, event: dict[str, object], send: AsgiSend) -> None:
        """Project one client event without reading ahead of lease work."""
        event_type = event.get("type")
        if (
            not self._configured
            and event_type != "transcription_session.update"
        ):
            await self._fail(
                send,
                "protocol_order",
                "first event must configure the transcription session",
                close=1008,
            )
            return
        if event_type == "transcription_session.update":
            await self._update(event, send)
        elif event_type == "input_audio_buffer.append":
            await self._append(event, send)
        elif event_type == "input_audio_buffer.commit":
            await self._send_json(
                send,
                {
                    "type": "input_audio_buffer.committed",
                    "event_id": _event_id(),
                    "item_id": self._item_id,
                    "content_index": 0,
                },
            )
        elif event_type == "input_audio_buffer.clear":
            self._riff_bytes = b""
            self._riff_remaining = None
            if self._deferred:
                self._frontend = None
            elif self._format_spec is not None:
                encoding, rate = self._format_spec
                self._frontend = StreamingAudioFrontend(
                    encoding=encoding, sample_rate_hz=rate
                )
            await self._send_json(
                send,
                {"type": "input_audio_buffer.cleared", "event_id": _event_id()},
            )
        elif event_type == "input_audio_buffer.done":
            await self._done(send)
        else:
            await self._fail(
                send,
                "invalid_event",
                f"unknown event type {event_type!r}",
                close=1008,
            )

    async def _update(self, event: dict[str, object], send: AsgiSend) -> None:
        """Open on first update; later updates can change locale."""
        session = event.get("session")
        if not isinstance(session, dict):
            await self._fail(
                send, "invalid_event", "session object required", close=1008
            )
            return
        structural_fields = _structural_session_fields(session)
        if structural_fields:
            await self._fail(
                send,
                "invalid_event",
                "invalid session field types",
                structural_fields,
                close=1008,
            )
            return
        transcription = session.get("input_audio_transcription", {})
        params = session.get("input_audio_params", {})
        assert isinstance(transcription, dict)
        assert isinstance(params, dict)
        semantic_rejections = _session_rejections(session)
        initial_transcription = (
            (self._initial_session or {}).get("input_audio_transcription", {})
            if self._configured
            else {}
        )
        assert isinstance(initial_transcription, dict)
        old_locale = str(initial_transcription.get("language", "auto"))
        locale = str(
            transcription.get(
                "language", old_locale if self._configured else "auto"
            )
        )
        served_model = getattr(self._config, "model_name", None)
        requested_model = transcription.get("model")
        locales = getattr(self._config, "locales", None)
        selector_rejections: list[tuple[str, str]] = []
        if served_model is not None and requested_model not in (
            None,
            served_model,
        ):
            selector_rejections.append(
                ("invalid_config_field", "input_audio_transcription.model")
            )
        if locales is not None and locale not in set(locales) | {"auto"}:
            selector_rejections.append(
                ("unknown_locale", "input_audio_transcription.language")
            )
        if not self._configured and (
            semantic_rejections or selector_rejections
        ):
            for code, field in semantic_rejections + selector_rejections:
                await self._fail(
                    send,
                    code,
                    "unsupported model or locale selector",
                    [field],
                )
            return
        if self._configured:
            if self._owner is None:
                await self._fail(
                    send, "internal", "missing lease owner", close=1011
                )
                return
            initial = self._initial_session or {}
            immutable = (
                "input_audio_format",
                "input_audio_params",
                "recognition_config",
                "speaker_diarization",
                "word_boosting",
                "endpointing_config",
            )
            changed = [
                name
                for name in immutable
                if name in session and session.get(name) != initial.get(name)
            ]
            for name in ("model", "prompt"):
                if name in transcription and transcription.get(
                    name
                ) != initial_transcription.get(name):
                    changed.append(f"input_audio_transcription.{name}")
            for code, field in semantic_rejections + selector_rejections:
                await self._fail(
                    send,
                    code,
                    "unsupported session field",
                    [field],
                )
            if changed:
                await self._fail(
                    send,
                    "config_change_rejected",
                    "field is immutable after session open",
                    changed,
                )
            if any(code == "unknown_locale" for code, _ in selector_rejections):
                return
            if locale == old_locale:
                await self._updated(send, old_locale)
                return
            try:
                await self._await_active(self._owner.update_locale(locale))
            except _LifecycleTimeout as error:
                await self._fail(
                    send,
                    error.code,
                    "session lifetime expired during locale update",
                    close=1008,
                )
                return
            except RuntimeError:
                await self._fail(
                    send, "session_terminal", "session is terminal", close=1008
                )
                return
            initial_transcription["language"] = locale
            initial["input_audio_transcription"] = initial_transcription
            await self._updated(send, locale)
            return

        fmt = str(session.get("input_audio_format", "pcm16"))
        rate = int(params.get("sample_rate_hz", 16000))
        channels = int(params.get("num_channels", 1))
        self._declared_rate = rate
        self._declared_channels = channels
        if fmt == _DEFERRED_FORMAT:
            self._deferred = True
        else:
            mapped = _FORMAT_NAMES.get(fmt, fmt)
            rejection = validate_format(mapped, rate, channels)
            if rejection is not None:
                await self._format_failure(send, rejection)
                return
            self._frontend = StreamingAudioFrontend(
                encoding=mapped, sample_rate_hz=rate
            )
            self._format_spec = (mapped, rate)
        self._owner = self._owner_factory(
            self._factory,
            cleanup_timeout=float(
                getattr(self._config, "session_cleanup_timeout", 5.0)
            ),
        )
        try:
            remaining = self._preconfiguration_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            await asyncio.wait_for(
                self._owner.open(cadence="560ms", locale=locale),
                remaining,
            )
        except asyncio.TimeoutError:
            await self._fail(
                send,
                "configuration_timeout",
                "session configuration and open timed out",
                close=1011,
            )
            return
        except Exception:
            await self._fail(send, "internal", "lease open failed", close=1011)
            return
        self._configured = True
        effective_model = (
            served_model
            if isinstance(served_model, str) and served_model
            else (
                requested_model
                if isinstance(requested_model, str) and requested_model
                else "served"
            )
        )
        effective = bootstrap_session(effective_model)
        for name, value in session.items():
            effective[name] = json.loads(json.dumps(value))
        effective_transcription = effective["input_audio_transcription"]
        assert isinstance(effective_transcription, dict)
        effective_transcription["language"] = locale
        effective_transcription["model"] = effective_model
        provenance = getattr(self._config, "provenance", None)
        if "provenance" not in session and isinstance(provenance, dict):
            # Deployment-owned path only: `provenance` here is always the
            # output of build_deployment_provenance (allowlist-built from
            # DeploymentMetadata), never an arbitrary copied mapping, so
            # embedding it verbatim is safe.  A client-supplied
            # `session["provenance"]` was already copied into `effective`
            # above and is intentionally left untouched -- the ING-NIMWS-008
            # echo-key contract requires echoing client session fields, not
            # rewriting them.
            effective["provenance"] = dict(provenance)
        self._initial_session = effective
        self._configured_at = time.monotonic()
        self._last_accepted_audio_at = self._configured_at
        await self._updated(send, locale)

    async def _append(self, event: dict[str, object], send: AsgiSend) -> None:
        """Base64-decode one FIFO append and await its lease feed."""
        if not self._configured:
            await self._fail(
                send, "protocol_order", "audio before session update"
            )
            return
        payload = event.get("audio")
        if not isinstance(payload, str):
            await self._fail(send, "invalid_audio", "audio must be base64")
            return
        try:
            raw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            await self._fail(send, "invalid_audio", "audio must be base64")
            return
        if self._deferred and self._frontend is None:
            riff_candidate = self._riff_bytes + raw
            result = sniff_riff(
                riff_candidate,
                max_header_bytes=int(
                    getattr(self._config, "max_riff_header_bytes", 65536)
                ),
            )
            if result is None:
                self._riff_bytes = riff_candidate
                return
            if isinstance(result, FormatError):
                await self._format_failure(send, result)
                return
            await self._resolve_riff(result, send)
            if self._terminal:
                return
            raw = riff_candidate[result.data_offset :]
            self._riff_bytes = b""
        if self._riff_remaining is not None:
            raw = raw[: self._riff_remaining]
            self._riff_remaining -= len(raw)
        if self._frontend is None or self._owner is None:
            await self._fail(
                send, "internal", "missing configured audio state", close=1011
            )
            return
        samples = self._frontend.push(raw)
        cap = int(getattr(self._config, "pre_submit_max_samples", 1 << 30))
        for start in range(0, len(samples), cap):
            await self._feed(samples[start : start + cap], send)
            if self._terminal:
                return

    async def _resolve_riff(self, result: RiffFormat, send: AsgiSend) -> None:
        """Bind deferred file mode once its header proves a valid cell."""
        fields: list[str] = []
        if result.sample_rate_hz != self._declared_rate:
            fields.append("input_audio_params.sample_rate_hz")
        if result.channels != self._declared_channels:
            fields.append("input_audio_params.num_channels")
        if fields:
            await self._fail(
                send,
                "unsupported_format",
                "RIFF format contradicts declared input_audio_params",
                fields,
                close=1008,
            )
            return
        rejection = validate_format(
            result.encoding, result.sample_rate_hz, result.channels
        )
        if rejection is not None:
            await self._format_failure(send, rejection)
            return
        self._frontend = StreamingAudioFrontend(
            encoding=result.encoding, sample_rate_hz=result.sample_rate_hz
        )
        self._format_spec = (result.encoding, result.sample_rate_hz)
        self._riff_remaining = result.data_bytes

    async def _feed(
        self,
        samples: object,
        send: AsgiSend,
        *,
        finalization_deadline: float | None = None,
    ) -> None:
        """Submit one normalized FIFO piece and emit only changed suffixes."""
        if self._owner is None:
            return
        accepted: int | None = None
        accepted_event = asyncio.Event()

        def accepted_callback(credit: int) -> None:
            nonlocal accepted
            if type(credit) is not int or credit != len(samples):  # type: ignore[arg-type]
                raise RuntimeError("invalid provider acceptance credit")
            if accepted is not None:
                raise RuntimeError("duplicate provider acceptance credit")
            accepted = credit
            self._last_accepted_audio_at = time.monotonic()
            accepted_event.set()

        try:
            operation = self._owner.feed(samples, on_accepted=accepted_callback)
            if finalization_deadline is None:
                hypotheses = await self._await_active(
                    operation,
                    accepted_event=accepted_event,
                )
            else:
                hypotheses = await asyncio.wait_for(
                    operation,
                    finalization_deadline - time.monotonic(),
                )
        except _LifecycleTimeout as error:
            await self._fail(
                send,
                error.code,
                "session lifetime expired during audio handoff",
                close=1008,
            )
            return
        except asyncio.TimeoutError:
            await self._fail(
                send,
                "finalization_timeout",
                "terminal audio handoff timed out",
                close=1011,
            )
            return
        except Exception:
            await self._fail(send, "internal", "lease feed failed", close=1011)
            return
        if type(accepted) is not int or accepted != len(samples):  # type: ignore[arg-type]
            await self._fail(
                send,
                "internal",
                "lease returned invalid acceptance credit",
                close=1011,
            )
            return
        for hypothesis in hypotheses:
            suffix = _suffix(self._last_hypothesis, hypothesis)
            self._last_hypothesis = hypothesis
            if suffix:
                await self._send_json(
                    send,
                    {
                        "type": _DELTA_EVENT,
                        "event_id": _event_id(),
                        "item_id": self._item_id,
                        "content_index": 0,
                        "delta": suffix,
                    },
                )

    async def _done(self, send: AsgiSend) -> None:
        """Flush frontend/lease once, then emit one completion and close."""
        if not self._configured or self._owner is None:
            await self._fail(
                send, "protocol_order", "done before session update"
            )
            return
        if self._riff_remaining not in {None, 0}:
            await self._fail(
                send,
                "invalid_audio",
                "truncated RIFF data",
                close=1008,
            )
            return
        finalization_deadline = time.monotonic() + float(
            getattr(self._config, "session_finalization_timeout", 30.0)
        )
        if self._configured_at is not None:
            limit = getattr(self._config, "max_session_duration", None)
            if (
                limit is not None
                and time.monotonic() - self._configured_at > limit
            ):
                await self._fail(
                    send,
                    "request_timeout",
                    "session duration exceeded",
                    close=1008,
                )
                return
        if self._frontend is not None:
            tail = self._frontend.flush()
            if len(tail):
                await self._feed(
                    tail,
                    send,
                    finalization_deadline=finalization_deadline,
                )
                if self._terminal:
                    return
        try:
            transcript = await asyncio.wait_for(
                self._owner.complete(),
                finalization_deadline - time.monotonic(),
            )
        except asyncio.TimeoutError:
            await self._fail(
                send,
                "finalization_timeout",
                "terminal drain timed out",
                close=1011,
            )
            return
        except Exception:
            await self._fail(
                send, "internal", "normal completion failed", close=1011
            )
            return
        if transcript is None:
            await self._fail(
                send, "internal", "normal completion failed", close=1011
            )
            return
        suffix = _suffix(self._last_hypothesis, transcript)
        if suffix:
            await self._send_json(
                send,
                {
                    "type": _DELTA_EVENT,
                    "event_id": _event_id(),
                    "item_id": self._item_id,
                    "content_index": 0,
                    "delta": suffix,
                },
            )
        await self._send_json(
            send,
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "event_id": _event_id(),
                "item_id": self._item_id,
                "content_index": 0,
                "transcript": transcript,
                "is_last_result": True,
            },
        )
        self._terminal = True
        await send({"type": "websocket.close", "code": 1000})
        await self._release_registration()

    async def _format_failure(self, send: AsgiSend, error: FormatError) -> None:
        """Project frontend rejection without retaining an open owner."""
        await self._fail(
            send,
            error.code,
            error.detail or "unsupported audio",
            list(error.fields),
            close=1008 if self._configured else None,
        )

    async def _updated(self, send: AsgiSend, locale: str) -> None:
        """Acknowledge the effective echoable session without admission."""
        session = json.loads(json.dumps(self._initial_session or {}))
        transcription = session.get("input_audio_transcription", {})
        if isinstance(transcription, dict):
            transcription["language"] = locale
            session["input_audio_transcription"] = transcription
        await self._send_json(
            send,
            {
                "type": "transcription_session.updated",
                "event_id": _event_id(),
                "session": session,
            },
        )

    async def _fail(
        self,
        send: AsgiSend,
        code: str,
        message: str,
        fields: list[str] | None = None,
        *,
        close: int | None = None,
    ) -> None:
        """Settle a terminal lease, then send its stable error and close."""
        event = _error(code, message, fields)
        if event["type"] != "error":
            event["event_id"] = _event_id()
            event["item_id"] = self._item_id
            event["content_index"] = 0
        if close is not None:
            self._terminal = True
            await self._abort()
        await self._send_json(send, event)
        if close is not None:
            await send({"type": "websocket.close", "code": close})

    async def _abort(self) -> None:
        """Select abnormal cleanup for an owned lease exactly once."""
        if self._owner is not None:
            # DirectLeaseOwner reports terminal cleanup faults to the plugin
            # lifetime. They remain secondary to the transport/protocol cause
            # already selected by this adapter.
            with contextlib.suppress(BaseException):
                await self._owner.cancel()
        await self._release_registration()

    async def _release_registration(self) -> None:
        """Release local owners before the host admission lease."""
        try:
            if self._tracked_owner is not None:
                release = getattr(self._tracked_owner, "release", None)
                if callable(release):
                    await release()
                self._tracked_owner = None
        finally:
            try:
                if self._registration is not None:
                    await self._registration.release()
                    self._registration = None
            finally:
                if self._admission_lease is not None:
                    self._admission_lease.release()
                    self._admission_lease = None

    async def _send_json(self, send: AsgiSend, event: dict[str, Any]) -> None:
        """Serialize one server event through the existing ASGI socket only."""
        await send({"type": "websocket.send", "text": json.dumps(event)})


# @spec ING-NIMWS-002, ING-VEH-014
def dispatch_nim_realtime(
    *,
    app: AsgiApp,
    factory: object,
    config: object,
    admission: HostAdmission | None = None,
    gate: LoadShedGate | None = None,
    owner_register: OwnerRegister | None = None,
    owner_factory: LeaseOwnerFactory | None = None,
) -> AsgiApp:
    """Wrap the host ASGI app with the exact-query NIM realtime dispatcher."""
    # Protocol runtime checks are not available; retain a narrow failure
    # message for a miswired plugin context without imposing a concrete type.
    if not hasattr(factory, "open"):
        raise TypeError("factory must expose async open")
    # Lifecycle supplies the process-wide gate. The fallback is only for
    # direct unit construction before the plugin lifetime is installed.
    shared_gate = gate or LoadShedGate(
        int(getattr(config, "load_shed_max_sessions", 1))
    )
    shared_owner_factory = owner_factory or (
        lambda factory, *, cleanup_timeout: DirectLeaseOwner(
            factory, cleanup_timeout=cleanup_timeout
        )
    )

    async def middleware(scope: object, receive: object, send: object) -> None:
        """Claim NIM, deny bad query bytes, otherwise preserve host scope."""
        if (
            not isinstance(scope, dict)
            or not callable(receive)
            or not callable(send)
        ):
            raise TypeError("raw ASGI scope, receive, and send are required")
        decision = dispatch_decision(scope)
        if decision is DispatchDecision.PASS:
            await app(scope, receive, send)
            return
        if decision is DispatchDecision.DENY:
            await send({"type": "websocket.close", "code": 1008})
            return
        admission_lease = try_acquire_admission(admission)
        if admission_lease is None:
            await send({"type": "websocket.close"})
            return
        connection = _Connection(
            factory,
            config,
            shared_gate,
            owner_register,
            shared_owner_factory,
            admission_lease,
        )
        try:
            await connection.run(receive, send)
        finally:
            await connection._release_registration()

    return middleware
