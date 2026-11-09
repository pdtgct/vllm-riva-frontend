"""The Riva Realtime WebSocket dialect, driven over ``AsgiWebSocket``.

Event builders in the public shape (the field names of
``riva/client/realtime.py`` and the OpenAI Realtime events) and
``RealtimeClient``, the one driver every leaf uses for a claimed connection.
The shapes of the server's events are pinned in the implementation layout
(Shapes, Realtime events); ``error_code``, ``error_fields`` and
``error_authority`` read them from there and nowhere else.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Iterable
from typing import Any

from support.asgi import AsgiWebSocket
from support.fake_engine import DEFAULT_MODEL

REALTIME_PATH = "/v1/realtime"
REALTIME_URL = "/v1/realtime?intent=transcription"
BOOTSTRAP_PATH = "/v1/realtime/transcription_sessions"

#: The marker event ``RealtimeClient.next`` returns for the close.
CLOSE = "websocket.close"

CREATED = "conversation.created"
UPDATED = "transcription_session.updated"
DELTA = "conversation.item.input_audio_transcription.delta"
COMPLETED = "conversation.item.input_audio_transcription.completed"
FAILED = "conversation.item.input_audio_transcription.failed"
COMMITTED = "input_audio_buffer.committed"
CLEARED = "input_audio_buffer.cleared"
ERROR = "error"

#: Loop turns that let a cancel, its settlement and the cleanup calls run.
SETTLE = 60


# ---------------------------------------------------------------- builders


def default_session(
    model: str = DEFAULT_MODEL,
    language: str | None = "en-US",
    **overrides: Any,
) -> dict[str, Any]:
    """The transcription-session object the stock client echoes back.

    The public shape with every default value spelled out: the keys the
    bootstrap object carries plus the default-valued keys the stock client
    may add (``prompt``, the ``recognition_config`` toggles,
    ``max_speaker_count``). ``overrides`` replace top-level keys.
    """
    session: dict[str, Any] = {
        "modalities": ["text"],
        "input_audio_format": "pcm16",
        "input_audio_params": {"sample_rate_hz": 16_000, "num_channels": 1},
        "input_audio_transcription": {
            "language": "" if language is None else language,
            "model": model,
            "prompt": "",
        },
        "recognition_config": {
            "max_alternatives": 1,
            "enable_automatic_punctuation": False,
            "enable_word_time_offsets": False,
            "enable_profanity_filter": False,
            "enable_verbatim_transcripts": False,
        },
        "speaker_diarization": {
            "enable_speaker_diarization": False,
            "max_speaker_count": 0,
        },
        "word_boosting": {
            "enable_word_boosting": False,
            "word_boosting_list": [],
        },
        "endpointing_config": {},
        "client_secret": None,
    }
    session.update(overrides)
    return session


def session_cell(
    audio_format: str, sample_rate_hz: int, num_channels: int = 1, **kw: Any
) -> dict[str, Any]:
    """A default session declaring one audio cell."""
    return default_session(
        input_audio_format=audio_format,
        input_audio_params={
            "sample_rate_hz": sample_rate_hz,
            "num_channels": num_channels,
        },
        **kw,
    )


def update_event(session: dict[str, Any]) -> dict[str, Any]:
    return {"type": "transcription_session.update", "session": session}


def append_event(data: bytes) -> dict[str, Any]:
    return {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(data).decode(),
    }


def commit_event() -> dict[str, Any]:
    return {"type": "input_audio_buffer.commit"}


def clear_event() -> dict[str, Any]:
    return {"type": "input_audio_buffer.clear"}


def done_event() -> dict[str, Any]:
    return {"type": "input_audio_buffer.done"}


# ----------------------------------------------------------------- readers


def error_object(event: dict[str, Any]) -> dict[str, Any] | None:
    """The ``error`` object of an ``error`` or ``failed`` event."""
    error = event.get("error")
    return error if isinstance(error, dict) else None


def error_code(event: dict[str, Any]) -> str | None:
    """The catalog code: ``event["error"]["code"]``."""
    error = error_object(event)
    return None if error is None else error.get("code")


def error_fields(event: dict[str, Any]) -> list[str]:
    """The named fields: ``event["error"]["param"]`` split on commas."""
    error = error_object(event)
    param = None if error is None else error.get("param")
    if not param:
        return []
    return [name.strip() for name in str(param).split(",")]


def error_authority(event: dict[str, Any]) -> str | None:
    """``event["error"]["authority"]`` (present for ``busy`` only)."""
    error = error_object(event)
    return None if error is None else error.get("authority")


def is_close(event: dict[str, Any]) -> bool:
    return event.get("type") == CLOSE


# ------------------------------------------------------------------ client


class RealtimeClient:
    """One claimed Riva Realtime connection through the app.

    ``next()`` returns the next server event as a dict; the close arrives
    as ``{"type": "websocket.close", "code": n}`` and sets ``close_code``.
    ``open_session`` connects, reads ``conversation.created``, sends the
    first update and returns the ``transcription_session.updated`` event.
    ``settle``, ``advance`` and ``drain_events`` use ``clock`` when given.
    """

    def __init__(
        self,
        app: Any,
        clock: Any = None,
        *,
        url: str = REALTIME_URL,
        headers: Iterable[tuple[str, str]] = (),
        ws: AsgiWebSocket | None = None,
    ) -> None:
        self.clock = clock
        self.ws = (
            ws if ws is not None else AsgiWebSocket(app, url, headers=headers)
        )
        self.events: list[dict[str, Any]] = []
        self.close_code: int | None = None

    # -- connection -----------------------------------------------------------
    async def connect(self) -> bool:
        """``True`` when the app accepted the WebSocket."""
        return await self.ws.connect()

    @property
    def denied(self) -> int | None:
        """The HTTP status of a denied handshake, or ``None``."""
        denial = self.ws.denied
        return None if denial is None else denial.get("status")

    async def disconnect(self, code: int = 1000) -> None:
        await self.ws.disconnect(code)

    async def wait_closed(self, timeout: float = 5.0) -> None:
        await self.ws.wait_closed(timeout)

    @property
    def done(self) -> bool:
        return self.ws.done

    # -- receiving ------------------------------------------------------------
    async def next(self, timeout: float = 2.0) -> dict[str, Any]:
        """The next server event (see the class docstring)."""
        message = await self.ws.receive(timeout)
        return self._record(message)

    def _record(self, message: dict[str, Any]) -> dict[str, Any]:
        if message["type"] == CLOSE:
            self.close_code = message.get("code", 1000)
            event: dict[str, Any] = {"type": CLOSE, "code": self.close_code}
            if message.get("reason"):
                event["reason"] = message["reason"]
        elif message["type"] == "websocket.send":
            text = message.get("text")
            if text is None:
                text = message["bytes"].decode()
            event = json.loads(text)
        else:
            event = {"type": message["type"]}
        self.events.append(event)
        return event

    async def until_type(
        self, *types: str, timeout: float = 2.0
    ) -> dict[str, Any]:
        """Receive until an event of one of ``types`` (or the close)."""
        while True:
            event = await self.next(timeout)
            if event["type"] in types or event["type"] == CLOSE:
                return event

    async def drain_events(self, settle: int = SETTLE) -> list[dict[str, Any]]:
        """Every event already sent, without waiting for more."""
        await self.settle(settle)
        events: list[dict[str, Any]] = []
        while self.close_code is None:
            try:
                message = await self.ws.receive(timeout=0.05)
            except (TimeoutError, ConnectionError):
                break
            event = self._record(message)
            events.append(event)
            if event["type"] == CLOSE:
                break
        return events

    async def assert_closes(
        self, code: int, *, timeout: float = 2.0
    ) -> list[dict[str, Any]]:
        """Read until the close, assert its code; return the events before."""
        before: list[dict[str, Any]] = []
        while True:
            event = await self.next(timeout)
            if event["type"] == CLOSE:
                assert event["code"] == code, (event, before)
                return before
            before.append(event)

    async def outcome(self, timeout: float = 5.0) -> list[dict[str, Any]]:
        """Read to the close (or the app's end); return every event so far."""
        while self.close_code is None:
            try:
                event = await self.next(timeout)
            except ConnectionError:
                break
            if event["type"] == CLOSE:
                break
        return list(self.events)

    def of_type(self, kind: str) -> list[dict[str, Any]]:
        return [event for event in self.events if event.get("type") == kind]

    def types(self) -> list[str]:
        return [event.get("type") for event in self.events]

    @property
    def terminal(self) -> dict[str, Any] | None:
        """The last ``error`` or ``failed`` event received."""
        for event in reversed(self.events):
            if event.get("type") in (ERROR, FAILED):
                return event
        return None

    # -- sending --------------------------------------------------------------
    async def send(self, payload: Any) -> None:
        await self.ws.send_json(payload)

    async def send_text(self, text: str) -> None:
        await self.ws.send_text(text)

    async def send_bytes(self, data: bytes) -> None:
        await self.ws.send_bytes(data)

    async def update(self, session: dict[str, Any]) -> None:
        await self.send(update_event(session))

    async def append(self, data: bytes) -> None:
        await self.send(append_event(data))

    async def commit(self) -> None:
        await self.send(commit_event())

    async def clear(self) -> None:
        await self.send(clear_event())

    async def done_event(self) -> None:
        await self.send(done_event())

    async def open_session(
        self, session: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Connect, read ``conversation.created``, configure; return the ack."""
        assert await self.connect(), self.ws.denied
        created = await self.next()
        assert created["type"] == CREATED, created
        await self.update(session if session is not None else default_session())
        updated = await self.next()
        assert updated["type"] == UPDATED, updated
        return updated

    # -- time -----------------------------------------------------------------
    async def settle(self, rounds: int = SETTLE) -> None:
        if self.clock is not None:
            await self.clock.settle(rounds)
        else:
            for _ in range(rounds):
                await asyncio.sleep(0)

    async def advance(self, seconds: float) -> None:
        assert self.clock is not None, "advance needs the plugin's clock"
        await self.clock.advance(seconds, settle=SETTLE)


__all__ = [
    "BOOTSTRAP_PATH",
    "CLEARED",
    "CLOSE",
    "COMMITTED",
    "COMPLETED",
    "CREATED",
    "DELTA",
    "ERROR",
    "FAILED",
    "REALTIME_PATH",
    "REALTIME_URL",
    "SETTLE",
    "UPDATED",
    "RealtimeClient",
    "append_event",
    "clear_event",
    "commit_event",
    "default_session",
    "done_event",
    "error_authority",
    "error_code",
    "error_fields",
    "error_object",
    "is_close",
    "session_cell",
    "update_event",
]
