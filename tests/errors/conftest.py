"""Fixtures for the errors leaf.

The errors leaf checks the catalog and its projections as units, and the
phase table end to end on both streaming surfaces: the Riva servicer called
directly (``support.riva.StreamingCall``) and the Riva Realtime adapter
reached through its WebSocket route ``/plugins/riva/v1/realtime`` on
``test_app`` (``support.realtime.RealtimeClient``). Time is the plugin's
``FakeClock``; the plugin is started (admission opens last in ``start``),
so the gRPC listener binds a free port.

``riva_options`` (shared) is the ``riva_frontend`` mapping a test adds to the
defaults (a module or test overrides it). ``stream`` builds a driver for a
surface, so one test body can assert one phase-table row on both surfaces
through ``Outcome`` and ``assert_row``, which are specific to this leaf.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from support.clock import until
from support.realtime import (
    COMPLETED,
    CREATED,
    ERROR,
    FAILED,
    REALTIME_URL,
    UPDATED,
    RealtimeClient,
    default_session,
    error_authority,
    error_code,
    error_fields,
)
from support.riva import (
    StreamingCall,
    audio_request,
    config_request,
    recognition_config,
    recognize,
    recognize_request,
)

SURFACES = ("grpc", "realtime")
FAILED_EVENT = FAILED
COMPLETED_EVENT = COMPLETED


# ------------------------------------------------------------------ plugin
#
# ``riva_options``, ``fake_engine``, ``plugin`` and ``started`` are the shared
# fixtures of ``tests/conftest.py``; this leaf serves the plugin.


@pytest.fixture
def plugins(plugin):
    return [plugin]


# ----------------------------------------------------------------- helpers


def status_name(status: Any) -> str:
    """A gRPC status (enum or name) as its name."""
    return getattr(status, "name", str(status))


def speech_bytes(seconds: float = 0.1) -> bytes:
    from support.audio import pcm16_bytes, pcm16_sine

    return pcm16_bytes(pcm16_sine(seconds))


@dataclass
class Outcome:
    """How one stream ended, on either surface."""

    surface: str
    status: str | None = None
    details: str = ""
    responses: list[Any] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    close: int | None = None

    @property
    def finals(self) -> list[Any]:
        if self.surface == "grpc":
            return [
                r for r in self.responses if any(x.is_final for x in r.results)
            ]
        return [e for e in self.events if e.get("type") == COMPLETED]

    @property
    def terminal(self) -> dict[str, Any] | None:
        """The last ``error`` or ``failed`` event (Realtime)."""
        for event in reversed(self.events):
            if event.get("type") in (ERROR, FAILED):
                return event
        return None

    @property
    def code(self) -> str | None:
        """The catalog code the client saw."""
        if self.surface == "grpc":
            if not self.details:
                return None
            return self.details.split(":", 1)[0]
        terminal = self.terminal
        return None if terminal is None else error_code(terminal)


def assert_row(
    outcome: Outcome,
    code: str,
    grpc_status: str,
    event: str,
    close: int,
) -> None:
    """Assert one phase-table row on the surface the outcome came from."""
    if outcome.surface == "grpc":
        assert outcome.status == grpc_status, outcome
        assert outcome.details.startswith(f"{code}: "), outcome.details
    else:
        terminal = outcome.terminal
        assert terminal is not None, outcome.events
        assert terminal["type"] == event, terminal
        assert error_code(terminal) == code, terminal
        assert outcome.close == close, outcome
        # every pending event precedes the close: nothing follows the
        # terminal event except the close itself
        assert outcome.events[-1] is terminal


class GrpcStream:
    """One ``StreamingRecognize`` call on the servicer (``StreamingCall``)."""

    surface = "grpc"

    def __init__(self, plugin: Any, *, time_remaining: float | None = None):
        self.plugin = plugin
        self.call = StreamingCall(
            plugin.servicer,
            plugin.clock,
            time_remaining=time_remaining,
            start=False,
        )

    @property
    def requests(self) -> Any:
        return self.call.requests

    @property
    def context(self) -> Any:
        return self.call.context

    @property
    def responses(self) -> list[Any]:
        return self.call.responses

    def begin(self) -> None:
        self.call.start()

    async def configure(self, **fields: Any) -> None:
        """Send the configuration and wait until ``open`` returned."""
        leases = len(self.plugin.factory.leases)
        self.call.configure(**fields)
        self.begin()
        await until(
            lambda: len(self.plugin.factory.leases) > leases or self.call.done
        )
        await asyncio.sleep(0)

    async def send_first(self, message: Any) -> None:
        self.call.push(message)
        self.begin()
        await asyncio.sleep(0)

    async def send_audio(self, data: bytes) -> None:
        self.call.audio(data)
        await asyncio.sleep(0)

    async def finalize(self) -> None:
        self.call.half_close()
        await asyncio.sleep(0)

    async def outcome(self, timeout: float = 5.0) -> Outcome:
        await self.call.wait(timeout)
        return Outcome(
            "grpc",
            status=status_name(self.call.status),
            details=self.call.details,
            responses=list(self.call.responses),
        )


class RealtimeStream:
    """One claimed Riva Realtime connection (``RealtimeClient``)."""

    surface = "realtime"

    def __init__(self, app: Any, plugin: Any):
        self.plugin = plugin
        self.client = RealtimeClient(app, plugin.clock, url=REALTIME_URL)

    @property
    def ws(self) -> Any:
        return self.client.ws

    @property
    def events(self) -> list[dict[str, Any]]:
        return [e for e in self.client.events if e["type"] != "websocket.close"]

    @property
    def close(self) -> int | None:
        return self.client.close_code

    async def connect(self) -> bool:
        return await self.client.connect()

    async def receive(self, timeout: float = 5.0) -> dict[str, Any] | None:
        """The next event, or ``None`` once the app closed."""
        if self.close is not None:
            return None
        event = await self.client.next(timeout)
        if event["type"] == "websocket.close":
            return None
        return event

    def session(self, **overrides: Any) -> dict[str, Any]:
        """The default session object in the public shape."""
        return default_session(
            self.plugin.served_model,
            self.plugin.factory.default_locale,
            **overrides,
        )

    async def update(self, session: dict[str, Any]) -> dict[str, Any] | None:
        await self.client.update(session)
        return await self.receive()

    async def configure(self, **overrides: Any) -> None:
        """Connect, read ``conversation.created``, configure and open."""
        assert await self.connect()
        created = await self.receive()
        assert created is not None
        assert created["type"] == CREATED, created
        event = await self.update(self.session(**overrides))
        assert event is not None
        assert event["type"] == UPDATED, event

    async def send_first(self, payload: dict[str, Any]) -> None:
        assert await self.connect()
        created = await self.receive()
        assert created is not None
        await self.client.send(payload)

    async def send_audio(self, data: bytes) -> None:
        await self.client.append(data)
        await asyncio.sleep(0)

    async def finalize(self) -> None:
        await self.client.done_event()
        await asyncio.sleep(0)

    async def outcome(self, timeout: float = 5.0) -> Outcome:
        await self.client.outcome(timeout)
        return Outcome("realtime", events=self.events, close=self.close)


@pytest.fixture
def stream(test_app):
    """``stream(surface, plugin)`` builds a driver for that surface."""

    def build(surface: str, plugin: Any, **kwargs: Any):
        if surface == "grpc":
            return GrpcStream(plugin, **kwargs)
        return RealtimeStream(test_app, plugin)

    return build


@pytest.fixture
def helpers():
    """The helper functions of this module, for the test files."""
    return SimpleNamespace(
        until=until,
        assert_row=assert_row,
        event_code=error_code,
        event_fields=error_fields,
        event_authority=error_authority,
        status_name=status_name,
        recognition_config=recognition_config,
        config_request=config_request,
        audio_request=audio_request,
        recognize_request=recognize_request,
        recognize=recognize,
        speech_bytes=speech_bytes,
        Outcome=Outcome,
        FAILED_EVENT=FAILED_EVENT,
        COMPLETED_EVENT=COMPLETED_EVENT,
        REALTIME_URL=REALTIME_URL,
    )
