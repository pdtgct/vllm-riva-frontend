"""Fixtures for the session leaf: the seam, the stream owner and its bounds.

The plugin under test is ``lifecycle.RivaFrontendPlugin`` built with the
shared ``FakeClock`` and a ``fake_entry_points`` callable that publishes the
shared ``FakeSessionFactory``. Surface-level tests drive the owner through the
Riva servicer (``support.riva.StreamingCall``) or the Realtime adapter
(``support.realtime.RealtimeClient``) and observe the fake lease's record;
owner-level tests build a ``lease.StreamOwner`` directly (the shared
``make_owner`` fixture).

Frontend modules are imported inside fixtures and helpers, never at module
level, so collection succeeds while the implementation is missing and every
test then fails on the missing module instead of being skipped.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from typing import Any

import pytest
from support.asgi import AsgiWebSocket
from support.fake_session import FakeSessionFactory
from support.realtime import REALTIME_URL, RealtimeClient, default_session
from support.riva import SETTLE, StreamingCall

__all__ = ["REALTIME_URL", "SETTLE"]


# ------------------------------------------------------------------ helpers


def lease_calls(factory: FakeSessionFactory) -> list[str]:
    """Every lease call of every lease, in global order."""
    return [c.name for c in factory.calls if c.target.startswith("lease:")]


def startup_failure(exc: BaseException) -> BaseException | None:
    """The ``StartupError`` in an exception's cause chain, if any."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "failures"):
            return current
        current = current.__cause__ or current.__context__
    return None


def failure_text(exc: BaseException) -> str:
    """Every message in the chain, with a ``StartupError``'s failures."""
    parts = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(str(current))
        parts.extend(str(f) for f in getattr(current, "failures", ()) or ())
        current = current.__cause__ or current.__context__
    return "\n".join(parts)


# ------------------------------------------------------------- fixtures


@pytest.fixture
async def fake_factory(clock):
    """The published factory, on the plugin's clock.

    Its ``flush`` returns ``"final words"``; what is still running at the
    end of the test is cancelled.
    """
    factory = FakeSessionFactory(final_text="final words", clock=clock)
    try:
        yield factory
    finally:
        await factory.aclose()


# ``riva_options``, ``fake_engine``, ``entry_points``, ``plugin`` and
# ``started`` are the shared fixtures of ``tests/conftest.py``.


@pytest.fixture
def plugins(plugin):
    return [plugin]


@pytest.fixture
def riva(started, clock) -> Callable[..., StreamingCall]:
    """Start a ``StreamingRecognize`` call: ``riva(**context_kwargs)``."""

    def make(**context: Any) -> StreamingCall:
        return StreamingCall(started.servicer, clock, **context)

    return make


@pytest.fixture
def realtime(started, test_app, clock) -> Callable[..., RealtimeClient]:
    """A Realtime session over the test app: ``realtime(ws=None)``."""

    def make(ws: AsgiWebSocket | None = None) -> RealtimeClient:
        return RealtimeClient(test_app, clock, ws=ws)

    return make


@pytest.fixture
def first_update(fake_engine) -> Callable[..., dict[str, Any]]:
    """The public default session as a first update, with ``language`` set."""

    def make(language: str = "en-US") -> dict[str, Any]:
        return default_session(
            fake_engine.model_config.served_model_name, language
        )

    return make


@pytest.fixture
async def drain_trigger(started, app_state, clock):
    """Start ``drain`` with an 8 s budget and reach the settle point.

    With ``timeout = 8`` the reserve is 5 s, so ``D = now + 3`` and the
    settle point is ``D - 0.75``. Returns an async callable that starts the
    drain task, advances virtual time to just past the settle point and
    returns ``(task, D)``; ``past_deadline()`` then moves past ``D``.
    """
    tasks: list[asyncio.Task[Any]] = []

    async def trigger() -> tuple[asyncio.Task[Any], float]:
        from vllm_riva_frontend.lifecycle import engine_reserve, settle_point

        timeout = 8.0
        now = clock.monotonic()
        deadline = now + timeout - engine_reserve(timeout)
        app_state.draining = True
        task = asyncio.create_task(started.drain(app_state, timeout))
        tasks.append(task)
        await clock.settle(SETTLE)
        point = settle_point(now, deadline)
        await clock.advance(point - clock.monotonic() + 0.01, settle=SETTLE)
        return task, deadline

    yield trigger
    for task in tasks:
        if not task.done():
            task.cancel()
        with contextlib.suppress(BaseException):
            await task


# ``owner_parts`` and ``make_owner`` are shared fixtures of
# ``tests/conftest.py``.


@pytest.fixture
def cause() -> Callable[..., Any]:
    """Build a ``lease.Cause`` by keyword."""

    def make(
        code: str, phase: str = "transport", deadline: float | None = None
    ):
        from vllm_riva_frontend.lease import Cause

        return Cause(code=code, phase=phase, deadline=deadline)

    return make
