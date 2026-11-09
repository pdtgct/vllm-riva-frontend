"""Fixtures and helpers for the vehicle leaf's tests.

The plugin under test is ``lifecycle.RivaFrontendPlugin`` built with the
shared ``FakeClock`` and a ``fake_entry_points`` callable that serves the
shared ``FakeSessionFactory``. It is attached through ``support.app`` (core's
real ``attach_endpoint_plugins``), initialized through core's
``init_endpoint_plugins_state`` and started with ``start``. ``start`` binds a
real ``grpc.aio`` server on a free loopback port, because the design refuses
port 0.

Frontend modules are imported inside fixtures and helpers only, so that
collection succeeds while the modules do not exist yet.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import sys
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

import pytest

# ------------------------------------------------------------ constants

SERVED = "nvidia/nemotron-asr-streaming-test"
SMALL_CAP = {
    "load_shed_max_sessions": 3,
    "max_concurrent_files": 1,
    "max_file_owners": 2,
}


# -------------------------------------------------------------- helpers


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    """Whether a TCP connect to ``host:port`` succeeds right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        try:
            sock.connect((host, port))
        except OSError:
            return False
        return True


def make_frontend(
    clock: Any,
    *,
    factories: dict[str, Any] | None = None,
    entry_points: Callable[..., Any] | None = None,
) -> Any:
    """A ``RivaFrontendPlugin`` whose factory discovery serves ``factories``.

    ``factories`` maps an entry-point name to a constructor
    ``(engine_client, model_config) -> factory | None``.
    """
    from support.fake_session import fake_entry_points

    from vllm_riva_frontend.lifecycle import RivaFrontendPlugin

    if entry_points is None:
        entry_points = fake_entry_points(**(factories or {}))
    return RivaFrontendPlugin(clock=clock, entry_points=entry_points)


def counting(constructor: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an entry-point constructor; ``wrapper.calls`` counts calls."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        wrapper.calls += 1  # type: ignore[attr-defined]
        return constructor(*args, **kwargs)

    wrapper.calls = 0  # type: ignore[attr-defined]
    return wrapper


def build(
    plugin: Any,
    *,
    tasks: Iterable[str] = ("transcription", "realtime"),
    extra_plugins: Iterable[Any] = (),
    api_key: str | None = None,
    args: Any = None,
) -> Any:
    """The core-like app with ``plugin`` (then ``extra_plugins``) attached."""
    from support.app import build_app

    return build_app(
        [plugin, *extra_plugins],
        supported_tasks=tuple(tasks),
        api_key=api_key,
        args=args,
    )


def prepare_state(
    app: Any,
    engine: Any,
    *,
    models: Any = None,
    transcription: Any = None,
) -> Any:
    """``app.state`` filled as core's ``init_app_state`` fills it."""
    from support.app import supported_tasks
    from support.state import make_state

    return make_state(
        engine,
        tasks=supported_tasks(app),
        args=app.state.args,
        state=app.state,
        models=models,
        transcription=transcription,
    )


async def run_init(app: Any, engine: Any, state: Any) -> Any:
    """Core's Phase B: ``init_endpoint_plugins_state``."""
    from vllm.plugins.endpoint_plugins.interface import (
        init_endpoint_plugins_state,
    )

    await init_endpoint_plugins_state(engine, state, app.state.args)
    return state


async def initialize(
    plugin: Any,
    engine: Any,
    *,
    tasks: Iterable[str] = ("transcription", "realtime"),
    models: Any = None,
    transcription: Any = None,
    extra_plugins: Iterable[Any] = (),
    api_key: str | None = None,
) -> tuple[Any, Any]:
    """Build the app, fill the state and run ``init_state`` (not ``start``).

    Returns ``(app, state)``; ``init_state``'s exception propagates.
    """
    from support.app import serve_plugin

    served = await serve_plugin(
        plugin,
        engine,
        tasks=tuple(tasks),
        models=models,
        transcription=transcription,
        extra_plugins=extra_plugins,
        api_key=api_key,
        start=False,
    )
    return served.app, served.state


def engine_with(
    port: int | None = None, *, options: dict[str, Any] | None = None, **kw: Any
) -> Any:
    """A ``FakeEngineClient`` whose ``riva_frontend`` options bind ``port``."""
    from support.fake_engine import FakeEngineClient
    from support.grpc_harness import free_port

    merged: dict[str, Any] = {
        "grpc_bind": f"127.0.0.1:{port if port is not None else free_port()}"
    }
    merged.update(options or {})
    return FakeEngineClient(riva_frontend=merged, **kw)


async def startup_failures(
    clock: Any,
    factory: Any,
    *,
    options: dict[str, Any] | None = None,
    **engine_kw: Any,
) -> list[str]:
    """Run ``init_state`` expecting ``StartupError``; return its failures."""
    from vllm_riva_frontend.lifecycle import StartupError

    plugin = make_frontend(clock, factories={"nemotron": factory.constructor})
    engine = engine_with(options=options, **engine_kw)
    with pytest.raises(StartupError) as info:
        await initialize(plugin, engine)
    failures = list(info.value.failures)
    assert failures, "StartupError must list its failures"
    return failures


def mentions(failures: Iterable[str], *needles: str) -> bool:
    """Whether some failure text mentions every needle."""
    return any(all(n in text for n in needles) for text in failures)


async def drive(
    clock: Any,
    awaitable: Awaitable[Any],
    *,
    step: float = 0.25,
    limit: int = 400,
) -> Any:
    """Await ``awaitable`` while advancing ``clock`` in ``step`` increments.

    A real-time guard (5 s) stops a hung awaitable from hanging the run.
    """
    task = asyncio.ensure_future(awaitable)
    for _ in range(limit):
        if task.done():
            break
        await clock.advance(step)
        await asyncio.sleep(0.001)
    return await asyncio.wait_for(task, 5.0)


async def until(predicate: Callable[[], bool], *, timeout: float = 3.0) -> None:
    """``support.clock.until`` with a longer bound, for real gRPC I/O."""
    from support.clock import until as poll

    await poll(predicate, timeout=timeout)


async def settle_until(
    clock: Any, predicate: Callable[[], bool], *, rounds: int = 200
) -> None:
    """Yield to the loop (no time advance) until ``predicate`` holds."""
    for _ in range(rounds):
        if predicate():
            return
        await clock.settle(5)
        await asyncio.sleep(0.001)
    raise AssertionError("condition not reached")


# ------------------------------------------------------------- Riva wire


def audio_message(seconds: float = 0.1) -> Any:
    """One ``audio_content`` message of PCM16 sine at 16 kHz."""
    from support.audio import pcm16_bytes, pcm16_sine
    from support.riva import audio_request

    return audio_request(pcm16_bytes(pcm16_sine(seconds)))


def stream(plugin: Any, clock: Any, *messages: Any) -> Any:
    """A started ``StreamingCall`` on the servicer (``support.riva``)."""
    from support.riva import StreamingCall

    return StreamingCall(plugin.servicer, clock, messages=messages)


async def stream_with_lease(
    plugin: Any, factory: Any, clock: Any, *, locale: str = "en-US"
) -> tuple[Any, Any]:
    """Open one stream and feed one audio message; return it and its lease."""
    from support.riva import config_request

    before = len(factory.leases)
    call = stream(
        plugin, clock, config_request(language_code=locale), audio_message()
    )
    await settle_until(
        clock,
        lambda: (
            len(factory.leases) > before
            and "feed" in factory.leases[before].names()
        ),
    )
    return call, factory.leases[before]


def file_request_body() -> tuple[bytes, list[tuple[str, str]]]:
    """A minimal valid file-route body and its headers."""
    from support.audio import pcm16_sine, wav_pcm16
    from support.http import content_type, multipart_body

    body = multipart_body(
        [("model", SERVED), ("language", "en")], file=wav_pcm16(pcm16_sine(0.5))
    )
    return body, [
        ("content-type", content_type()),
        ("content-length", str(len(body))),
    ]


def realtime_update(**transcription: Any) -> dict[str, Any]:
    """A first ``transcription_session.update`` in the public default shape."""
    from support.realtime import default_session, update_event

    session = default_session(SERVED, transcription.pop("language", "en-US"))
    session["input_audio_transcription"].update(transcription)
    return update_event(session)


def frontend_tasks(plugin: Any) -> set[asyncio.Task[Any]]:
    """Every task the plugin's registry knows about."""
    tasks: set[asyncio.Task[Any]] = set()
    for record in list(plugin.registry.records.values()):
        for task in (record.task, record.continuation, *record.children):
            if task is not None:
                tasks.add(task)
    return tasks


# -------------------------------------------------------------- fixtures


@pytest.fixture
def helpers() -> Any:
    """This module, for tests that use its helpers."""
    return sys.modules[__name__]


@pytest.fixture
def grpc_port() -> int:
    from support.grpc_harness import free_port

    return free_port()


@pytest.fixture
def frontend_options(grpc_port: int) -> dict[str, Any]:
    """The ``riva_frontend`` options of the default engine."""
    return {"grpc_bind": f"127.0.0.1:{grpc_port}"}


@pytest.fixture
def fake_engine(frontend_options: dict[str, Any]) -> Any:
    """The shared engine, with ``grpc_bind`` on a free loopback port."""
    from support.fake_engine import FakeEngineClient

    return FakeEngineClient(riva_frontend=dict(frontend_options))


@pytest.fixture
def frontend(clock: Any, fake_factory: Any) -> Any:
    """The plugin under test, discovering ``fake_factory`` as ``nemotron``."""
    return make_frontend(
        clock, factories={"nemotron": fake_factory.constructor}
    )


@pytest.fixture
def plugins(frontend: Any) -> list[Any]:
    return [frontend]


@pytest.fixture
async def initialized(frontend: Any, app_state: Any) -> Any:
    """``(plugin, state)`` after ``init_state``."""
    yield frontend, app_state
    if not getattr(frontend, "torn_down", True):
        with contextlib.suppress(Exception):
            await asyncio.wait_for(frontend.stop(app_state), 5.0)


@pytest.fixture
async def started(frontend: Any, app_state: Any, clock: Any) -> Any:
    """``(plugin, state)`` after ``init_state`` and ``start``.

    Teardown runs ``stop`` while advancing the fake clock.
    """
    await frontend.start(app_state)
    yield frontend, app_state
    if not frontend.torn_down:
        with contextlib.suppress(Exception):
            await drive(clock, frontend.stop(app_state))


@pytest.fixture
async def small_cap(clock: Any, fake_factory: Any, grpc_port: int) -> Any:
    """A started plugin with ``SMALL_CAP``: 3 owners, 2 of them files."""
    from support.fake_transcription import (
        FakeServingModels,
        FakeTranscriptionServing,
    )

    plugin = make_frontend(
        clock, factories={"nemotron": fake_factory.constructor}
    )
    engine = engine_with(grpc_port, options=dict(SMALL_CAP))
    models = FakeServingModels(SERVED)
    transcription = FakeTranscriptionServing(models)
    app, state = await initialize(
        plugin, engine, models=models, transcription=transcription
    )
    await plugin.start(state)
    yield plugin, app, state, transcription
    if not plugin.torn_down:
        with contextlib.suppress(Exception):
            await drive(clock, plugin.stop(state))
