"""Fixtures and helpers for the streaming leaf.

The streaming surfaces are exercised on a plugin that went through core's
real wiring (``support.app.serve_plugin``): ``attach_router`` in
``build_app``, ``init_state`` through ``init_endpoint_plugins_state`` and
``start`` through ``start_endpoint_plugins``, so admission is open exactly as
in service. The gRPC servicer is driven with ``support.riva.StreamingCall``
and ``support.riva.recognize``; the Realtime WebSocket with
``support.realtime.RealtimeClient``. Time is the plugin's ``FakeClock``.

Frontend modules are imported inside fixtures and helpers so that a missing
module fails the test instead of breaking collection.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Iterable
from typing import Any

import pytest
from support.app import Served, serve_plugin
from support.clock import FakeClock, until
from support.fake_engine import FakeEngineClient
from support.fake_session import FakeSessionFactory, fake_entry_points
from support.grpc_harness import free_port
from support.riva import StreamingCall, rasr

SERVED = "nvidia/nemotron-asr-streaming-test"
PRE_SUBMIT = 1600


# ------------------------------------------------------------- plugin


def riva_options(**overrides: Any) -> dict[str, Any]:
    """A ``riva_frontend`` mapping with a free port and a small pre-submit.

    Every other option keeps its default.
    """
    options: dict[str, Any] = {
        "grpc_bind": f"127.0.0.1:{free_port()}",
        "pre_submit_max_samples": PRE_SUBMIT,
    }
    options.update(overrides)
    return options


def small_unary_options(**overrides: Any) -> dict[str, Any]:
    """Small unary limits that still satisfy the startup relations."""
    options = {
        "unary_max_encoded_audio_bytes": 8192,
        "grpc_config_envelope_max_bytes": 1024,
        "grpc_receive_max_bytes": 8192 + 1024,
        "max_riff_header_bytes": 2048,
    }
    options.update(overrides)
    return riva_options(**options)


@contextlib.asynccontextmanager
async def serving(
    *,
    clock: FakeClock | None = None,
    factory: FakeSessionFactory | None = None,
    options: dict[str, Any] | None = None,
    tasks: Iterable[str] = ("transcription", "realtime"),
    start: bool = True,
) -> AsyncIterator[Served]:
    """The plugin attached, initialized and (by default) started."""
    from vllm_riva_frontend.lifecycle import RivaFrontendPlugin

    tasks = tuple(tasks)
    clock = clock or FakeClock()
    factory = factory or FakeSessionFactory(
        final_text="hello world", clock=clock
    )
    engine = FakeEngineClient(
        tasks=tasks,
        riva_frontend=options if options is not None else riva_options(),
    )
    plugin = RivaFrontendPlugin(
        clock=clock,
        entry_points=fake_entry_points(nemotron=factory.constructor),
    )
    served = await serve_plugin(plugin, engine, tasks=tasks, start=start)
    try:
        yield served
    finally:
        await served.stop()


@pytest.fixture
async def served(clock):
    """The default serving plugin (16 slots, files enabled)."""
    async with serving(clock=clock) as result:
        yield result


@pytest.fixture
async def served_through_drain(clock):
    """``served`` whose idle streams outlast a 60 s drain.

    A stream idle since the drain began must still be open at the settle
    point (49 s) and at ``D`` (54 s): ``session_idle_timeout`` 59 s outlasts
    both and stays below the engine's 60 s parked-idle deadline (ING-EP-014).
    """
    options = riva_options(session_idle_timeout=59)
    async with serving(clock=clock, options=options) as result:
        yield result


def opens(served: Served) -> list[dict[str, Any]]:
    """The ``open`` calls the factory saw, as their keyword arguments."""
    return [
        call.kwargs
        for call in served.factory.calls
        if call.target == "factory" and call.name == "open"
    ]


def served_lease(served: Served, index: int = 0) -> Any:
    """The ``index``-th lease the factory opened."""
    return served.factory.leases[index]


def lease_names(served: Served, index: int = 0) -> list[str]:
    return served.factory.names(f"lease:{index}")


# ----------------------------------------------------------- gRPC: wire


def start_stream(
    served: Served,
    messages: Iterable[Any] = (),
    *,
    time_remaining: float | None = None,
) -> StreamingCall:
    return StreamingCall(
        served.servicer,
        served.clock,
        messages=messages,
        time_remaining=time_remaining,
    )


async def opened(served: Served, call: StreamingCall, index: int = 0) -> Any:
    """Wait until the stream's lease exists and return it."""
    await until(
        lambda: len(served.factory.leases) > index or call.done,
        what="open",
    )
    assert len(served.factory.leases) > index, (
        f"no lease opened: {call.status} {call.details!r}"
    )
    return served.factory.leases[index]


async def first_message_outcome(
    served: Served, request: rasr.StreamingRecognizeRequest
) -> StreamingCall:
    """Send one first message, then half-close once opened or rejected."""
    call = start_stream(served, [request])
    await until(lambda: call.done or bool(opens(served)), what="open or reject")
    if not call.done:
        await until(lambda: bool(served.factory.leases), what="lease")
    call.half_close()
    await call.wait()
    return call


async def saturate_gate(served: Served) -> list[Any]:
    """Take every file-gate permit; the caller releases them."""
    gate = served.plugin.file_gate
    deadline = served.clock.monotonic() + 1_000
    return [
        await gate.acquire(deadline)
        for _ in range(served.plugin.config.max_concurrent_files)
    ]
