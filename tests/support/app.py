"""A FastAPI app built the way core's ``build_app`` builds it.

``build_app`` follows ``vllm/entrypoints/launchers/app.py`` (``build_app``,
lines 19-54): routers first, then endpoint-plugin hooks, then exception
handlers, then middleware (``init_entrypoints_middleware`` of
``vllm/entrypoints/serve/middleware/register.py``), so the plugin's
dispatcher middleware sits inside core's and ``--api-key`` applies to it. The
routers are core's real ones where the frontend interacts with them:

* ``GET /health`` (``vllm/entrypoints/serve/instrumentator/health.py``),
  which reads ``engine_client.check_health``, ``state.draining`` and
  ``endpoint_plugins_ready``;
* ``POST /v1/audio/transcriptions`` and ``POST /v1/audio/translations``
  (``vllm/entrypoints/speech_to_text/factories.py:21-33``), present only when
  the server has the ``transcription`` task;
* a recording stand-in for core's ``/v1/realtime`` WebSocket route, present
  with the ``realtime`` task, so a test sees which scopes the dispatcher
  passed through.

Plugins are attached through core's real ``attach_endpoint_plugins`` with
entry-point discovery replaced by the given list, so the route override and
``state.endpoint_plugins`` are core's. ``init_state`` fills the state as
core's ``init_app_state`` does and runs core's ``init_endpoint_plugins_state``.
The lifecycle helpers wrap core's ``start_endpoint_plugins``,
``endpoint_plugins_ready``, ``drain_endpoint_plugins`` and
``stop_endpoint_plugins`` (``vllm/plugins/endpoint_plugins/interface.py``).
"""

from __future__ import annotations

import asyncio
import time
from argparse import Namespace
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any
from unittest import mock

import httpx
from fastapi import APIRouter, FastAPI, WebSocket
from fastapi.routing import APIRoute
from starlette.datastructures import State
from vllm.entrypoints.serve.exception_handling.register import (
    init_exception_handler,
)
from vllm.entrypoints.serve.middleware.register import (
    init_entrypoints_middleware,
)
from vllm.plugins.endpoint_plugins.interface import (
    attach_endpoint_plugins,
    drain_endpoint_plugins,
    endpoint_plugins_ready,
    init_endpoint_plugins_state,
    start_endpoint_plugins,
    stop_endpoint_plugins,
)

from support.fake_engine import FakeEngineClient
from support.fake_transcription import (
    FakeServingModels,
    FakeTranscriptionServing,
)
from support.state import make_args, make_state

SUPPORTED_TASKS_KEY = "supported_tasks"

core_realtime_router = APIRouter()


@core_realtime_router.websocket("/v1/realtime")
async def core_realtime_stand_in(websocket: WebSocket) -> None:
    """Core's ``/v1/realtime`` as the dispatcher's pass-through target.

    Records the scope on ``app.state.core_realtime_scopes``, accepts, sends
    one ``{"type": "core.realtime"}`` event and closes normally.
    """
    scopes = getattr(websocket.app.state, "core_realtime_scopes", None)
    if scopes is None:
        scopes = websocket.app.state.core_realtime_scopes = []
    scopes.append(
        {
            "path": websocket.scope["path"],
            "query_string": websocket.scope["query_string"],
            "headers": list(websocket.scope["headers"]),
        }
    )
    await websocket.accept()
    await websocket.send_json({"type": "core.realtime"})
    await websocket.close(code=1000)


def build_app(
    plugins: Sequence[Any] = (),
    *,
    supported_tasks: Iterable[str] = ("transcription", "realtime"),
    args: Namespace | None = None,
    api_key: str | list[str] | None = None,
    enable_request_id_headers: bool = False,
    realtime_stand_in: bool = True,
) -> FastAPI:
    """Build the app; see the module docstring.

    ``api_key`` installs core's ``AuthenticationMiddleware`` for the given
    token(s). The result's ``extra["supported_tasks"]`` records the tasks.
    """
    tasks = tuple(supported_tasks)
    if args is None:
        args = make_args(
            api_key=(
                None
                if api_key is None
                else ([api_key] if isinstance(api_key, str) else list(api_key))
            ),
            enable_request_id_headers=enable_request_id_headers,
        )
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    app.state.args = args
    app.root_path = args.root_path or ""
    app.extra[SUPPORTED_TASKS_KEY] = tasks

    from vllm.entrypoints.serve.instrumentator.health import (
        router as health_router,
    )

    app.include_router(health_router)
    if "realtime" in tasks and realtime_stand_in:
        app.include_router(core_realtime_router)
    if "transcription" in tasks:
        from vllm.entrypoints.speech_to_text.transcription.api_router import (
            router as transcription_router,
        )
        from vllm.entrypoints.speech_to_text.translation.api_router import (
            router as translation_router,
        )

        app.include_router(transcription_router)
        app.include_router(translation_router)

    with mock.patch(
        "vllm.plugins.load_endpoint_plugins", return_value=list(plugins)
    ):
        attach_endpoint_plugins(app, tasks)

    init_exception_handler(app)
    init_entrypoints_middleware(args, app, tasks)
    return app


def supported_tasks(app: FastAPI) -> tuple[str, ...]:
    return tuple(app.extra[SUPPORTED_TASKS_KEY])


async def init_state(
    app: FastAPI,
    engine: FakeEngineClient,
    *,
    models: FakeServingModels | None = None,
    transcription: FakeTranscriptionServing | None = None,
) -> State:
    """Fill ``app.state`` as core does, then run core's Phase B.

    ``init_endpoint_plugins_state`` calls each plugin's ``init_state`` and
    sets ``state.draining = False``.
    """
    state = make_state(
        engine,
        tasks=supported_tasks(app),
        args=app.state.args,
        state=app.state,
        models=models,
        transcription=transcription,
    )
    await init_endpoint_plugins_state(engine, state, app.state.args)
    return state


async def start_plugins(state: State) -> None:
    """Core's lifespan step: ``start`` each attached plugin in order."""
    await start_endpoint_plugins(state, getattr(state, "endpoint_plugins", []))


def plugins_ready(state: State) -> bool:
    return endpoint_plugins_ready(state)


async def drain_plugins(state: State, timeout: float) -> None:
    """Core's shutdown step: set ``draining`` and drain the plugins.

    ``timeout`` is seconds from now (``drain_endpoint_plugins`` takes a
    monotonic deadline). Core caps the whole drain in real time, so a test
    under ``FakeClock`` uses ``drain_plugin`` instead.
    """
    await drain_endpoint_plugins(state, time.monotonic() + timeout)


async def drain_plugin(plugin: Any, state: State, timeout: float) -> None:
    """One plugin's ``drain`` as core calls it, without core's real-time cap.

    Sets ``state.draining = True`` first (``drain_endpoint_plugins`` does so
    before anything is awaited), then awaits ``plugin.drain(state, timeout)``
    with ``timeout`` in seconds. The plugin turns the seconds into deadlines
    on its own clock, so a ``FakeClock`` drives every wait.
    """
    state.draining = True
    await plugin.drain(state, timeout)


async def stop_plugins(state: State) -> None:
    await stop_endpoint_plugins(state)


@dataclass
class Served:
    """A plugin attached through core's wiring, with state and (maybe) started.

    ``stop()`` runs core's ``stop_endpoint_plugins`` once when it started.
    """

    plugin: Any
    app: FastAPI
    state: State
    engine: FakeEngineClient
    models: FakeServingModels
    transcription: FakeTranscriptionServing | None
    started: bool = False

    @property
    def servicer(self) -> Any:
        return self.plugin.servicer

    @property
    def factory(self) -> Any:
        return self.plugin.factory

    @property
    def clock(self) -> Any:
        return self.plugin.clock

    async def start(self) -> None:
        if not self.started:
            await start_plugins(self.state)
            self.started = True

    async def stop(self, timeout: float = 10.0) -> None:
        if self.started:
            self.started = False
            await asyncio.wait_for(stop_plugins(self.state), timeout)


async def serve_plugin(
    plugin: Any,
    engine: FakeEngineClient,
    *,
    tasks: Iterable[str] = ("transcription", "realtime"),
    models: FakeServingModels | None = None,
    transcription: FakeTranscriptionServing | None = None,
    args: Namespace | None = None,
    api_key: str | list[str] | None = None,
    extra_plugins: Iterable[Any] = (),
    start: bool = True,
) -> Served:
    """Build the app with ``plugin``, fill the state, run ``init_state``, start.

    ``transcription`` defaults to a ``FakeTranscriptionServing`` over
    ``models`` when the ``transcription`` task is present, else ``None``.
    ``init_state``'s exception propagates (nothing is started then).
    """
    tasks = tuple(tasks)
    if models is None:
        models = FakeServingModels(engine.model_config.served_model_name)
    if transcription is None and "transcription" in tasks:
        transcription = FakeTranscriptionServing(models)
    app = build_app(
        [plugin, *extra_plugins],
        supported_tasks=tasks,
        args=args,
        api_key=api_key,
    )
    state = await init_state(
        app, engine, models=models, transcription=transcription
    )
    served = Served(plugin, app, state, engine, models, transcription)
    if start:
        await served.start()
    return served


def http_client(
    app: FastAPI,
    *,
    api_key: str | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.AsyncClient:
    """An ``httpx`` client over the ASGI app, on the test's loop."""
    all_headers = dict(headers or {})
    if api_key is not None:
        all_headers["Authorization"] = f"Bearer {api_key}"
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
        headers=all_headers,
    )


def route_table(app: FastAPI) -> list[tuple[str, frozenset[str], Any]]:
    """``(path, methods, endpoint)`` for every ``APIRoute``, in order."""
    return [
        (route.path, frozenset(route.methods or ()), route.endpoint)
        for route in app.routes
        if isinstance(route, APIRoute)
    ]


def find_route(app: FastAPI, method: str, path: str) -> APIRoute | None:
    """The ``APIRoute`` serving ``method path``, or ``None``."""
    for route in app.routes:
        if (
            isinstance(route, APIRoute)
            and route.path == path
            and method.upper() in (route.methods or ())
        ):
            return route
    return None


def middleware_stack(app: FastAPI) -> list[type]:
    """The classes of ``app.user_middleware``, outermost first."""
    return [m.cls for m in app.user_middleware]
