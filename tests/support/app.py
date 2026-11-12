"""A FastAPI app built the way core's ``build_app`` builds it.

``build_app`` follows ``vllm/entrypoints/launchers/app.py`` (``build_app``,
lines 19-54): routers first, then endpoint-plugin hooks, then exception
handlers, then middleware (``init_entrypoints_middleware`` of
``vllm/entrypoints/serve/middleware/register.py``), so the plugin's
dispatcher middleware sits inside core's and ``--api-key`` applies to it. The
routers are core's real ones where the frontend interacts with them:

* ``GET /version``, ``GET /load`` and ``GET /kv_event_sources``
  (``vllm/entrypoints/serve/instrumentator/basic.py``): core's own
  ``/version`` stays beside the frontend's version route;
* ``GET /health`` and ``GET /ready``
  (``vllm/entrypoints/serve/instrumentator/health.py``): ``/health`` reads
  ``engine_client.check_health`` only; ``/ready`` also reads
  ``state.draining`` and ``endpoint_plugins_ready``;
* ``POST /v1/audio/transcriptions`` and ``POST /v1/audio/translations``
  (``vllm/entrypoints/speech_to_text/factories.py:21-33``), present only when
  the server has the ``transcription`` task;
* a recording stand-in for core's ``/v1/realtime`` WebSocket route, present
  with the ``realtime`` task, so a test sees that core's route still serves
  its own scopes, unchanged, beside the frontend's routes.

The frontend adds its routes under ``PLUGIN_PREFIX`` (``/plugins/riva/v1``)
and its path-scoped authentication middleware with ``app.add_middleware``
in ``attach_router``; since core installs its own middleware after the
plugin hooks, core's WebSocket metrics, authentication and CORS middleware
wrap the frontend's. ``api_key`` installs core's ``AuthenticationMiddleware``
the way ``--api-key`` does; a test that needs ``VLLM_API_KEY`` instead sets
the environment variable (``monkeypatch.setenv``) before ``build_app`` and
keeps it set through ``init_state``, since core reads it in both places.
``auth_registrations`` adds further registrations of core's
``AuthenticationMiddleware`` (each a keyword mapping, possibly without
``tokens``) after core's own, for the authentication-decision tests
(ING-EP-086).

``core_app`` is core's real route table for a task set (every router
``register_api_routers`` attaches) without any plugin; ``attach_plugins``
attaches plugins to it through core's ``attach_endpoint_plugins``;
``all_routes`` lists every route of any kind; ``RouteAddingPlugin`` is an
endpoint plugin that adds given routes (an earlier or later plugin for the
collision and route-table checks, ING-EP-091 and ING-EP-070).

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

#: The frontend's namespace (ING-SHIM-023) and its route set.
PLUGIN_BASE = "/plugins/riva"
PLUGIN_PREFIX = f"{PLUGIN_BASE}/v1"
OPERATIONAL_ROUTES = (
    ("GET", f"{PLUGIN_PREFIX}/health/ready"),
    ("GET", f"{PLUGIN_PREFIX}/health/live"),
    ("GET", f"{PLUGIN_PREFIX}/version"),
    ("GET", f"{PLUGIN_PREFIX}/metadata"),
)
BOOTSTRAP_ROUTE = ("POST", f"{PLUGIN_PREFIX}/realtime/transcription_sessions")
REALTIME_ROUTE = ("WEBSOCKET", f"{PLUGIN_PREFIX}/realtime")
FILE_ROUTE = ("POST", f"{PLUGIN_PREFIX}/audio/transcriptions")
#: Core's speech routes the frontend leaves untouched.
CORE_TRANSCRIPTION_ROUTE = ("POST", "/v1/audio/transcriptions")
CORE_TRANSLATION_ROUTE = ("POST", "/v1/audio/translations")
CORE_REALTIME_ROUTE = ("WEBSOCKET", "/v1/realtime")


def frontend_routes(*, file_route: bool = True) -> set[tuple[str, str]]:
    """The frontend's route set (ING-SHIM-023) as ``(method, path)``.

    ``WEBSOCKET`` stands for a WebSocket route. ``file_route`` is whether
    core's transcription route is registered (ING-FILE-075).
    """
    routes = {*OPERATIONAL_ROUTES, BOOTSTRAP_ROUTE, REALTIME_ROUTE}
    if file_route:
        routes.add(FILE_ROUTE)
    return routes


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
    root_path: str | None = None,
    auth_registrations: Iterable[dict[str, Any]] = (),
) -> FastAPI:
    """Build the app; see the module docstring.

    ``api_key`` installs core's ``AuthenticationMiddleware`` for the given
    token(s); ``root_path`` sets ``--root-path``; ``auth_registrations``
    adds registrations of core's ``AuthenticationMiddleware`` with those
    keywords. The result's ``extra["supported_tasks"]`` records the tasks.
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
            root_path=root_path,
        )
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    app.state.args = args
    app.root_path = args.root_path or ""
    app.extra[SUPPORTED_TASKS_KEY] = tasks

    from vllm.entrypoints.serve.instrumentator.basic import (
        router as basic_router,
    )
    from vllm.entrypoints.serve.instrumentator.health import (
        router as health_router,
    )

    app.include_router(basic_router)
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
    if auth_registrations:
        from vllm.entrypoints.serve.middleware.authenticate import (
            AuthenticationMiddleware,
        )

        for keywords in auth_registrations:
            app.add_middleware(AuthenticationMiddleware, **keywords)
    return app


def auth_registrations(app: FastAPI) -> list[dict[str, Any]]:
    """The keywords of every registration of core's authentication.

    Outermost first, as ``app.user_middleware`` keeps them.
    """
    from vllm.entrypoints.serve.middleware.authenticate import (
        AuthenticationMiddleware,
    )

    return [
        dict(m.kwargs)
        for m in app.user_middleware
        if m.cls is AuthenticationMiddleware
    ]


def core_app(
    supported_tasks: Iterable[str] = ("transcription", "realtime"),
    *,
    args: Namespace | None = None,
) -> FastAPI:
    """Core's full route table for ``supported_tasks``, with no plugin."""
    from vllm.entrypoints.launchers.api_server.routers import (
        register_api_routers,
    )

    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    app.state.args = args if args is not None else make_args()
    app.extra[SUPPORTED_TASKS_KEY] = tuple(supported_tasks)
    register_api_routers(app.state.args, app, tuple(supported_tasks), None)
    return app


def attach_plugins(
    app: FastAPI,
    plugins: Sequence[Any],
    supported_tasks: Iterable[str] | None = None,
) -> None:
    """Core's ``attach_endpoint_plugins`` with discovery replaced."""
    tasks = (
        tuple(supported_tasks)
        if supported_tasks is not None
        else tuple(app.extra.get(SUPPORTED_TASKS_KEY, ()))
    )
    with mock.patch(
        "vllm.plugins.load_endpoint_plugins", return_value=list(plugins)
    ):
        attach_endpoint_plugins(app, tasks)


def all_routes(app: FastAPI) -> list[tuple[str, str, frozenset[str], Any]]:
    """``(kind, path, methods, endpoint-or-app)`` for every route, in order.

    ``kind`` is ``http`` (an ``APIRoute`` or Starlette ``Route``),
    ``websocket`` or ``mount``; methods are empty for the last two.
    """
    from starlette.routing import Mount, Route, WebSocketRoute

    rows: list[tuple[str, str, frozenset[str], Any]] = []
    for route in app.routes:
        if isinstance(route, APIRoute | Route):
            rows.append(
                (
                    "http",
                    route.path,
                    frozenset(route.methods or ()),
                    route.endpoint,
                )
            )
        elif isinstance(route, WebSocketRoute):
            rows.append(("websocket", route.path, frozenset(), route.endpoint))
        elif isinstance(route, Mount):
            rows.append(("mount", route.path, frozenset(), route.app))
        else:
            kind = type(route).__name__
            path = getattr(route, "path", "")
            rows.append((kind, path, frozenset(), route))
    return rows


class RouteAddingPlugin:
    """An endpoint plugin that adds the given routes in ``attach_router``.

    ``http`` is ``(method, path)`` pairs, ``websocket`` and ``mounts`` are
    paths. Its handlers answer ``{"plugin": name}`` (HTTP) or accept and
    close with 1000 (WebSocket); a mount serves a bare ``Starlette`` app.
    """

    required_tasks = None

    def __init__(
        self,
        name: str = "other",
        *,
        http: Iterable[tuple[str, str]] = (),
        websocket: Iterable[str] = (),
        mounts: Iterable[str] = (),
    ) -> None:
        self.name = name
        self.http = list(http)
        self.websocket = list(websocket)
        self.mounts = list(mounts)

    def attach_router(self, app: FastAPI) -> None:
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse

        name = self.name

        async def answer() -> JSONResponse:
            return JSONResponse({"plugin": name})

        async def accept(websocket: WebSocket) -> None:
            await websocket.accept()
            await websocket.close(code=1000)

        for method, path in self.http:
            app.add_api_route(path, answer, methods=[method])
        for path in self.websocket:
            app.add_api_websocket_route(path, accept)
        for path in self.mounts:
            app.mount(path, Starlette())

    async def init_state(self, engine_client, state, args) -> None:
        return None


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

    ``init_endpoint_plugins_state`` sets ``state.draining = False`` and then
    calls each plugin's ``init_state``.
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
    await start_endpoint_plugins(state)


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
    root_path: str = "",
) -> httpx.AsyncClient:
    """An ``httpx`` client over the ASGI app, on the test's loop.

    With ``root_path``, request paths must carry that prefix, as behind a
    proxy that forwards the full path.
    """
    all_headers = dict(headers or {})
    if api_key is not None:
        all_headers["Authorization"] = f"Bearer {api_key}"
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, root_path=root_path),
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
