"""Coexistence with core's routes: nothing shadowed, collisions fail startup.

The collision check runs against core's real route table: every router
``register_api_routers`` attaches for a server with the speech tasks (and,
separately, with ``generate`` too), with the plugin attached through core's
``attach_endpoint_plugins``.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.routing import APIRoute
from support.app import route_table
from support.state import make_args

pytestmark = pytest.mark.engine

OPERATIONAL = (
    "/v1/health/ready",
    "/v1/health/live",
    "/v1/version",
    "/v1/metadata",
)
BOOTSTRAP = "/v1/realtime/transcription_sessions"
NEW_ROUTES = [(path, "GET") for path in OPERATIONAL] + [(BOOTSTRAP, "POST")]
CORE_AUTHORITATIVE = [
    ("/health", "GET"),
    ("/v1/models", "GET"),
    ("/version", "GET"),
    ("/metrics", "GET"),
]
FILE_ROUTE = ("/v1/audio/transcriptions", "POST")
TASK_SETS = [
    ("transcription", "realtime"),
    ("generate", "transcription", "realtime"),
]


def _core_app(tasks) -> FastAPI:
    from vllm.entrypoints.launchers.api_server.routers import (
        register_api_routers,
    )

    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    app.state.args = make_args()
    register_api_routers(app.state.args, app, tuple(tasks), None)
    return app


def _attach(app: FastAPI, plugins: list[Any], tasks) -> None:
    from vllm.plugins.endpoint_plugins.interface import (
        attach_endpoint_plugins,
    )

    with mock.patch(
        "vllm.plugins.load_endpoint_plugins", return_value=list(plugins)
    ):
        attach_endpoint_plugins(app, tuple(tasks))


def _serving(app: FastAPI, path: str, method: str) -> list[APIRoute]:
    return [
        route
        for route in app.routes
        if isinstance(route, APIRoute)
        and route.path == path
        and method in (route.methods or ())
    ]


# @spec ING-SHIM-015
@pytest.mark.parametrize("tasks", TASK_SETS)
def test_new_paths_collide_with_no_route_in_cores_table(tasks) -> None:
    app = _core_app(tasks)
    core_paths = {path for path, _, _ in route_table(app)}
    for path, _ in NEW_ROUTES:
        assert path not in core_paths, path


# @spec ING-SHIM-015, ING-SHIM-008, ING-SHIM-007
@pytest.mark.parametrize("tasks", TASK_SETS)
def test_plugin_attaches_on_cores_full_table(tasks, plugin) -> None:
    app = _core_app(tasks)
    before = {
        (path, method): endpoint
        for path, methods, endpoint in route_table(app)
        for method in methods
    }
    _attach(app, [plugin], tasks)
    after = {
        (path, method): endpoint
        for path, methods, endpoint in route_table(app)
        for method in methods
    }
    for key, endpoint in before.items():
        if key == FILE_ROUTE:
            continue  # the deliberate override (files leaf)
        assert after.get(key) is endpoint, key
    for path, method in NEW_ROUTES:
        routes = _serving(app, path, method)
        assert len(routes) == 1, (path, method)
        assert (path, method) not in before


# @spec ING-SHIM-008
@pytest.mark.parametrize("tasks", TASK_SETS)
def test_cores_operational_routes_stay_authoritative(tasks, plugin) -> None:
    app = _core_app(tasks)
    endpoints = {
        key: [r.endpoint for r in _serving(app, *key)]
        for key in CORE_AUTHORITATIVE
    }
    _attach(app, [plugin], tasks)
    for key, before in endpoints.items():
        assert before, key
        assert [r.endpoint for r in _serving(app, *key)] == before, key


# @spec ING-SHIM-008
def test_plugin_adds_no_health_or_metrics_route(plugin) -> None:
    tasks = TASK_SETS[0]
    app = _core_app(tasks)
    _attach(app, [plugin], tasks)
    for path in ("/v1/health", "/v1/metrics"):
        assert _serving(app, path, "GET") == [], path


def _app_with(path: str, method: str) -> FastAPI:
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    router = APIRouter()

    async def existing() -> dict:
        return {"owner": "core"}

    router.add_api_route(path, existing, methods=[method])
    app.include_router(router)
    return app


# @spec ING-SHIM-008
@pytest.mark.parametrize(("path", "method"), NEW_ROUTES)
def test_attach_router_fails_when_core_already_serves_a_new_path(
    path, method, plugin
) -> None:
    app = _app_with(path, method)
    with pytest.raises(Exception):  # noqa: B017 (any startup failure)
        plugin.attach_router(app)


class _EarlierPlugin:
    """An endpoint plugin attached before the frontend."""

    name = "earlier"
    required_tasks = None

    def __init__(self, path: str, method: str) -> None:
        self.path = path
        self.method = method

    def attach_router(self, app: FastAPI) -> None:
        router = APIRouter()

        async def mine() -> dict:
            return {"owner": "earlier"}

        router.add_api_route(self.path, mine, methods=[self.method])
        app.include_router(router)


# @spec ING-SHIM-008
@pytest.mark.parametrize(("path", "method"), NEW_ROUTES)
def test_attach_fails_when_an_earlier_plugin_serves_a_new_path(
    path, method, plugin
) -> None:
    tasks = TASK_SETS[0]
    app = _core_app(tasks)
    with pytest.raises(Exception):  # noqa: B017 (any startup failure)
        _attach(app, [_EarlierPlugin(path, method), plugin], tasks)


# @spec ING-SHIM-008
def test_other_methods_on_a_new_path_do_not_collide(plugin) -> None:
    app = _app_with("/v1/version", "POST")
    plugin.attach_router(app)
    assert len(_serving(app, "/v1/version", "GET")) == 1


def _metric_names() -> set[str]:
    from prometheus_client import REGISTRY

    return {metric.name for metric in REGISTRY.collect()}


# @spec ING-SHIM-008, ING-SHIM-014
async def test_plugin_registers_no_metric_families_of_its_own(
    test_app, fake_engine, fake_models, fake_transcription, plugin
) -> None:
    from support.app import init_state, start_plugins, stop_plugins

    before = _metric_names()
    state = await init_state(
        test_app,
        fake_engine,
        models=fake_models,
        transcription=fake_transcription,
    )
    await start_plugins(state)
    try:
        assert _metric_names() - before == set()
    finally:
        await stop_plugins(state)
