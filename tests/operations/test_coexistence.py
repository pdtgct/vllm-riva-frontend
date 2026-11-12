"""Coexistence with core's routes: nothing shadowed, collisions fail startup.

The checks run against core's real route table (``support.app.core_app``:
every router ``register_api_routers`` attaches for a server with the speech
tasks, with and without ``generate``), with the plugin attached through
core's ``attach_endpoint_plugins`` (``support.app.attach_plugins``). The
frontend's routes all live under ``/plugins/riva/v1/``; core's routes keep
their path, methods, endpoint and order.
"""

from __future__ import annotations

import pytest
from support.app import (
    BOOTSTRAP_ROUTE,
    OPERATIONAL_ROUTES,
    PLUGIN_PREFIX,
    REALTIME_ROUTE,
    RouteAddingPlugin,
    all_routes,
    attach_plugins,
    build_app,
    core_app,
    frontend_routes,
    init_state,
    middleware_stack,
)

TASK_SETS = [
    ("transcription", "realtime"),
    ("realtime",),
    ("generate", "transcription", "realtime"),
]
#: Core's routes the plugin must leave authoritative (ING-SHIM-020).
CORE_AUTHORITATIVE = [
    ("http", "/health", "GET"),
    ("http", "/ready", "GET"),
    ("http", "/v1/models", "GET"),
    ("http", "/version", "GET"),
    ("http", "/metrics", "GET"),
    ("mount", "/metrics", None),
    ("websocket", "/v1/realtime", None),
    ("http", "/v1/audio/transcriptions", "POST"),
    ("http", "/v1/audio/translations", "POST"),
]
HTTP_FRONTEND = [*OPERATIONAL_ROUTES, BOOTSTRAP_ROUTE]
REALTIME_PATH = REALTIME_ROUTE[1]


def _pairs(rows) -> set[tuple[str, str]]:
    """``(method, path)`` rows; ``WEBSOCKET``/``MOUNT`` for the others."""
    pairs: set[tuple[str, str]] = set()
    for kind, path, methods, _ in rows:
        if kind == "http":
            pairs |= {(m, path) for m in methods if m != "HEAD"}
        else:
            pairs.add((kind.upper(), path))
    return pairs


def _serving(app, kind: str, path: str, method: str | None):
    """The endpoints (or mounted apps) of every matching route, in order."""
    return [
        endpoint
        for row_kind, row_path, methods, endpoint in all_routes(app)
        if row_kind == kind
        and row_path == path
        and (method is None or method in methods)
    ]


def _collides(route: tuple[str, str], rows) -> bool:
    method, path = route
    for kind, row_path, methods, _ in rows:
        if kind == "http" and method != "WEBSOCKET":
            if row_path == path and method in methods:
                return True
        elif kind == "websocket" and method == "WEBSOCKET":
            if row_path == path:
                return True
        elif kind == "mount" and (
            path == row_path or path.startswith(row_path.rstrip("/") + "/")
        ):
            return True
    return False


# ------------------------------------------------------------ attach_router


# @spec ING-EP-063
@pytest.mark.parametrize("tasks", TASK_SETS)
def test_attach_router_adds_namespace_routes_and_auth_middleware_only(
    tasks, plugin
) -> None:
    app = core_app(tasks)
    rows_before = all_routes(app)
    middleware_before = list(app.user_middleware)

    plugin.attach_router(app)

    rows_after = all_routes(app)
    added = [row for row in rows_after if row not in rows_before]
    assert _pairs(added) == frontend_routes(file_route="transcription" in tasks)
    for _, path, _, _ in added:
        assert path.startswith(f"{PLUGIN_PREFIX}/"), path
    # No core route is replaced, shadowed or wrapped.
    assert [row for row in rows_after if row in rows_before] == rows_before
    # Exactly one middleware, the frontend's own, added with add_middleware.
    new = [m for m in app.user_middleware if m not in middleware_before]
    assert len(app.user_middleware) == len(middleware_before) + 1
    assert len(new) == 1
    assert new[0].cls.__module__.startswith("vllm_riva_frontend"), new[0].cls


# @spec ING-EP-063
def test_frontend_middleware_runs_inside_cores_middleware(plugin) -> None:
    from starlette.middleware.cors import CORSMiddleware
    from vllm.entrypoints.serve.middleware.authenticate import (
        AuthenticationMiddleware,
    )
    from vllm.entrypoints.speech_to_text.realtime.metrics import (
        WebSocketMetricsMiddleware,
    )

    app = build_app([plugin], api_key="token")
    stack = middleware_stack(app)  # outermost first
    ours = [
        index
        for index, cls in enumerate(stack)
        if cls.__module__.startswith("vllm_riva_frontend")
    ]
    assert len(ours) == 1, stack
    for core in (
        WebSocketMetricsMiddleware,
        AuthenticationMiddleware,
        CORSMiddleware,
    ):
        assert stack.index(core) < ours[0], (core, stack)


# @spec ING-EP-063
@pytest.mark.parametrize("tasks", TASK_SETS)
def test_file_route_is_added_only_beside_cores_transcription_route(
    tasks, plugin
) -> None:
    app = core_app(tasks)
    core_has_it = bool(
        _serving(app, "http", "/v1/audio/transcriptions", "POST")
    )
    plugin.attach_router(app)
    file_route = ("POST", f"{PLUGIN_PREFIX}/audio/transcriptions")
    assert (file_route in _pairs(all_routes(app))) is core_has_it


# ---------------------------------------------------------- route checks


# @spec ING-EP-092
@pytest.mark.parametrize("tasks", TASK_SETS)
def test_route_table_is_cores_plus_exactly_the_frontend_routes(
    tasks, plugin
) -> None:
    app = core_app(tasks)
    before = all_routes(app)
    attach_plugins(app, [plugin], tasks)
    after = all_routes(app)

    # Every core route keeps its path, methods, endpoint and relative order.
    assert [row for row in after if row in before] == before
    added = [row for row in after if row not in before]
    for _, path, _, _ in added:
        assert path.startswith(f"{PLUGIN_PREFIX}/"), path
    expected = frontend_routes(file_route="transcription" in tasks)
    assert _pairs(added) == expected
    # The frontend's route set collides with no route of core's table.
    for route in expected:
        assert not _collides(route, before), route


# @spec ING-SHIM-020
@pytest.mark.parametrize("tasks", TASK_SETS)
def test_core_routes_stay_authoritative_and_nothing_outside_plugins_riva(
    tasks, plugin
) -> None:
    app = core_app(tasks)
    before = {key: _serving(app, *key) for key in CORE_AUTHORITATIVE}
    rows_before = all_routes(app)
    attach_plugins(app, [plugin], tasks)
    for key, endpoints in before.items():
        assert _serving(app, *key) == endpoints, key
    for row in all_routes(app):
        if row not in rows_before:
            assert row[1].startswith("/plugins/riva/"), row
    for path in ("/v1/health", "/v1/metrics"):
        assert _serving(app, "http", path, "GET") == [], path


# @spec ING-SHIM-020
async def test_core_realtime_still_serves_its_own_scope(
    test_app, fake_engine, fake_models, fake_transcription, plugin
) -> None:
    """Core's ``/v1/realtime`` scope reaches core's handler, unchanged."""
    from support.asgi import AsgiWebSocket
    from support.realtime import CORE_REALTIME_URL

    await init_state(
        test_app,
        fake_engine,
        models=fake_models,
        transcription=fake_transcription,
    )
    ws = AsgiWebSocket(test_app, CORE_REALTIME_URL)
    assert await ws.connect() is True
    assert (await ws.receive_json()) == {"type": "core.realtime"}
    await ws.wait_closed()
    scopes = test_app.state.core_realtime_scopes
    assert [s["path"] for s in scopes] == ["/v1/realtime"]


# @spec ING-EP-091
@pytest.mark.parametrize(("method", "path"), HTTP_FRONTEND)
def test_collision_with_http_ws_or_mount_prefix_raises(
    method, path, plugin
) -> None:
    from vllm_riva_frontend.lifecycle import StartupError

    tasks = TASK_SETS[0]
    app = core_app(tasks)
    earlier = RouteAddingPlugin("earlier", http=[(method, path)])
    with pytest.raises(StartupError) as raised:
        attach_plugins(app, [earlier, plugin], tasks)
    assert path in str(raised.value)


# @spec ING-EP-091
def test_collision_with_an_earlier_websocket_route_raises(plugin) -> None:
    from vllm_riva_frontend.lifecycle import StartupError

    tasks = TASK_SETS[0]
    app = core_app(tasks)
    earlier = RouteAddingPlugin("earlier", websocket=[REALTIME_PATH])
    with pytest.raises(StartupError) as raised:
        attach_plugins(app, [earlier, plugin], tasks)
    assert REALTIME_PATH in str(raised.value)


# @spec ING-EP-091
@pytest.mark.parametrize("mount", ["/plugins", "/plugins/riva", PLUGIN_PREFIX])
def test_collision_with_an_earlier_mount_prefix_raises(mount, plugin) -> None:
    from vllm_riva_frontend.lifecycle import StartupError

    tasks = TASK_SETS[0]
    app = core_app(tasks)
    earlier = RouteAddingPlugin("earlier", mounts=[mount])
    with pytest.raises(StartupError):
        attach_plugins(app, [earlier, plugin], tasks)


# @spec ING-EP-091
def test_other_methods_on_a_frontend_path_do_not_collide(plugin) -> None:
    tasks = TASK_SETS[0]
    app = core_app(tasks)
    version = f"{PLUGIN_PREFIX}/version"
    earlier = RouteAddingPlugin("earlier", http=[("POST", version)])
    attach_plugins(app, [earlier, plugin], tasks)
    assert len(_serving(app, "http", version, "GET")) == 1
    assert len(_serving(app, "http", version, "POST")) == 1


# @spec ING-EP-091
def test_unrelated_mount_and_sibling_namespace_do_not_collide(plugin) -> None:
    tasks = TASK_SETS[0]
    app = core_app(tasks)
    earlier = RouteAddingPlugin(
        "earlier",
        mounts=["/plugins/rivax"],
        http=[("GET", "/plugins/other/v1/version")],
    )
    attach_plugins(app, [earlier, plugin], tasks)
    assert ("GET", f"{PLUGIN_PREFIX}/version") in _pairs(all_routes(app))


class _PrependingPlugin(RouteAddingPlugin):
    """A later plugin that puts its WebSocket route first in the table."""

    def attach_router(self, app) -> None:
        before = list(app.router.routes)
        super().attach_router(app)
        new = [r for r in app.router.routes if r not in before]
        app.router.routes[:] = new + before


async def _init(app, fake_engine, fake_models, fake_transcription):
    return await init_state(
        app,
        fake_engine,
        models=fake_models,
        transcription=fake_transcription,
    )


# @spec ING-EP-070
@pytest.mark.parametrize(("method", "path"), HTTP_FRONTEND)
async def test_init_state_fails_when_a_later_plugin_shadows_a_frontend_route(
    method, path, plugin, fake_engine, fake_models, fake_transcription
) -> None:
    from vllm_riva_frontend.lifecycle import StartupError

    later = RouteAddingPlugin("later", http=[(method, path)])
    app = build_app([plugin, later])
    with pytest.raises(StartupError) as raised:
        await _init(app, fake_engine, fake_models, fake_transcription)
    assert path in str(raised.value)


# @spec ING-EP-070
async def test_init_state_fails_when_a_later_plugin_shadows_the_websocket(
    plugin, fake_engine, fake_models, fake_transcription
) -> None:
    from vllm_riva_frontend.lifecycle import StartupError

    later = _PrependingPlugin("later", websocket=[REALTIME_PATH])
    app = build_app([plugin, later])
    with pytest.raises(StartupError) as raised:
        await _init(app, fake_engine, fake_models, fake_transcription)
    assert REALTIME_PATH in str(raised.value)


# @spec ING-EP-070
async def test_init_state_passes_when_later_routes_do_not_shadow(
    plugin, fake_engine, fake_models, fake_transcription
) -> None:
    later = RouteAddingPlugin(
        "later",
        http=[("POST", f"{PLUGIN_PREFIX}/version"), ("GET", "/later")],
        websocket=[REALTIME_PATH],
    )
    app = build_app([plugin, later])
    await _init(app, fake_engine, fake_models, fake_transcription)
    assert plugin.factory is not None


def _metric_names() -> set[str]:
    from prometheus_client import REGISTRY

    return {metric.name for metric in REGISTRY.collect()}


# @spec ING-SHIM-020
async def test_plugin_registers_no_metric_families_of_its_own(
    test_app, fake_engine, fake_models, fake_transcription, plugin
) -> None:
    from support.app import start_plugins, stop_plugins

    before = _metric_names()
    state = await _init(test_app, fake_engine, fake_models, fake_transcription)
    await start_plugins(state)
    try:
        assert _metric_names() - before == set()
    finally:
        await stop_plugins(state)
