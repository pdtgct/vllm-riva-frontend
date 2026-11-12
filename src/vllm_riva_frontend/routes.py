"""The frontend's route set under ``/plugins/riva/v1/`` and its checks.

Every frontend route is new and lives under ``PLUGIN_PREFIX``: the four
operational routes, the Realtime bootstrap, the Realtime WebSocket route
and, beside core's ``POST /v1/audio/transcriptions``, the frontend's file
route. ``attach_router`` checks each against core's table before adding it
(``collisions``) and ``init_state`` checks that core's final table still
routes each to the frontend (``shadowed``). Nothing here imports vLLM.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from starlette.routing import BaseRoute, Match, Mount, Route, WebSocketRoute

#: The frontend's namespace; the NIM base URL clients are given.
PLUGIN_BASE = "/plugins/riva"
#: The prefix every frontend route lives under (ING-SHIM-023).
PLUGIN_PREFIX = f"{PLUGIN_BASE}/v1"
#: The paths the path-scoped authentication covers (ING-EP-083).
AUTH_SCOPE_PREFIX = f"{PLUGIN_BASE}/"
#: The Realtime bootstrap route (``POST``).
BOOTSTRAP_PATH = f"{PLUGIN_PREFIX}/realtime/transcription_sessions"
#: The Realtime WebSocket route.
REALTIME_PATH = f"{PLUGIN_PREFIX}/realtime"
#: The frontend's file route (``POST``).
FILE_ROUTE_PATH = f"{PLUGIN_PREFIX}/audio/transcriptions"
#: Core's complete-file route, beside which the file route is added.
CORE_FILE_ROUTE_PATH = "/v1/audio/transcriptions"

HTTP = "http"
WEBSOCKET = "websocket"


@dataclass(frozen=True, eq=False)
class FrontendRoute:
    """One route of the frontend's set.

    ``kind`` is ``HTTP`` (with ``method``) or ``WEBSOCKET`` (``method`` is
    ``None``). ``endpoint`` is what core's table must still route the path
    to after every plugin attached: the HTTP handler, or the ASGI app of the
    WebSocket route.
    """

    kind: str
    path: str
    method: str | None
    endpoint: Any

    def describe(self) -> str:
        """``GET /path`` or ``WebSocket /path``, as a startup error names it."""
        if self.kind == WEBSOCKET:
            return f"WebSocket {self.path}"
        return f"{self.method} {self.path}"


def http_route(method: str, path: str, endpoint: Any) -> FrontendRoute:
    """An HTTP route of the frontend's set."""
    return FrontendRoute(HTTP, path, method, endpoint)


def websocket_route(path: str, app: Any) -> FrontendRoute:
    """A WebSocket route of the frontend's set, served by the ASGI ``app``."""
    return FrontendRoute(WEBSOCKET, path, None, app)


def _http_methods(route: BaseRoute) -> frozenset[str] | None:
    """The methods of an HTTP route, or ``None`` for any other kind."""
    if isinstance(route, Route):
        return frozenset(route.methods or ())
    return None


def _under(path: str, mount_path: str) -> bool:
    """Whether a mount at ``mount_path`` covers ``path`` (segment prefix)."""
    return path == mount_path or path.startswith(mount_path.rstrip("/") + "/")


def serves(app: Any, method: str, path: str) -> bool:
    """Whether ``app`` has an HTTP route for ``method path``."""
    return any(
        getattr(route, "path", None) == path
        and method in (_http_methods(route) or ())
        for route in app.routes
    )


def core_serves_files(app: Any) -> bool:
    """Whether core's ``POST /v1/audio/transcriptions`` is registered."""
    return serves(app, "POST", CORE_FILE_ROUTE_PATH)


# @spec ING-EP-091
def collisions(app: Any, frontend: Iterable[FrontendRoute]) -> list[str]:
    """Every frontend route an existing route of ``app`` collides with.

    An existing HTTP route with the same path and method, an existing
    WebSocket route with the same path, or a mount whose path is a prefix
    of the frontend route's path; each failure names the frontend route.
    """
    failures: list[str] = []
    existing = list(app.routes)
    for wanted in frontend:
        for route in existing:
            path = getattr(route, "path", None)
            if path is None:
                continue
            methods = _http_methods(route)
            if isinstance(route, Mount):
                hit = _under(wanted.path, path)
                what = f"the mount at {path or '/'}"
            elif isinstance(route, WebSocketRoute):
                hit = wanted.kind == WEBSOCKET and path == wanted.path
                what = f"an existing WebSocket route {path}"
            elif methods is not None:
                hit = (
                    wanted.kind == HTTP
                    and path == wanted.path
                    and wanted.method in methods
                )
                what = f"an existing route {wanted.method} {path}"
            else:
                continue
            if hit:
                failures.append(
                    f"riva_frontend: {wanted.describe()} collides with "
                    f"{what}; the frontend adds every route under "
                    f"{PLUGIN_PREFIX}/ as new"
                )
                break
    return failures


def add_routes(app: Any, frontend: Iterable[FrontendRoute]) -> None:
    """Add each route to ``app``: new routes only, nothing replaced."""
    for route in frontend:
        if route.kind == WEBSOCKET:
            app.router.add_websocket_route(route.path, route.endpoint)
        else:
            assert route.method is not None
            app.add_api_route(
                route.path,
                route.endpoint,
                methods=[route.method],
                response_model=None,
            )


def _scope(route: FrontendRoute) -> dict[str, Any]:
    scope: dict[str, Any] = {
        "type": route.kind,
        "path": route.path,
        "root_path": "",
        "headers": [],
        "query_string": b"",
    }
    if route.kind == HTTP:
        scope["method"] = route.method
    return scope


def routing_target(app: Any, route: FrontendRoute) -> BaseRoute | None:
    """The first route of ``app``'s table that fully matches ``route``."""
    scope = _scope(route)
    candidates: list[BaseRoute] = list(app.router.routes)
    for candidate in candidates:
        match, _ = candidate.matches(scope)
        if match is Match.FULL:
            return candidate
    return None


# @spec ING-EP-070
def shadowed(app: Any, frontend: Iterable[FrontendRoute]) -> list[str]:
    """Every frontend route core's final table no longer routes to us.

    The first route that matches the path (and the method, for an HTTP
    route) must be the frontend's own; a later plugin's route or mount
    ahead of it fails startup, naming the path.
    """
    failures: list[str] = []
    for route in frontend:
        target = routing_target(app, route)
        if getattr(target, "endpoint", None) is not route.endpoint:
            failures.append(
                f"riva_frontend: {route.describe()} no longer reaches the "
                "frontend's handler; another endpoint plugin overrides or "
                "shadows it"
            )
    return failures
