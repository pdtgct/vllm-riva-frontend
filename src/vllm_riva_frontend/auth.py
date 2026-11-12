"""Path-scoped authentication of the frontend's namespace ``/plugins/riva/``.

Core's ``AuthenticationMiddleware`` guards core's own prefixes only, so the
frontend adds its own middleware in ``attach_router``; core adds its
middleware after the plugin hooks, so core's WebSocket metrics,
authentication and CORS middleware wrap this one. It checks every in-scope
HTTP request and WebSocket handshake with the ``verify_token`` of an
instance of core's ``AuthenticationMiddleware`` that ``init_state`` builds
with the resolved tokens, so the token rule is core's own.

Every answer is sent inline in the ASGI call that received the request: no
task, no owner registry and no owner, because the middleware also answers
before ``init_state`` has built them (it then fails closed with 503).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from vllm_riva_frontend.config import PLUGIN_KEY, StartupError
from vllm_riva_frontend.routes import AUTH_SCOPE_PREFIX

#: Core's 401 body (``AuthenticationMiddleware``), pinned by ING-EP-089.
UNAUTHORIZED = {"error": "Unauthorized"}
#: The fail-closed body while authentication is unresolved (ING-EP-088).
UNAVAILABLE = {"error": "Service Unavailable"}
#: The ASGI WebSocket denial-response extension.
DENIAL_EXTENSION = "websocket.http.response"
#: Close codes of a handshake refused before accept (answered as HTTP 403).
_CLOSE_UNAUTHORIZED = 1008
_CLOSE_UNAVAILABLE = 1013


class AuthState:
    """What the middleware reads at request time; ``init_state`` resolves it.

    ``resolved`` is false until ``init_state`` has decided authentication;
    ``verifier`` is then core's ``AuthenticationMiddleware`` instance whose
    ``verify_token`` checks a request, or ``None`` while authentication is
    off. The tokens themselves are held only by the verifier.
    """

    def __init__(self) -> None:
        """Start unresolved: every in-scope request fails closed."""
        self.resolved = False
        self.verifier: Any = None

    def resolve(self, verifier: Any) -> None:
        """Turn authentication on (``verifier``) or off (``None``)."""
        self.verifier = verifier
        self.resolved = True


# @spec ING-EP-083
def in_scope(scope: Mapping[str, Any]) -> bool:
    """Whether the namespace's check applies to ``scope``.

    An ``http`` or ``websocket`` scope, not ``OPTIONS``, whose path after
    removing ``root_path`` (as core does) starts with ``/plugins/riva/``.
    """
    if scope.get("type") not in ("http", "websocket"):
        return False
    if scope.get("method") == "OPTIONS":
        return False
    root_path = scope.get("root_path", "") or ""
    path = str(scope.get("path", "")).removeprefix(root_path)
    return path.startswith(AUTH_SCOPE_PREFIX)


async def _refuse(
    scope: Scope,
    receive: Receive,
    send: Send,
    *,
    status: int,
    body: Mapping[str, str],
    close_code: int,
) -> None:
    """Answer an in-scope request inline; a handshake is never accepted."""
    if scope["type"] == "websocket" and DENIAL_EXTENSION not in (
        scope.get("extensions") or {}
    ):
        await send({"type": "websocket.close", "code": close_code})
        return
    # Over a WebSocket scope Starlette sends the denial response through
    # the extension (``websocket.http.response.start`` and ``.body``).
    await JSONResponse(dict(body), status_code=status)(scope, receive, send)


class PathScopedAuthMiddleware:
    """Pure-ASGI middleware checking every request under ``/plugins/riva/``.

    Registered by ``attach_router`` with ``app.add_middleware``; built by
    Starlette with the plugin's ``AuthState``.
    """

    def __init__(self, app: ASGIApp, *, auth: AuthState) -> None:
        """Wrap ``app`` (the routing below this middleware)."""
        self.app = app
        self.auth = auth

    # @spec ING-EP-084, ING-EP-087, ING-EP-088, ING-SHIM-021, ING-ERR-061
    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Check an in-scope request; pass every other scope unchanged."""
        if not in_scope(scope):
            await self.app(scope, receive, send)
            return
        auth = self.auth
        if not auth.resolved:
            await _refuse(
                scope,
                receive,
                send,
                status=503,
                body=UNAVAILABLE,
                close_code=_CLOSE_UNAVAILABLE,
            )
            return
        verifier = auth.verifier
        if verifier is not None and not verifier.verify_token(
            Headers(scope=scope)
        ):
            await _refuse(
                scope,
                receive,
                send,
                status=401,
                body=UNAUTHORIZED,
                close_code=_CLOSE_UNAUTHORIZED,
            )
            return
        await self.app(scope, receive, send)


# @spec ING-EP-085
def configured_keys(
    api_key: Sequence[str] | None, environment_key: str | None
) -> list[str]:
    """Core's API-key selection rule.

    The candidates are ``--api-key`` when that list is non-empty, otherwise
    the single value of ``VLLM_API_KEY``; empty strings are discarded.
    """
    candidates: list[str | None] = list(api_key or ()) or [environment_key]
    return [key for key in candidates if key]


def _readable_tokens(keywords: Mapping[str, Any]) -> list[str] | None:
    """A registration's ``tokens``: a non-empty list of non-empty strings."""
    tokens = keywords.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        return None
    if not all(isinstance(token, str) and token for token in tokens):
        return None
    return list(tokens)


# @spec ING-EP-086
def decide_tokens(
    key_configured: bool, registrations: Iterable[Mapping[str, Any]]
) -> list[str] | None:
    """The tokens to check with, ``None`` for authentication off.

    ``registrations`` are the keywords of each registration of core's
    ``AuthenticationMiddleware`` on the app. Off with no key and no
    registration; on with a configured key and exactly one readable
    registration (its tokens); every other combination fails startup with
    an error that names the signals compared and never a token.
    """
    entries = list(registrations)
    readable = [
        tokens
        for tokens in (_readable_tokens(keywords) for keywords in entries)
        if tokens is not None
    ]
    if not key_configured and not entries:
        return None
    if key_configured and len(readable) == 1:
        return readable[0]
    raise StartupError(
        [
            f"{PLUGIN_KEY}: cannot decide authentication for /plugins/riva/: "
            "an API key is "
            f"{'configured' if key_configured else 'not configured'} "
            "(--api-key, else VLLM_API_KEY, empty values discarded), and "
            f"the app has {len(entries)} registration(s) of core's "
            f"AuthenticationMiddleware, {len(readable)} of them with a "
            "readable tokens list (a non-empty list of non-empty strings); "
            "the frontend needs no key and no registration, or a key and "
            "exactly one readable registration"
        ]
    )
