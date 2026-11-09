"""The NIM operational routes: readiness, liveness, version and metadata.

The routes carry no inference behaviour. Readiness answers from the plugin's
own ``ready(state)`` predicate, the one core's ``/health`` reads, so the two
cannot disagree about the plugin.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, Protocol, cast

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from vllm_riva_frontend.config import DeploymentProvenance
from vllm_riva_frontend.version import __version__

#: The four plugin-owned operational paths (all ``GET``).
OPERATIONAL_PATHS = (
    "/v1/health/ready",
    "/v1/health/live",
    "/v1/version",
    "/v1/metadata",
)


class _OperationalPlugin(Protocol):
    """What the routes read from the plugin, at request time."""

    served_model: str | None
    provenance: DeploymentProvenance | None

    def ready(self, state: Any) -> bool:
        """Return the plugin's readiness predicate."""
        ...


def _health(status: str) -> dict[str, str]:
    return {"object": "health.response", "message": status, "status": status}


# @spec ING-SHIM-009
def readiness_body(ready: bool) -> tuple[int, dict[str, str]]:
    """Return the readiness status code and NIM-shaped body."""
    if ready:
        return 200, _health("ready")
    return 503, {
        "object": "health.response",
        "message": "not ready",
        "status": "not_ready",
    }


# @spec ING-SHIM-010
def liveness_body() -> dict[str, str]:
    """Return the liveness body; there is no ``not_live`` answer."""
    return _health("live")


# @spec ING-SHIM-011
def version_body() -> dict[str, str]:
    """Return the frontend distribution's release and the API version."""
    return {"release": __version__, "api": "v1"}


# @spec ING-SHIM-002, ING-SHIM-012, ING-SHIM-017
def metadata_body(
    served_model: str, provenance: DeploymentProvenance
) -> dict[str, Any]:
    """Return this deployment's own identity in the NIM metadata shape.

    ``provenance`` is the typed four-field object, so nothing but those
    identifiers can reach the body; the registry fields stay empty.
    """
    return {
        "version": __version__,
        "modelInfo": [{"modelUrl": "", "shortName": served_model}],
        "repository_override": "",
        "assetInfo": [],
        "licenseInfo": {},
        "provenance": dataclasses.asdict(provenance),
    }


# @spec ING-SHIM-007, ING-SHIM-008
def operational_router(plugin: _OperationalPlugin) -> APIRouter:
    """Return the router of the four operational ``GET`` routes.

    The handlers read the plugin when a request arrives, so the router can
    be attached before ``init_state`` has run.
    """
    router = APIRouter()

    async def health_ready(request: Request) -> JSONResponse:
        try:
            ready = bool(plugin.ready(request.app.state))
        except Exception:
            ready = False
        status, body = readiness_body(ready)
        return JSONResponse(body, status_code=status)

    async def health_live() -> JSONResponse:
        return JSONResponse(liveness_body())

    async def version() -> JSONResponse:
        return JSONResponse(version_body())

    async def metadata() -> JSONResponse:
        served_model = plugin.served_model
        provenance = plugin.provenance
        if served_model is None or provenance is None:
            status, body = readiness_body(False)
            return JSONResponse(body, status_code=status)
        return JSONResponse(metadata_body(served_model, provenance))

    for path, endpoint in zip(
        OPERATIONAL_PATHS,
        (health_ready, health_live, version, metadata),
        strict=True,
    ):
        router.add_api_route(
            path,
            cast(Callable[..., Any], endpoint),
            methods=["GET"],
            response_model=None,
        )
    return router
