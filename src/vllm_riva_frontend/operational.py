"""Plugin-owned compatibility operations without host-route shadowing."""

from collections.abc import Mapping

OPERATIONAL_PATHS = frozenset(
    {
        "/v1/health/ready",
        "/v1/health/live",
        "/v1/version",
        "/v1/metadata",
    }
)


def _health(status: str, *, available: bool) -> tuple[int, dict[str, str]]:
    """Return the NIM-shaped health object for one process state."""
    if available:
        return 200, {
            "object": "health.response",
            "message": status,
            "status": status,
        }
    unavailable = f"not {status}"
    return 503, {
        "object": "health.response",
        "message": unavailable,
        "status": unavailable.replace(" ", "_"),
    }


# @spec ING-SHIM-001, ING-SHIM-002, ING-SHIM-006
def operational_response(
    path: str,
    *,
    ready: bool,
    live: bool,
    release: str = "unknown",
    api: str = "v1",
    model: str = "unknown",
    provenance: Mapping[str, str] | None = None,
) -> tuple[int, object] | None:
    """Return one plugin-owned response or pass an unowned path through.

    ``provenance`` must already be this deployment's own server-authored
    identifiers -- see ``build_deployment_provenance``, the only allowed
    constructor -- and is embedded verbatim; this function does not (and
    structurally cannot, given a flat ``Mapping[str, str]``) filter it.
    """
    if path == "/v1/health/ready":
        return _health("ready", available=ready)
    if path == "/v1/health/live":
        return _health("live", available=live)
    if path == "/v1/version":
        return 200, {"release": release, "api": api}
    if path == "/v1/metadata":
        return 200, {
            "version": release,
            "modelInfo": [{"modelUrl": "", "shortName": model}],
            "repository_override": "",
            "assetInfo": [],
            "licenseInfo": {},
            "provenance": dict(provenance or {}),
        }
    return None
