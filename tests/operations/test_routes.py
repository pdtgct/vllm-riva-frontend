"""The four operational routes: readiness, liveness, version and metadata."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
from typing import Any

import pytest

READY = {"object": "health.response", "message": "ready", "status": "ready"}
NOT_READY = {
    "object": "health.response",
    "message": "not ready",
    "status": "not_ready",
}
LIVE = {"object": "health.response", "message": "live", "status": "live"}
OPERATIONAL = (
    "/v1/health/ready",
    "/v1/health/live",
    "/v1/version",
    "/v1/metadata",
)
PROVENANCE = {
    "deployment_image": "registry.example/riva-frontend:1.4.0",
    "pin": "0123456789abcdef0123456789abcdef01234567",
    "precision_policy": "fp32-bringup-v1",
}
SERVED = "nvidia/nemotron-asr-streaming-test"


def _strings(value: Any) -> list[str]:
    """Every string value (not key) in a JSON document."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


# @spec ING-SHIM-007
def test_the_operational_paths_are_exactly_four() -> None:
    from vllm_riva_frontend.operational import OPERATIONAL_PATHS

    assert set(OPERATIONAL_PATHS) == set(OPERATIONAL)


# @spec ING-SHIM-009
def test_readiness_bodies() -> None:
    from vllm_riva_frontend.operational import readiness_body

    assert readiness_body(True) == (200, READY)
    assert readiness_body(False) == (503, NOT_READY)


# @spec ING-SHIM-007, ING-SHIM-009
async def test_readiness_is_503_before_start(app_state, client) -> None:
    async with client:
        response = await client.get("/v1/health/ready")
    assert response.status_code == 503
    assert response.json() == NOT_READY


# @spec ING-SHIM-007, ING-SHIM-009
async def test_readiness_is_200_once_started(started, client) -> None:
    async with client:
        response = await client.get("/v1/health/ready")
    assert response.status_code == 200
    assert response.json() == READY


# @spec ING-SHIM-007
@pytest.mark.parametrize("ready", [True, False])
async def test_readiness_answers_from_the_ready_predicate(
    ready, started, app_state, client
) -> None:
    seen = []

    def predicate(state) -> bool:
        seen.append(state)
        return ready

    started.ready = predicate
    async with client:
        response = await client.get("/v1/health/ready")
    assert seen and seen[-1] is app_state
    assert response.status_code == (200 if ready else 503)
    assert response.json()["status"] == ("ready" if ready else "not_ready")


# @spec ING-SHIM-010
async def test_liveness_is_200_before_start(app_state, client) -> None:
    async with client:
        response = await client.get("/v1/health/live")
    assert response.status_code == 200
    assert response.json() == LIVE


# @spec ING-SHIM-010
async def test_liveness_stays_200_when_not_ready(started, client) -> None:
    started.ready = lambda state: False
    async with client:
        ready = await client.get("/v1/health/ready")
        live = await client.get("/v1/health/live")
    assert ready.status_code == 503
    assert live.status_code == 200
    assert live.json() == LIVE


# @spec ING-SHIM-010
def test_liveness_defines_no_not_live_response() -> None:
    import inspect

    from vllm_riva_frontend import operational

    # The body takes no input, so no state can select another answer.
    assert inspect.signature(operational.liveness_body).parameters == {}
    assert "not_live" not in json.dumps(operational.liveness_body())


# @spec ING-SHIM-010
async def test_liveness_stays_live_while_draining_and_after_stop(
    started, app_state, client
) -> None:
    from support.app import stop_plugins

    app_state.draining = True
    async with client:
        draining = await client.get("/v1/health/live")
        await asyncio.wait_for(stop_plugins(app_state), 5)
        stopped = await client.get("/v1/health/live")
    for response in (draining, stopped):
        assert response.status_code == 200
        assert response.json() == LIVE


# @spec ING-SHIM-011
async def test_version_reports_the_distribution_release(
    app_state, client
) -> None:
    import vllm_riva_frontend
    from vllm_riva_frontend.version import __version__

    async with client:
        response = await client.get("/v1/version")
    assert response.status_code == 200
    assert response.json() == {"release": __version__, "api": "v1"}
    assert vllm_riva_frontend.__version__ == __version__
    try:
        installed = importlib.metadata.version("vllm-riva-frontend")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed is not None:
        assert installed == __version__


@pytest.fixture
def riva_options():
    return dict(PROVENANCE)


# @spec ING-SHIM-017
async def test_metadata_version_equals_the_release(app_state, client) -> None:
    async with client:
        version = (await client.get("/v1/version")).json()
        metadata = await client.get("/v1/metadata")
    assert metadata.status_code == 200
    assert metadata.json()["version"] == version["release"]


# @spec ING-SHIM-012
async def test_metadata_has_the_nim_shape_with_this_deployments_values(
    app_state, plugin, client
) -> None:
    async with client:
        body = (await client.get("/v1/metadata")).json()
    assert len(body["modelInfo"]) == 1
    assert body["modelInfo"][0]["shortName"] == plugin.served_model == SERVED
    for empty in ("repository_override", "assetInfo", "licenseInfo"):
        assert empty in body
        assert not body[empty], empty
    provenance = body["provenance"]
    assert len(provenance) == 4
    assert provenance["resampler"] == "scipy-poly-v1"
    assert set(provenance.values()) == {
        PROVENANCE["deployment_image"],
        PROVENANCE["pin"],
        PROVENANCE["precision_policy"],
        "scipy-poly-v1",
    }


# @spec ING-SHIM-002, ING-SHIM-012
async def test_metadata_reports_only_this_deployments_identifiers(
    app_state, plugin, client
) -> None:
    from vllm_riva_frontend.version import __version__

    async with client:
        body = (await client.get("/v1/metadata")).json()
    allowed = {
        __version__,
        SERVED,
        PROVENANCE["deployment_image"],
        PROVENANCE["pin"],
        PROVENANCE["precision_policy"],
        "scipy-poly-v1",
        "",
    }
    assert set(_strings(body)) <= allowed
    text = json.dumps(body).lower()
    for marker in ("ngc", "nvcr.io", "://", "profile"):
        assert marker not in text, marker


# @spec ING-SHIM-012, ING-SHIM-002
def test_metadata_body_is_built_from_the_typed_provenance() -> None:
    from vllm_riva_frontend.config import (
        build_deployment_provenance,
        parse_config,
    )
    from vllm_riva_frontend.operational import metadata_body
    from vllm_riva_frontend.version import __version__

    config = parse_config({"riva_frontend": dict(PROVENANCE)})
    provenance = build_deployment_provenance(config)
    body = metadata_body("served/name", provenance)
    if isinstance(body, tuple):
        status, body = body
        assert status == 200
    assert body["version"] == __version__
    assert body["modelInfo"][0]["shortName"] == "served/name"
    assert set(body["provenance"].values()) == {
        PROVENANCE["deployment_image"],
        PROVENANCE["pin"],
        PROVENANCE["precision_policy"],
        "scipy-poly-v1",
    }
