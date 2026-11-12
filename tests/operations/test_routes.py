"""The frontend's route set and its four operational routes.

The routes live under ``/plugins/riva/v1`` (``support.app.PLUGIN_PREFIX``):
readiness, liveness, version and metadata, beside the Realtime bootstrap,
the Realtime WebSocket route and (with core's transcription route) the
frontend's file route.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.metadata
import json
from typing import Any

import pytest
from support.app import (
    BOOTSTRAP_ROUTE,
    OPERATIONAL_ROUTES,
    PLUGIN_PREFIX,
    all_routes,
    build_app,
    frontend_routes,
)

READY = {"object": "health.response", "message": "ready", "status": "ready"}
NOT_READY = {
    "object": "health.response",
    "message": "not ready",
    "status": "not_ready",
}
LIVE = {"object": "health.response", "message": "live", "status": "live"}
_OPERATIONAL = {
    path.rsplit("/v1/", 1)[1]: path for _, path in OPERATIONAL_ROUTES
}
READY_PATH = _OPERATIONAL["health/ready"]
LIVE_PATH = _OPERATIONAL["health/live"]
VERSION_PATH = _OPERATIONAL["version"]
METADATA_PATH = _OPERATIONAL["metadata"]
BOOTSTRAP_PATH = BOOTSTRAP_ROUTE[1]
PROVENANCE = {
    "deployment_image": "registry.example/riva-frontend:1.4.0",
    "pin": "0123456789abcdef0123456789abcdef01234567",
    "precision_policy": "fp32-bringup-v1",
}
SERVED = "nvidia/nemotron-asr-streaming-test"
TASK_SETS = [
    ("transcription", "realtime"),
    ("realtime",),
    ("generate", "transcription", "realtime"),
]


def _strings(value: Any) -> list[str]:
    """Every string value (not key) in a JSON document."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _pairs(app) -> set[tuple[str, str]]:
    """``(method, path)`` of every route; ``WEBSOCKET``/``MOUNT`` for those."""
    pairs: set[tuple[str, str]] = set()
    for kind, path, methods, _ in all_routes(app):
        if kind == "http":
            pairs |= {(m, path) for m in methods if m != "HEAD"}
        else:
            pairs.add((kind.upper(), path))
    return pairs


# @spec ING-SHIM-023
@pytest.mark.parametrize("tasks", TASK_SETS)
def test_route_set_is_exactly_seven_or_six_under_namespace(
    tasks, plugin
) -> None:
    without = _pairs(build_app([], supported_tasks=tasks))
    added = _pairs(build_app([plugin], supported_tasks=tasks)) - without
    file_route = "transcription" in tasks
    expected = frontend_routes(file_route=file_route)
    assert len(expected) == (7 if file_route else 6)
    assert added == expected
    for method, path in added:
        assert path.startswith(f"{PLUGIN_PREFIX}/"), (method, path)


# @spec ING-SHIM-009
def test_readiness_bodies() -> None:
    from vllm_riva_frontend.operational import readiness_body

    assert readiness_body(True) == (200, READY)
    assert readiness_body(False) == (503, NOT_READY)


# @spec ING-SHIM-018, ING-SHIM-009
async def test_readiness_is_503_before_start(app_state, client) -> None:
    async with client:
        response = await client.get(READY_PATH)
    assert response.status_code == 503
    assert response.json() == NOT_READY


# @spec ING-SHIM-018, ING-SHIM-009
async def test_readiness_is_200_once_started(started, client) -> None:
    async with client:
        response = await client.get(READY_PATH)
    assert response.status_code == 200
    assert response.json() == READY


# @spec ING-SHIM-018
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
        response = await client.get(READY_PATH)
    assert seen and seen[-1] is app_state
    assert response.status_code == (200 if ready else 503)
    assert response.json()["status"] == ("ready" if ready else "not_ready")


# @spec ING-SHIM-024
async def test_readiness_route_reports_ready_alone_and_core_health_is_engine_only(  # noqa: E501
    started, app_state, fake_engine, client
) -> None:
    calls: list[Any] = []

    def predicate(state) -> bool:
        calls.append(state)
        return predicate.answer

    predicate.answer = False  # type: ignore[attr-defined]
    started.ready = predicate
    async with client:
        # A not-ready plugin leaves core's /health reporting engine health
        # only: /health never reads the plugin's predicate.
        core_health = await client.get("/health")
        assert core_health.status_code == 200
        assert calls == []
        frontend = await client.get(READY_PATH)
        assert frontend.status_code == 503
        assert calls

        # Draining: core's /ready aggregates it; the frontend route does not.
        predicate.answer = True  # type: ignore[attr-defined]
        app_state.draining = True
        frontend = await client.get(READY_PATH)
        core_ready = await client.get("/ready")
        assert frontend.status_code == 200
        assert frontend.json() == READY
        assert core_ready.status_code == 503
        app_state.draining = False

        # Engine dead: core's /ready and /health see it; the frontend
        # route reports the predicate alone.
        fake_engine.die()
        frontend = await client.get(READY_PATH)
        core_ready = await client.get("/ready")
        core_health = await client.get("/health")
    assert frontend.status_code == 200
    assert core_ready.status_code == 503
    assert core_health.status_code == 503


# @spec ING-SHIM-018, ING-SHIM-010
async def test_liveness_is_200_before_start(app_state, client) -> None:
    async with client:
        response = await client.get(LIVE_PATH)
    assert response.status_code == 200
    assert response.json() == LIVE


# @spec ING-SHIM-018, ING-SHIM-010
async def test_liveness_stays_200_when_not_ready(started, client) -> None:
    started.ready = lambda state: False
    async with client:
        ready = await client.get(READY_PATH)
        live = await client.get(LIVE_PATH)
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


# @spec ING-SHIM-018, ING-SHIM-010
async def test_liveness_stays_live_while_draining_and_after_stop(
    started, app_state, client
) -> None:
    from support.app import stop_plugins

    app_state.draining = True
    async with client:
        draining = await client.get(LIVE_PATH)
        await asyncio.wait_for(stop_plugins(app_state), 5)
        stopped = await client.get(LIVE_PATH)
    for response in (draining, stopped):
        assert response.status_code == 200
        assert response.json() == LIVE


# @spec ING-SHIM-018, ING-SHIM-011
async def test_version_reports_the_distribution_release(
    app_state, client
) -> None:
    import vllm_riva_frontend
    from vllm_riva_frontend.version import __version__

    async with client:
        response = await client.get(VERSION_PATH)
    assert response.status_code == 200
    assert response.json() == {"release": __version__, "api": "v1"}
    assert vllm_riva_frontend.__version__ == __version__
    try:
        installed = importlib.metadata.version("vllm-riva-frontend")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed is not None:
        assert installed == __version__


# @spec ING-SHIM-018
async def test_core_version_is_untouched_by_the_frontend_version(
    app_state, client
) -> None:
    from vllm_riva_frontend.version import __version__

    async with client:
        frontend = await client.get(VERSION_PATH)
        core = await client.get("/version")
    assert frontend.status_code == 200
    assert core.status_code == 200
    assert core.json() != frontend.json()
    assert core.json().get("release") != __version__


@pytest.fixture
def riva_options():
    return dict(PROVENANCE)


# @spec ING-SHIM-025
async def test_metadata_version_equals_version_release(
    app_state, client
) -> None:
    async with client:
        version = await client.get(VERSION_PATH)
        metadata = await client.get(METADATA_PATH)
    assert version.status_code == 200
    assert metadata.status_code == 200
    assert metadata.json()["version"] == version.json()["release"]


# @spec ING-SHIM-018, ING-SHIM-012
async def test_metadata_has_the_nim_shape_with_this_deployments_values(
    app_state, plugin, client
) -> None:
    async with client:
        response = await client.get(METADATA_PATH)
    assert response.status_code == 200
    body = response.json()
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


# @spec ING-SHIM-019
async def test_metadata_reports_only_deployment_identifiers(
    app_state, plugin, client
) -> None:
    from vllm_riva_frontend.version import __version__

    async with client:
        response = await client.get(METADATA_PATH)
    assert response.status_code == 200
    body = response.json()
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


# @spec ING-SHIM-012, ING-SHIM-019
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


# @spec ING-SHIM-027
async def test_metadata_and_realtime_session_share_one_provenance_object(
    started, test_app, client
) -> None:
    from support.asgi import AsgiWebSocket

    from vllm_riva_frontend.config import DeploymentProvenance

    built = started.provenance
    assert isinstance(built, DeploymentProvenance)
    # Swap the one object: every surface must publish the swapped values,
    # so none keeps a copy of its own or rebuilds it from configuration.
    swapped = dataclasses.replace(
        built,
        **{f.name: f"swapped-{f.name}" for f in dataclasses.fields(built)},
    )
    started.provenance = swapped
    expected = dataclasses.asdict(swapped)

    async with client:
        metadata = await client.get(METADATA_PATH)
        bootstrap = await client.post(BOOTSTRAP_PATH, json={})
    assert metadata.status_code == 200
    assert metadata.json()["provenance"] == expected
    assert bootstrap.status_code == 200
    session = bootstrap.json()
    if "provenance" in session:
        assert session["provenance"] == expected

    ws = AsgiWebSocket(test_app)
    assert await ws.connect() is True
    assert (await ws.receive_json())["type"] == "conversation.created"
    await ws.send_json(
        {"type": "transcription_session.update", "session": session}
    )
    updated = await ws.receive_json()
    assert updated["type"] == "transcription_session.updated", updated
    if "provenance" in updated.get("session", {}):
        assert updated["session"]["provenance"] == expected
    await ws.disconnect()
    await ws.wait_closed()
