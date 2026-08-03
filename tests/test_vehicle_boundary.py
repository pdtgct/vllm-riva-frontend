"""Static no-second-stack invariants for the downstream plugin package."""

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on the lowest supported version
    import tomli as tomllib


# @spec ING-VEH-005, ING-VEH-006, ING-VEH-007, ING-VEH-008, ING-VEH-014
def test_package_has_no_renderer_batcher_or_option_b_serve_dependency() -> None:
    package_root = Path(__file__).parents[1] / "src" / "vllm_riva_frontend"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in package_root.glob("*.py")
    )
    for forbidden in (
        "nemotron_omni_port.serve",
        "renderer",
        "batcher",
        "loopback",
    ):
        assert forbidden not in source


# @spec ING-VEH-003, ING-VEH-019, ING-VEH-021
def test_downstream_never_invokes_host_admission_transitions() -> None:
    """The frontend can observe admission and request shutdown, not close it."""
    package_root = Path(__file__).parents[1] / "src" / "vllm_riva_frontend"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in package_root.glob("*.py")
    )

    assert "close_before_owner_drain" not in source
    assert "close_from_launcher_thread" not in source
    assert "open_after_http_listener_bound" not in source


# @spec ING-SHIM-006
def test_plugin_never_registers_host_streaming_metric_families() -> None:
    """Host metrics stay host-owned: the plugin registers nothing.

    ENV-MOD-004 forbids this package registering or duplicating any
    ``vllm_omni:streaming_*`` family in the host default registry. The
    guarantee is structural rather than filtered: the package neither
    depends on nor imports ``prometheus_client``, so it holds no means
    to register any family at all, and it carries no literal from that
    namespace that a host-side helper could register on its behalf.
    """
    project_root = Path(__file__).parents[1]
    package_root = project_root / "src" / "vllm_riva_frontend"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in package_root.glob("*.py")
    )

    assert "prometheus_client" not in source
    assert "prometheus-client" not in (
        project_root / "pyproject.toml"
    ).read_text(encoding="utf-8")
    assert "vllm_omni:streaming_" not in source


# Qualification gate: downstream package coverage policy (not runtime EARS).
def test_default_gate_enforces_more_than_ninety_percent_line_coverage() -> None:
    project_root = Path(__file__).parents[1]
    with (project_root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)

    dev_dependencies = project["dependency-groups"]["dev"]
    addopts = project["tool"]["pytest"]["ini_options"]["addopts"]

    assert any(item.startswith("pytest-cov") for item in dev_dependencies)
    thresholds = [
        float(item.removeprefix("--cov-fail-under="))
        for item in addopts
        if item.startswith("--cov-fail-under=")
    ]
    assert thresholds and min(thresholds) > 90.0
    assert project["tool"]["coverage"]["report"]["precision"] >= 2
