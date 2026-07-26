"""Static no-second-stack invariants for the downstream plugin package."""

from pathlib import Path


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
