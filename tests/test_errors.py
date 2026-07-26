"""Stable catalog and failure-projection tests first."""

from vllm_riva_frontend.errors import ERROR_CODES, catalog


# @spec ING-ERR-001, ING-ERR-002
def test_catalog_has_distinct_capacity_idle_and_finalize_codes() -> None:
    assert {
        "busy",
        "admission_wait_timeout",
        "idle_timeout",
        "finalization_timeout",
    } <= ERROR_CODES
    projections = catalog()
    assert projections["busy"].grpc_status == "RESOURCE_EXHAUSTED"
    assert projections["idle_timeout"].grpc_status == "ABORTED"


# @spec ING-ERR-001, ING-ERR-005, ING-ERR-006
def test_every_surface_failure_has_a_stable_non_silent_projection() -> None:
    projections = catalog()
    for code in ERROR_CODES:
        assert projections[code].grpc_status
        assert projections[code].nim_event
        assert projections[code].http_status >= 400


# @spec ING-ERR-004, ING-ERR-006
def test_session_terminal_and_service_unavailable_remain_distinct() -> None:
    projections = catalog()
    assert projections["session_terminal"] != projections["service_unavailable"]
