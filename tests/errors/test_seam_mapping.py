"""Seam reason mapping: each ``SessionRefused`` and ``SessionFailed`` reason.

The seam errors are ``support.fake_session.SeamError`` instances, which carry
only the two attributes the frontend reads with ``getattr``.
"""

from __future__ import annotations

import pytest
from support.fake_session import SeamError, failed, refused

REFUSED = {
    "locale": "unknown_locale",
    "cadence": "internal",
    "capacity": "busy",
    "active_limit": "busy",
    "engine_paused": "service_unavailable",
    "request_id_in_use": "internal",
}

FAILED = {
    "expired": "idle_timeout",
    "engine_dead": "service_unavailable",
    "rejected_input": "internal",
    "abnormal_end": "internal",
    "state": "internal",
    "input_capacity": "buffer_overflow",
    "output_capacity": "buffer_overflow",
}


def _project(exc: BaseException):
    from vllm_riva_frontend.errors import project_seam_error

    return project_seam_error(exc)


# @spec ING-ERR-008
@pytest.mark.parametrize("reason", sorted(REFUSED))
def test_each_refused_reason_maps_on_its_own(reason) -> None:
    assert _project(refused(reason)).code == REFUSED[reason]


# @spec ING-ERR-008
@pytest.mark.parametrize("reason", ["capacity", "active_limit"])
def test_engine_capacity_refusal_is_busy_with_provider_authority(
    reason,
) -> None:
    error = _project(refused(reason))
    assert error.code == "busy"
    assert error.authority == "provider"


# @spec ING-ERR-013
@pytest.mark.parametrize("reason", sorted(FAILED))
def test_each_failed_reason_maps_on_its_own(reason) -> None:
    assert _project(failed(reason)).code == FAILED[reason]


# @spec ING-ERR-013, ING-FE-015
@pytest.mark.parametrize("reason", ["input_capacity", "output_capacity"])
def test_capacity_failures_are_buffer_overflow_naming_the_layer(
    reason,
) -> None:
    error = _project(failed(reason))
    assert error.code == "buffer_overflow"
    named = " ".join([error.detail, *error.fields]).lower()
    layer = "input" if reason == "input_capacity" else "output"
    assert layer in named
    other = "output" if layer == "input" else "input"
    assert other not in named


# @spec ING-ERR-014
def test_cancelled_with_no_selected_cause_is_internal() -> None:
    assert _project(failed("cancelled")).code == "internal"


# @spec ING-ERR-010
def test_exception_without_session_error_is_internal() -> None:
    assert _project(RuntimeError("boom")).code == "internal"
    assert _project(ValueError("bad samples")).code == "internal"


# @spec ING-ERR-010
@pytest.mark.parametrize(
    "exc",
    [
        SeamError("failed", "unheard_of"),
        SeamError("refused", "unheard_of"),
    ],
)
def test_unknown_reason_is_internal(exc) -> None:
    assert _project(exc).code == "internal"


# @spec ING-ERR-010, ING-CORE-013
def test_seam_attributes_are_read_with_getattr() -> None:
    from vllm_riva_frontend.errors import seam_error

    assert seam_error(failed("expired")) == ("failed", "expired")
    assert seam_error(refused("capacity")) == ("refused", "capacity")
    assert seam_error(RuntimeError("x")) == (None, None)

    class OnlyReason(Exception):
        reason = "engine_dead"

    assert seam_error(OnlyReason()) == (None, "engine_dead")


# @spec ING-ERR-010
def test_engine_dead_is_the_escalating_reason() -> None:
    from vllm_riva_frontend.errors import ESCALATES

    assert frozenset({"engine_dead"}) == ESCALATES
