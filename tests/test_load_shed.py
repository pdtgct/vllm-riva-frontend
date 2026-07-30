"""Shared RFC-2 compatibility-owner load-shed contract."""

import asyncio

import pytest

from vllm_riva_frontend.admission import (
    EXCLUDED_OWNER_KINDS,
    INFERENCE_OWNER_KINDS,
    LoadShedGate,
    LoadShedRegistration,
    LoadShedRejected,
)


# @spec ING-ADM-006
def test_counter_includes_all_four_inference_families() -> None:
    assert {
        "grpc_streaming_recognize",
        "grpc_recognize",
        "nim_realtime_transcription",
        "nim_http_transcription",
    } == INFERENCE_OWNER_KINDS


# @spec ING-ADM-006
def test_counter_excludes_noninference_work() -> None:
    assert {
        "native_realtime",
        "nim_bootstrap",
        "grpc_get_config",
        "operational",
    } <= EXCLUDED_OWNER_KINDS


# @spec ING-ADM-006
def test_gate_rejects_invalid_capacity_and_unknown_owner_kind() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        LoadShedGate(0)
    with pytest.raises(ValueError, match="unknown owner kind"):
        asyncio.run(LoadShedGate(1).register("not-an-owner"))


# @spec ING-ADM-006, ING-LIFE-012
def test_atomic_limit_rejects_before_owner_creation() -> None:
    gate = LoadShedGate(max_sessions=1)
    first = asyncio.run(gate.register("grpc_streaming_recognize"))
    second = asyncio.run(gate.register("nim_http_transcription"))
    assert isinstance(first, LoadShedRegistration)
    assert second == LoadShedRejected()
    asyncio.run(first.release())
    assert gate.active == 0


# @spec ING-ADM-006, ING-LIFE-012
def test_context_releases_once_and_excluded_work_uses_no_slot() -> None:
    async def exercise() -> None:
        gate = LoadShedGate(max_sessions=1)
        assert await gate.register("nim_bootstrap") is None
        registration = await gate.register("grpc_recognize")
        assert isinstance(registration, LoadShedRegistration)
        async with registration:
            assert gate.active == 1
        await registration.release()
        assert gate.active == 0

    asyncio.run(exercise())


# @spec ING-ADM-006, ING-LIFE-012, ING-VEH-017
def test_shutdown_linearizes_against_registration_and_owner_failure() -> None:
    class FailingToken:
        async def release(self) -> None:
            raise RuntimeError("registry failure")

    async def exercise() -> None:
        async def register_owner(kind: str) -> FailingToken:
            assert kind == "grpc_recognize"
            return FailingToken()

        gate = LoadShedGate(
            1,
            owner_register=register_owner,
        )
        registration = await gate.register("grpc_recognize")
        assert isinstance(registration, LoadShedRegistration)
        await gate.close()
        rejected = await gate.register("nim_http_transcription")
        assert rejected == LoadShedRejected(
            code="service_unavailable",
            authority="load_shed",
        )
        with pytest.raises(RuntimeError, match="registry failure"):
            await registration.release()
        assert gate.active == 0

    asyncio.run(exercise())
