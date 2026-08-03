"""Atomic compatibility admission, ownership, and load shedding."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, TypeAlias

INFERENCE_OWNER_KINDS = frozenset(
    {
        "grpc_streaming_recognize",
        "grpc_recognize",
        "nim_realtime_transcription",
        "nim_http_transcription",
    }
)
EXCLUDED_OWNER_KINDS = frozenset(
    {
        "grpc_get_config",
        "nim_bootstrap",
        "operational",
        "host_operational",
        "native_realtime",
    }
)


@dataclass(frozen=True)
class LoadShedRejected:
    """Retryable rejection before an owner or lease has been created."""

    code: str = "busy"
    authority: str = "load_shed"


class AdmissionLease(Protocol):
    """One idempotently releasable host-admission owner."""

    def release(self) -> None:
        """Synchronously release the host's admitted-work registration."""


class HostAdmission(Protocol):
    """The host-owned composed-readiness authority."""

    def try_acquire(self) -> AdmissionLease | None:
        """Atomically register new work or reject after admission closes."""


class _UnmanagedAdmissionLease:
    """No-op lease for direct tests which do not install a host."""

    def release(self) -> None:
        """Preserve the same terminal-release shape without host state."""


_UNMANAGED_ADMISSION_LEASE = _UnmanagedAdmissionLease()


def try_acquire_admission(
    admission: HostAdmission | None,
) -> AdmissionLease | None:
    """Acquire host admission, treating an absent direct-test host as open."""
    if admission is None:
        return _UNMANAGED_ADMISSION_LEASE
    return admission.try_acquire()


class OwnerToken(Protocol):
    """One lifecycle-visible owner-task registration."""

    async def release(self) -> None:
        """Release the task registration after terminal cleanup."""


OwnerRegister: TypeAlias = Callable[[str], Awaitable[OwnerToken]]


class LoadShedRegistration:
    """One atomically admitted and tracked compatibility inference owner."""

    def __init__(
        self,
        gate: "LoadShedGate",
        kind: str,
        owner_token: OwnerToken | None,
    ) -> None:
        """Bind this registration to its gate and counted owner family."""
        self._gate = gate
        self.kind = kind
        self._owner_token = owner_token
        self._released = False

    async def release(self) -> None:
        """Release this owner exactly once after terminal lease cleanup."""
        if self._released:
            return
        self._released = True
        try:
            if self._owner_token is not None:
                await self._owner_token.release()
        finally:
            async with self._gate._lock:
                self._gate._active -= 1

    async def __aenter__(self) -> "LoadShedRegistration":
        """Enter this admitted owner scope."""
        return self

    async def __aexit__(
        self, exc_type: object, exc: object, traceback: object
    ) -> None:
        """Release the owner even when its session exits abnormally."""
        del exc_type, exc, traceback
        await self.release()


class LoadShedGate:
    """Atomically gate owner tracking and the compatibility-local SLO cap."""

    def __init__(
        self,
        max_sessions: int,
        *,
        owner_register: OwnerRegister | None = None,
    ) -> None:
        """Create a gate with one finite compatibility-owner limit."""
        if type(max_sessions) is not int or max_sessions <= 0:
            raise ValueError("max_sessions must be a positive integer")
        self._max_sessions = max_sessions
        self._owner_register = owner_register
        self._active = 0
        self._closed = False
        self._lock = asyncio.Lock()

    @property
    def active(self) -> int:
        """Return the current number of registered compatibility owners."""
        return self._active

    # @spec ING-ADM-006, ING-LIFE-012
    async def register(
        self, kind: str
    ) -> LoadShedRegistration | LoadShedRejected | None:
        """Atomically admit and track one counted inference candidate."""
        if kind in EXCLUDED_OWNER_KINDS:
            return None
        if kind not in INFERENCE_OWNER_KINDS:
            raise ValueError(f"unknown owner kind: {kind}")
        async with self._lock:
            if self._closed:
                return LoadShedRejected(
                    code="service_unavailable",
                    authority="load_shed",
                )
            if self._active >= self._max_sessions:
                return LoadShedRejected()
            owner_token = (
                await self._owner_register(kind)
                if self._owner_register is not None
                else None
            )
            self._active += 1
            return LoadShedRegistration(self, kind, owner_token)

    async def close(self) -> None:
        """Linearize shutdown against every candidate registration."""
        async with self._lock:
            self._closed = True
