"""The sole downstream inference boundary: the RFC-1 factory and lease."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol

PieceAccepted = Callable[[int], None]
CleanupFaultReporter = Callable[[BaseException], None]


class SessionLease(Protocol):
    """Transport-neutral operations owned by RFC-1."""

    async def feed(
        self, samples: object, *, on_accepted: PieceAccepted
    ) -> list[str]:
        """Submit one normalized piece and return parked hypotheses."""
        ...

    async def update_locale(self, locale: str) -> object:
        """Apply a validated locale to the next carrier mint."""
        ...

    async def flush(self) -> str:
        """Drain final-tail work and return the terminal transcript."""
        ...

    async def finish(self) -> None:
        """Complete normal engine cleanup."""
        ...

    async def abort(self) -> None:
        """Terminate an abnormal session."""
        ...

    async def release(self) -> None:
        """Release the lease after terminal cleanup."""
        ...


class SessionFactory(Protocol):
    """Creates one RFC-1 lease without exposing engine/rendering mechanics."""

    async def open(self, *, cadence: str, locale: str) -> SessionLease:
        """Open one validated engine-local session."""
        ...


class DirectLeaseOwner:
    """Serializes one compatibility owner over the direct RFC-1 lease."""

    def __init__(
        self,
        factory: SessionFactory,
        *,
        cleanup_timeout: float,
        fault_reporter: CleanupFaultReporter | None = None,
    ) -> None:
        """Bind one factory and finite terminal-cleanup bound to this owner."""
        if cleanup_timeout <= 0:
            raise ValueError("cleanup_timeout must be positive")
        self._factory = factory
        self._cleanup_timeout = cleanup_timeout
        self._fault_reporter = fault_reporter
        self._lock = asyncio.Lock()
        self._open_started = False
        self._lease: SessionLease | None = None
        self._terminal: str | None = None
        self._tail: str | None = None
        self._terminal_error: BaseException | None = None
        self._late_open_cleanup: asyncio.Task[None] | None = None

    async def _bounded_cleanup(self, operation: Awaitable[None]) -> None:
        """Await one shielded cleanup call without overlap on timeout."""
        try:
            await asyncio.wait_for(
                asyncio.shield(operation), self._cleanup_timeout
            )
        except BaseException as error:
            self._report_cleanup_fault(error)
            raise

    def _report_cleanup_fault(self, error: BaseException) -> None:
        """Report one unresolved or failed terminal-cleanup operation."""
        if self._fault_reporter is not None:
            self._fault_reporter(error)

    async def _release_after_abort(self, lease: SessionLease) -> None:
        """Perform ordered abnormal cleanup without overlapping lease calls."""
        abort_error: BaseException | None = None
        try:
            await self._bounded_cleanup(lease.abort())
        except TimeoutError:
            # The abort may still be executing under the shield.  A release now
            # would overlap terminal lease calls, so the caller escalates.
            raise
        except BaseException as error:
            abort_error = error
        try:
            await self._bounded_cleanup(lease.release())
        except BaseException:
            if abort_error is not None:
                abort_error.add_note(
                    "release also failed during abnormal cleanup"
                )
                raise abort_error from None
            raise
        if abort_error is not None:
            raise abort_error

    def _track_late_open(self, open_task: asyncio.Task[SessionLease]) -> None:
        """Abort/release a lease that arrives after bounded open settlement."""

        async def cleanup() -> None:
            try:
                lease = await open_task
            except BaseException as error:
                self._terminal_error = error
                self._report_cleanup_fault(error)
                return
            try:
                await self._release_after_abort(lease)
            except BaseException as error:
                # The bounded cleanup primitive already reported the
                # unresolved terminal operation. Retain it without emitting
                # a duplicate process-health signal.
                self._terminal_error = error

        self._late_open_cleanup = asyncio.create_task(cleanup())

    def _lease_or_raise(self) -> SessionLease:
        """Return the configured lease while normal work remains possible."""
        if self._lease is None:
            raise RuntimeError("open must succeed before lease operations")
        return self._lease

    # @spec ING-LIFE-012, ING-LIFE-013
    async def open(self, *, cadence: str, locale: str) -> None:
        """Open exactly one lease and clean it if cancellation wins mid-open."""
        async with self._lock:
            if self._open_started:
                raise RuntimeError(
                    "a DirectLeaseOwner manages exactly one lease"
                )
            self._open_started = True
            open_task = asyncio.create_task(
                self._factory.open(cadence=cadence, locale=locale)
            )
            try:
                self._lease = await asyncio.shield(open_task)
            except asyncio.CancelledError:
                self._terminal = "abnormal"
                try:
                    self._lease = await asyncio.wait_for(
                        asyncio.shield(open_task), self._cleanup_timeout
                    )
                except TimeoutError as error:
                    self._track_late_open(open_task)
                    self._report_cleanup_fault(error)
                    raise
                except asyncio.CancelledError:
                    raise
                await self._release_after_abort(self._lease_or_raise())
                raise
            except BaseException:
                self._terminal = "abnormal"
                raise

    # @spec ING-FE-005, ING-LIFE-012
    async def feed(
        self, samples: object, *, on_accepted: PieceAccepted
    ) -> list[str]:
        """Serialize one accepted audio piece through the owned lease."""
        async with self._lock:
            if self._terminal is not None:
                raise RuntimeError("session terminal")
            return await self._lease_or_raise().feed(
                samples, on_accepted=on_accepted
            )

    # @spec ING-LIFE-007, ING-LIFE-012
    async def update_locale(self, locale: str) -> object:
        """Serialize one validated locale update through the owned lease."""
        async with self._lock:
            if self._terminal is not None:
                raise RuntimeError("session terminal")
            return await self._lease_or_raise().update_locale(locale)

    # @spec ING-LIFE-010, ING-LIFE-012, ING-LIFE-013
    async def complete(self) -> str | None:
        """Run normal flush → finish → release cleanup under one owner lock."""
        async with self._lock:
            if self._terminal == "normal":
                return self._tail
            if self._terminal == "normal_failed":
                assert self._terminal_error is not None
                raise self._terminal_error
            if self._terminal == "abnormal":
                return None
            lease = self._lease_or_raise()
            try:
                transcript = await lease.flush()
                await lease.finish()
            except BaseException as primary_error:
                # This terminal direction has been selected even if its
                # cleanup raises.  A retry must never invoke finish, abort,
                # or release a second time on the same lease.
                self._terminal = "abnormal"
                try:
                    await self._release_after_abort(lease)
                except BaseException as cleanup_error:
                    primary_error.add_note(
                        f"abnormal cleanup failed: {cleanup_error!r}"
                    )
                raise
            # A successful finish commits the normal engine terminal.  A later
            # release failure is unhealthy, but abort would be an illegal
            # second terminal operation.
            self._terminal = "normal_finishing"
            try:
                await self._bounded_cleanup(lease.release())
            except BaseException as error:
                self._terminal = "normal_failed"
                self._terminal_error = error
                raise
            self._terminal = "normal"
            self._tail = transcript
            return transcript

    # @spec ING-LIFE-004, ING-LIFE-010, ING-LIFE-013
    async def cancel(self) -> None:
        """Select abnormal abort → release cleanup under one owner lock."""
        async with self._lock:
            if self._terminal in {
                "normal",
                "normal_finishing",
                "normal_failed",
            }:
                return
            if self._terminal == "abnormal":
                return
            lease = self._lease_or_raise()
            self._terminal = "abnormal"
            await self._release_after_abort(lease)
