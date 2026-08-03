"""The sole downstream inference boundary: the RFC-1 factory and lease."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol

PieceAccepted = Callable[[int], None]
CleanupFaultReporter = Callable[[BaseException], None]


def add_note(error: BaseException, note: str) -> None:
    """Retain one secondary diagnostic on an in-flight primary error.

    ``BaseException.add_note`` arrived in Python 3.11. Below it, write the
    same ``__notes__`` list the method maintains, so the diagnostic is
    retained and inspectable on every supported interpreter; only
    traceback *rendering* of notes is version-dependent.
    """
    native = getattr(error, "add_note", None)
    if native is not None:
        native(note)
        return
    notes = getattr(error, "__notes__", None)
    if notes is None:
        notes = []
        error.__notes__ = notes  # type: ignore[attr-defined]
    notes.append(note)


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

    async def flush(self) -> object:
        """Drain final-tail work and return the terminal output.

        The 0.24 host line returns the terminal transcript itself; the
        0.25 bounded-stream line returns a structured terminal result
        whose ``complete_text`` carries the transcript. The owner
        normalizes both shapes through ``_terminal_transcript``.
        """
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


def _terminal_transcript(terminal: object) -> str:
    """Normalize the lease's terminal shape across qualified host lines.

    Every dialect surface consumes exactly one thing from finalization:
    the terminal transcript string. Which shape ``flush`` returns is a
    property of the host line this package admits — a plain transcript
    on 0.24, a structured terminal result carrying ``complete_text``
    (plus a completion acknowledgement this frontend does not consume)
    on the 0.25 bounded-stream line. Any other shape is a host-contract
    violation and must fail loudly rather than serialize garbage into a
    dialect final.
    """
    if isinstance(terminal, str):
        return terminal
    text = getattr(terminal, "complete_text", None)
    if isinstance(text, str):
        return text
    raise TypeError(
        "lease flush returned neither a transcript string nor a "
        f"structured terminal result: {type(terminal).__name__}"
    )


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
        except asyncio.TimeoutError:
            # The abort may still be executing under the shield.  A release now
            # would overlap terminal lease calls, so the caller escalates.
            raise
        except BaseException as error:
            abort_error = error
        try:
            await self._bounded_cleanup(lease.release())
        except BaseException:
            if abort_error is not None:
                add_note(
                    abort_error,
                    "release also failed during abnormal cleanup",
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
                except asyncio.TimeoutError as error:
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
                # A shape violation is a flush-contract failure and takes
                # the same abort -> release path as a failed flush.
                transcript = _terminal_transcript(await lease.flush())
                await lease.finish()
            except BaseException as primary_error:
                # This terminal direction has been selected even if its
                # cleanup raises.  A retry must never invoke finish, abort,
                # or release a second time on the same lease.
                self._terminal = "abnormal"
                try:
                    await self._release_after_abort(lease)
                except BaseException as cleanup_error:
                    add_note(
                        primary_error,
                        f"abnormal cleanup failed: {cleanup_error!r}",
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
