"""What the two streaming surfaces share.

The Riva gRPC servicer and the Realtime adapter drive the same stream owner
and registry record and apply the same locale and RIFF/WAVE header rules;
these helpers keep the two projections identical. Both surfaces read client
input only through ``StreamOwner.receive``, which applies the bound of the
owner's state (and the RPC deadline set on the owner) and raises
``lease.ShutdownObserved`` once the drain's trigger reached the owner; every
terminal outcome is the owner's one selected cause. Both send through
``bounded_send`` and wait for cleanup through ``settle_shielded``.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, TypeVar

from vllm_riva_frontend import errors, frontend, lease

T = TypeVar("T")


def locale_error(
    value: str,
    *,
    field: str,
    locales: frozenset[str],
    default_locale: str | None,
) -> errors.FrontendError | None:
    """The session leaf's locale rule for one client value."""
    if not value:
        if default_locale is None:
            return errors.FrontendError(
                "invalid_config_field",
                "this model has no default locale; name one",
                fields=(field,),
            )
        return None
    if value not in locales:
        return errors.FrontendError(
            "unknown_locale",
            f"{value!r} is not one of {', '.join(sorted(locales))}",
            fields=(field,),
        )
    return None


def owner_error(owner: Any) -> errors.FrontendError:
    """The owner's selected cause as the error a surface projects."""
    cause = owner.cause
    if cause is None:
        return errors.FrontendError("internal", "the stream ended")
    projected: errors.FrontendError | None = getattr(cause, "error", None)
    if projected is not None and projected.code == cause.code:
        return projected
    return errors.FrontendError(cause.code)


def selected_error(
    owner: Any, error: errors.FrontendError
) -> errors.FrontendError:
    """``error`` when it is the owner's one selected cause, else that cause.

    The owner keeps the first abnormal cause, so a later failure of the
    surface's own never replaces it on the wire.
    """
    cause = owner.cause
    if cause is None or cause.code == error.code:
        return error
    return owner_error(owner)


async def settle(owner: Any, error: errors.FrontendError, phase: str) -> None:
    """Select ``error`` as the owner's cause unless one stands; clean up.

    ``fail`` is the owner's one terminal path: it keeps an already selected
    cause and returns once cleanup has run.
    """
    await owner.fail(lease.Cause(error.code, phase, None))


def child(record: Any, awaitable: Awaitable[T]) -> asyncio.Future[T]:
    """A task owned by the stream's registry record (``stop`` reaps it)."""
    task = asyncio.ensure_future(awaitable)
    children = getattr(record, "children", None)
    if isinstance(children, set):
        children.add(task)
        task.add_done_callback(children.discard)
    return task


def unregister(registry: Any, record: Any) -> None:
    """Drop the record if the owner has not already done so."""
    with contextlib.suppress(KeyError, ValueError, LookupError):
        registry.unregister(record)


# @spec ING-LIFE-058, ING-EP-060
def release_owner(registry: Any, record: Any, admitted: Any) -> None:
    """Release a handler's record and admission as its handler ends.

    A ``record.continuation`` still running (an owner's cleanup that
    outlived the answer) holds both until it ends: they are released from
    its done callback instead, once. ``Admission.release`` is idempotent and
    ``unregister`` tolerates a record the continuation already removed.
    """
    continuation = getattr(record, "continuation", None)
    if continuation is not None and not continuation.done():

        def released(_: Any) -> None:
            unregister(registry, record)
            admitted.release()

        continuation.add_done_callback(released)
        return
    unregister(registry, record)
    admitted.release()


def torn_down(plugin: Any) -> bool:
    """Whether ``stop`` has run: cleanup then yields to its cancellation."""
    registry = getattr(plugin, "registry", None)
    return bool(getattr(plugin, "torn_down", False)) or bool(
        getattr(registry, "torn_down", False)
    )


async def join(
    tasks: Iterable[asyncio.Future[Any]], *, stopped: Callable[[], bool]
) -> None:
    """Wait until every task is done, whatever cancels the caller meanwhile.

    A cancellation that arrives while they run is absorbed, so the caller's
    record and admission outlive the cleanup they guard; it propagates at
    once only when ``stopped()`` holds, since ``stop`` reaps what is left.
    """
    pending = {task for task in tasks if not task.done()}
    while pending:
        try:
            await asyncio.wait(pending)
        except asyncio.CancelledError:
            if stopped():
                raise
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                current.uncancel()
        pending = {task for task in pending if not task.done()}


async def settle_shielded(
    plugin: Any,
    record: Any,
    owner: Any,
    error: errors.FrontendError,
    phase: str,
) -> None:
    """``settle`` as a child of the record, joined through cancellation.

    Returns only once cleanup has run (``join``); a cleanup that ``stop``
    cancelled ends the caller with ``CancelledError``.
    """
    cleanup = child(record, settle(owner, error, phase))
    await join((cleanup,), stopped=lambda: torn_down(plugin))
    if cleanup.cancelled():
        raise asyncio.CancelledError
    cleanup.result()


# ------------------------------------------------------------------- sends


class ShutdownWatch:
    """A stream record's shutdown observer: the owner first, then its sends.

    It forwards the drain's trigger to the stream owner, then wakes a send
    in progress, which recomputes its bound against the owner's ``D``.
    """

    def __init__(self, owner: Any) -> None:
        """Watch for the trigger on behalf of ``owner``."""
        self.owner = owner
        self.observed = asyncio.Event()

    def observe_shutdown(self, trigger: Any) -> Any:
        """Deliver ``trigger`` to the owner, then wake the sends."""
        try:
            return self.owner.observe_shutdown(trigger)
        finally:
            self.observed.set()


def watch_shutdown(registry: Any, record: Any, owner: Any) -> ShutdownWatch:
    """Put a ``ShutdownWatch`` in front of ``owner`` as the record's observer.

    ``attach_owner`` delivers a trigger already raised, which the owner
    ignores when it has already observed it.
    """
    watch = ShutdownWatch(owner)
    attach = getattr(registry, "attach_owner", None)
    if callable(attach):
        attach(record, watch)
    else:
        with contextlib.suppress(AttributeError, TypeError):
            record.owner = watch
    return watch


def _send_bound(
    limit: float, plugin: Any, owner: Any, rpc_deadline: float | None
) -> float:
    candidates = [
        rpc_deadline,
        getattr(plugin, "drain_deadline", None),
        None if owner is None else owner.drain_deadline,
    ]
    return min([limit, *(float(d) for d in candidates if d is not None)])


# @spec ING-LIFE-051, ING-ADM-016
async def bounded_send(
    plugin: Any,
    send: Awaitable[T],
    *,
    owner: Any = None,
    rpc_deadline: float | None = None,
    watch: ShutdownWatch | None = None,
) -> T:
    """One send or close under ``response_send_timeout``, on either surface.

    The bound runs from the send's start and is clipped to ``rpc_deadline``
    and, while draining, to ``D`` (the plugin's, or the owner's once it
    observed the trigger). A watched send is woken by the trigger and
    recomputes its bound, so a send already waiting when ``D`` is set
    observes it. A bound already reached leaves no wait: the send starts and
    runs to its first suspension, so a transport that takes it at once still
    gets it (the abort row at ``D``), and anything else is cancelled. Raises
    ``TimeoutError`` when the bound is exceeded; the send's own failure
    propagates.
    """
    clock = plugin.clock
    limit = clock.monotonic() + float(plugin.config.response_send_timeout)
    task = asyncio.ensure_future(send)
    first = True
    try:
        while not task.done():
            bound = _send_bound(limit, plugin, owner, rpc_deadline)
            if clock.monotonic() >= bound:
                if not first:
                    break
                first = False
                await asyncio.sleep(0)
                continue
            first = False
            waits: set[asyncio.Future[Any]] = {task}
            signal: asyncio.Future[Any] | None = None
            if watch is not None and not watch.observed.is_set():
                signal = asyncio.ensure_future(watch.observed.wait())
                waits.add(signal)
            try:
                await clock.wait_for(
                    asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED),
                    bound,
                )
            except TimeoutError:
                pass
            finally:
                if signal is not None:
                    signal.cancel()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.wait({task})
    if task.cancelled():
        raise TimeoutError
    return task.result()


def header_problem(
    header: frontend.RiffFormat | frontend.FormatError,
    declared_rate: int,
    declared_channels: int,
) -> frontend.FormatError | None:
    """Why a resolved header cannot be served, in the audio leaf's names.

    A declared non-zero rate or channel count is a constraint the header
    must match; the header's cell must be on the accept-matrix.
    """
    if isinstance(header, frontend.FormatError):
        return header
    fields: list[str] = []
    details: list[str] = []
    if declared_rate and header.sample_rate_hz != declared_rate:
        fields.append("sample_rate_hz")
        details.append(
            f"the header's {header.sample_rate_hz} Hz is not the declared "
            f"{declared_rate} Hz"
        )
    if declared_channels and header.channels != declared_channels:
        fields.append("channels")
        details.append(
            f"the header's {header.channels} channels are not the declared "
            f"{declared_channels}"
        )
    cell = frontend.validate_format(
        header.encoding, header.sample_rate_hz, header.channels
    )
    if cell is not None:
        fields.extend(cell.fields)
        details.append(cell.detail)
    if not fields:
        return None
    return frontend.FormatError(
        tuple(dict.fromkeys(fields)), "; ".join(d for d in details if d)
    )
