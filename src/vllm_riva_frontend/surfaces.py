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
import inspect
import math
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, NoReturn, TypeVar

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


def hold(record: Any) -> None:
    """Make the handler, not the owner's cleanup, end ``record``.

    The record then stays registered while the handler sends its final,
    error or close after the cleanup ended, so the drain and ``stop`` still
    reach it; ``release_owner`` removes it when the handler ends.
    """
    with contextlib.suppress(AttributeError, TypeError):
        record.held_by_handler = True


def _holding(record: Any) -> asyncio.Future[Any] | None:
    """A task that still holds ``record``: its continuation or a send."""
    continuation: asyncio.Future[Any] | None = getattr(
        record, "continuation", None
    )
    if continuation is not None and not continuation.done():
        return continuation
    sends: Iterable[asyncio.Future[Any]] = getattr(
        record, "unsettled_sends", ()
    )
    for task in list(sends):
        if not task.done():
            return task
    return None


# @spec ING-LIFE-058, ING-EP-060
def release_owner(registry: Any, record: Any, admitted: Any) -> None:
    """Release a handler's record and admission as its handler ends.

    A ``record.continuation`` still running (an owner's cleanup that
    outlived the answer) or a send cancelled at its bound and not yet
    settled holds both until it ends: they are released from its done
    callback instead, once nothing holds the record. ``Admission.release``
    is idempotent and ``unregister`` tolerates a record already removed.
    """

    def release(_: Any = None) -> None:
        holder = _holding(record)
        if holder is not None:
            holder.add_done_callback(release)
            return
        unregister(registry, record)
        admitted.release()

    release()


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


def _send_bound(
    limit: float, plugin: Any, owner: Any, rpc_deadline: float | None
) -> float:
    """The send's absolute bound as it stands now.

    ``limit`` clipped to the RPC deadline and, while draining, to ``D``;
    once the plugin is torn down every bound has passed.
    """
    if torn_down(plugin):
        return -math.inf
    candidates = [
        rpc_deadline,
        getattr(plugin, "drain_deadline", None),
        None if owner is None else owner.drain_deadline,
    ]
    return min([limit, *(float(d) for d in candidates if d is not None)])


def _start(
    plugin: Any, record: Any, awaitable: Awaitable[T]
) -> asyncio.Future[T]:
    """Start a send's task under ``record`` or, with none, the registry."""
    if record is not None:
        return child(record, awaitable)
    task = asyncio.ensure_future(awaitable)
    keep = getattr(getattr(plugin, "registry", None), "keep", None)
    if callable(keep):
        keep(task)
    return task


def _consume(task: asyncio.Future[Any]) -> None:
    """Retrieve an abandoned send's outcome so none is reported unseen."""
    if not task.cancelled():
        task.exception()


async def _until(
    plugin: Any,
    record: Any,
    futures: set[asyncio.Future[Any]],
    deadline: float,
) -> None:
    """Wait for one of ``futures`` or ``deadline`` on the plugin's clock.

    The deadline is a ``clock.sleep`` in a task kept like the send's, so
    the wait starts no task outside the registry, and ``stop``, which
    cancels that task, wakes the wait.
    """
    remaining = max(0.0, deadline - plugin.clock.monotonic())
    sleeper = _start(plugin, record, plugin.clock.sleep(remaining))
    try:
        await asyncio.wait(
            {*futures, sleeper}, return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        sleeper.cancel()


def _drain_notice(plugin: Any) -> asyncio.Future[None] | None:
    """The plugin's drain announcement, while ``D`` is not yet set."""
    announced = getattr(plugin, "drain_announced", None)
    if not callable(announced):
        return None
    notice: asyncio.Future[None] = announced()
    if notice.done():
        return None
    return notice


def _leave_unsettled(
    plugin: Any, record: Any, task: asyncio.Future[Any]
) -> None:
    """Leave a cancelled send where ``stop`` reaches it until it settles.

    Under ``record.unsettled_sends``, which also holds the record; with no
    record (a refusal), under ``registry.keep``.
    """
    sends = getattr(record, "unsettled_sends", None)
    if isinstance(sends, set):
        sends.add(task)
        task.add_done_callback(sends.discard)
        return
    keep = getattr(getattr(plugin, "registry", None), "keep", None)
    if callable(keep):
        keep(task)


async def _abandon(plugin: Any, record: Any, task: asyncio.Future[Any]) -> None:
    """Cancel a failed send and give it one loop turn; never wait longer.

    It is left under its owner first, so a cancellation of the caller
    during that turn cannot orphan it; one that settles in the turn
    leaves again by its done callback.
    """
    task.cancel()
    _leave_unsettled(plugin, record, task)
    await asyncio.sleep(0)


def _withdraw_cancellation() -> None:
    """Withdraw the cancellation the current task has just absorbed."""
    current = asyncio.current_task()
    if current is not None and current.cancelling():
        current.uncancel()


async def _failed(
    plugin: Any, record: Any, task: asyncio.Future[Any]
) -> NoReturn:
    """A send whose failure is established: abandon it, raise ``TimeoutError``.

    The failure is final. A cancellation of the caller during the
    abandonment's turn is absorbed and withdrawn, so the caller still takes
    the transport-failure path and sends nothing more; the send stays where
    ``stop`` reaches it.
    """
    if not task.done():
        try:
            await _abandon(plugin, record, task)
        except asyncio.CancelledError:
            _withdraw_cancellation()
    raise TimeoutError


async def _last_attempt(plugin: Any, record: Any, send: Awaitable[T]) -> T:
    """A terminal status past its bound: one attempt that adds no wait.

    The send starts and runs to its first suspension; it succeeds only if
    it is complete then, and has otherwise failed. The caller resumes only
    after that first step, so a cancellation of the caller meanwhile does
    not change an incomplete attempt's failure; one that finds the attempt
    complete propagates.
    """
    task = _start(plugin, record, send)
    task.add_done_callback(_consume)
    try:
        await asyncio.sleep(0)
    except asyncio.CancelledError:
        if task.done():
            raise
        _withdraw_cancellation()
    except BaseException:
        if not task.done():
            await _abandon(plugin, record, task)
        raise
    if task.done():
        return task.result()
    await _failed(plugin, record, task)


# @spec ING-LIFE-051, ING-ADM-016, ING-EP-059, ING-EP-060
async def bounded_send(
    plugin: Any,
    send: Awaitable[T],
    *,
    owner: Any = None,
    record: Any = None,
    rpc_deadline: float | None = None,
    terminal: bool = False,
) -> T:
    """One send or close under ``response_send_timeout``, on either surface.

    The bound runs from the send's start and is clipped to ``rpc_deadline``
    and, while draining, to ``D`` (the plugin's, or the owner's once it
    observed the trigger); once the plugin is torn down it has passed. A
    waiting send also waits for the plugin's drain announcement, so one in
    flight when the drain begins re-clips to ``D`` at once, with or without
    an owner.

    A send whose bound has already passed does not start, unless it is
    ``terminal`` (a gRPC status; a Realtime error or ``failed`` event and
    its close): that one gets one attempt that adds no wait. A send not
    complete when its bound passes has failed, even if it completes
    before this coroutine resumes: whenever it resumes, the clock is
    checked before the send's outcome. A failed send is cancelled and
    given one loop turn, never a wait for it to settle; one still pending
    stays under ``record`` (or, with none, the registry) until it settles
    or ``stop``. Raises ``TimeoutError`` for a send past its bound; the
    send's own failure propagates.

    A cancellation of the caller while the send waits, before its bound
    is seen to pass, abandons the send the same way and propagates: the
    caller takes the shutdown path. Once the bound is seen to pass, the
    failure is established and final: a cancellation during the
    abandonment is withdrawn and ``TimeoutError`` raised all the same, so
    the caller takes the transport-failure path and sends nothing more.
    """
    clock = plugin.clock
    limit = clock.monotonic() + float(plugin.config.response_send_timeout)
    deadline = _send_bound(limit, plugin, owner, rpc_deadline)
    if clock.monotonic() >= deadline:
        if not terminal:
            if inspect.iscoroutine(send):
                send.close()
            raise TimeoutError
        return await _last_attempt(plugin, record, send)
    task = _start(plugin, record, send)
    task.add_done_callback(_consume)
    try:
        while clock.monotonic() < deadline:
            if task.done():
                return task.result()
            waits: set[asyncio.Future[Any]] = {task}
            notice = _drain_notice(plugin)
            if notice is not None:
                waits.add(notice)
            try:
                await _until(plugin, record, waits, deadline)
            finally:
                if notice is not None:
                    notice.cancel()
            deadline = _send_bound(limit, plugin, owner, rpc_deadline)
    except BaseException:
        if not task.done():
            await _abandon(plugin, record, task)
        raise
    await _failed(plugin, record, task)


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
