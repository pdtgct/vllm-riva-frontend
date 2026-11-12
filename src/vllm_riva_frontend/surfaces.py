"""What the two streaming surfaces share.

The Riva gRPC servicer and the Realtime adapter drive the same stream owner
and registry record and apply the same locale and RIFF/WAVE header rules;
these helpers keep the two projections identical. Both surfaces read client
input only through ``StreamOwner.receive``, which applies the bound of the
owner's state (and the RPC deadline set on the owner) and raises
``lease.ShutdownObserved`` once the drain's trigger reached the owner; every
terminal outcome is the owner's one selected cause. Both send through
``bounded_send`` (refusals that create no owner too) and wait for cleanup
through ``settle_shielded``.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import math
from collections.abc import Awaitable, Iterable
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
    """A task owned by the stream's record (the settlement awaits it)."""
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


def _running(task: asyncio.Future[Any] | None) -> bool:
    return task is not None and not task.done()


# @spec ING-LIFE-063, ING-EP-106
def release_owner(registry: Any, record: Any, admitted: Any) -> None:
    """Release a handler's admission and record as its handler ends.

    The stream's one ingress-cap count goes once the owner and its cleanup
    continuation have ended: a ``record.continuation`` still running (an
    owner's cleanup that outlived the answer) releases it from its done
    callback. The record goes once, in addition, every send left unsettled
    under it has ended; a record that persists only for such sends takes no
    count. ``Admission.release`` is idempotent and ``unregister`` tolerates
    a record already removed.
    """

    def release_record(_: Any = None) -> None:
        for task in list(getattr(record, "unsettled_sends", ())):
            if _running(task):
                task.add_done_callback(release_record)
                return
        unregister(registry, record)

    def release_count(_: Any = None) -> None:
        continuation = getattr(record, "continuation", None)
        if continuation is not None and not continuation.done():
            continuation.add_done_callback(release_count)
            return
        admitted.release()
        release_record()

    release_count()


def torn_down(plugin: Any) -> bool:
    """Whether ``stop`` has run."""
    registry = getattr(plugin, "registry", None)
    return bool(getattr(plugin, "torn_down", False)) or bool(
        getattr(registry, "torn_down", False)
    )


def send_bounds_passed(plugin: Any) -> bool:
    """Whether every send bound counts as passed (ING-EP-110).

    After the plugin is torn down and, for the rest of a ``drain`` call,
    once its settlement began or it skipped its graceful steps; ``D``
    passing is read from ``drain_deadline`` and the clock.
    """
    return bool(getattr(plugin, "send_bounds_passed", False)) or torn_down(
        plugin
    )


async def join(tasks: Iterable[asyncio.Future[Any]]) -> None:
    """Wait until every task is done, whatever cancels the caller meanwhile.

    A cancellation that arrives while they run is absorbed and withdrawn,
    so the caller's record and admission outlive the cleanup they guard;
    the vehicle's settlement awaits the caller, it does not reap it.
    """
    pending = {task for task in tasks if not task.done()}
    while pending:
        try:
            await asyncio.wait(pending)
        except asyncio.CancelledError:
            _withdraw_cancellation()
        pending = {task for task in pending if not task.done()}


async def settle_shielded(
    record: Any,
    owner: Any,
    error: errors.FrontendError,
    phase: str,
) -> None:
    """``settle`` as a child of the record, joined through cancellation.

    Returns only once cleanup has run (``join``).
    """
    cleanup = child(record, settle(owner, error, phase))
    await join((cleanup,))
    if cleanup.cancelled():
        raise asyncio.CancelledError
    cleanup.result()


# ------------------------------------------------------------------- sends


def _send_bound(limit: float, plugin: Any, rpc_deadline: float | None) -> float:
    """The send's absolute bound as it stands now.

    ``limit`` clipped to the RPC deadline and, while a ``drain`` call runs
    its graceful work, to ``D``; once the send bounds pass (ING-EP-110)
    every bound has passed. Core's ``state.draining`` alone clips nothing.
    """
    if send_bounds_passed(plugin):
        return -math.inf
    candidates = [rpc_deadline, getattr(plugin, "drain_deadline", None)]
    return min([limit, *(float(d) for d in candidates if d is not None)])


def _start(record: Any, awaitable: Awaitable[T]) -> asyncio.Future[T]:
    """Start a send's task: under ``record``, or owned by its handler.

    A refusal that creates no owner (``record`` is ``None``) keeps its send
    with its own handler, which awaits it inline (ING-EP-095).
    """
    if record is not None:
        return child(record, awaitable)
    return asyncio.ensure_future(awaitable)


def _consume(task: asyncio.Future[Any]) -> None:
    """Retrieve an abandoned send's outcome so none is reported unseen."""
    if not task.cancelled():
        task.exception()


# @spec ING-EP-106
async def _until(
    plugin: Any,
    futures: set[asyncio.Future[Any]],
    deadline: float,
) -> None:
    """Wait for one of ``futures`` or ``deadline`` on the plugin's clock.

    The deadline is a timer of the clock (``clock.timer``), not a task, so
    a refusal's wait starts nothing outside the registry but its own send
    (ING-EP-106); the timer is cancelled when the wait ends.
    """
    timer = plugin.clock.timer(deadline)
    try:
        await asyncio.wait(
            {*futures, timer}, return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        timer.cancel()


def _drain_notice(plugin: Any) -> asyncio.Future[None] | None:
    """The plugin's next change of the send bounds, if one is pending."""
    announced = getattr(plugin, "drain_announced", None)
    if not callable(announced):
        return None
    notice: asyncio.Future[None] = announced()
    if notice.done():
        return None
    return notice


def _withdraw_cancellation() -> None:
    """Withdraw the cancellation the current task has just absorbed."""
    current = asyncio.current_task()
    if current is not None and current.cancelling():
        current.uncancel()


async def _await_to_end(task: asyncio.Future[Any]) -> bool:
    """Await ``task`` through any number of cancellations of the caller.

    Returns whether the caller was cancelled meanwhile; those
    cancellations are withdrawn, for the caller to re-raise.
    """
    cancelled = False
    while not task.done():
        try:
            await asyncio.wait({task})
        except asyncio.CancelledError:
            cancelled = True
            _withdraw_cancellation()
    return cancelled


# @spec ING-LIFE-061, ING-EP-094, ING-EP-095
async def _abandon(plugin: Any, record: Any, task: asyncio.Future[Any]) -> None:
    """Cancel a failed or interrupted send without waiting for it.

    An owner's send stays under ``record.unsettled_sends``, which also
    holds the record, and gets one loop turn. A refusal's send (no record)
    is registered in the owner registry as a send without an owner while
    the registry is open (the check and the registration with no
    suspension between them); once the registry is closed its handler
    awaits it itself, through any number of cancellations, and re-raises a
    cancellation only once the send has ended.
    """
    task.cancel()
    if record is not None:
        sends = getattr(record, "unsettled_sends", None)
        if isinstance(sends, set) and not task.done():
            sends.add(task)
            task.add_done_callback(sends.discard)
        await asyncio.sleep(0)
        return
    adopt = getattr(getattr(plugin, "registry", None), "adopt_send", None)
    if task.done() or (callable(adopt) and adopt(task)):
        await asyncio.sleep(0)
        return
    if await _await_to_end(task):
        raise asyncio.CancelledError


async def _failed(
    plugin: Any, record: Any, task: asyncio.Future[Any]
) -> NoReturn:
    """A send whose failure is established: abandon it, raise ``TimeoutError``.

    A send still running is abandoned (``_abandon``); one already ended is
    left as it is. For an owner's send (``record``) the failure is final: a
    cancellation of the caller during the abandonment's turn is absorbed
    and withdrawn, so the caller still takes the transport-failure path and
    sends nothing more, and the send stays where the settlement reaches it.
    For a refusal's send (no ``record``), a cancellation of its handler
    during the abandonment (its turn after registering the send, or its
    wait for the send once the registry is closed) propagates instead of
    ``TimeoutError`` (ING-EP-095).
    """
    if not task.done():
        try:
            await _abandon(plugin, record, task)
        except asyncio.CancelledError:
            if record is None:
                raise
            _withdraw_cancellation()
    raise TimeoutError


async def _last_attempt(plugin: Any, record: Any, send: Awaitable[T]) -> T:
    """A terminal send past its bound: one attempt that adds no wait.

    The send starts and runs to its first suspension, and the caller
    resumes after that one loop turn. A send complete then returns its
    result, or raises its own failure; one still running has failed and is
    abandoned (``_failed``, ``TimeoutError``). When the attempt is already
    complete, a cancellation of the caller during that turn propagates.
    When it is incomplete, the cancellation is withdrawn: for an owner's
    send (``record``) the failure stands; a refusal's handler (no
    ``record``) retains the cancellation, abandons the send (``_abandon``)
    and re-raises the cancellation once the send has been registered or,
    with the registry closed, has ended (ING-EP-095).
    """
    task = _start(record, send)
    task.add_done_callback(_consume)
    retained = False
    try:
        await asyncio.sleep(0)
    except asyncio.CancelledError:
        if task.done():
            raise
        _withdraw_cancellation()
        retained = record is None
    except BaseException:
        if not task.done():
            await _abandon(plugin, record, task)
        raise
    if task.done():
        return task.result()
    if retained:
        await _abandon(plugin, record, task)
        raise asyncio.CancelledError
    await _failed(plugin, record, task)


# @spec ING-LIFE-061, ING-ADM-016, ING-EP-095, ING-EP-110, ING-EP-106
async def bounded_send(
    plugin: Any,
    send: Awaitable[T],
    *,
    record: Any = None,
    rpc_deadline: float | None = None,
    terminal: bool = False,
) -> T:
    """One send or close under ``response_send_timeout``, on either surface.

    The bound runs from the send's start and is clipped to ``rpc_deadline``
    and, while a ``drain`` call runs its graceful work, to ``D``; once the
    send bounds pass (ING-EP-110) it has passed. A waiting send also waits
    for the plugin's drain announcement, so one in flight when the drain
    begins re-clips to ``D`` at once, with or without an owner.

    A send whose bound has already passed does not start, unless it is
    ``terminal`` (a gRPC status; a Realtime error or ``failed`` event and
    its close; a refusal's whole sequence): that one gets one attempt that
    adds no wait. A send not complete when its bound passes has failed,
    even if it completes before this coroutine resumes. The judgement runs
    in the send's own task, as its last step, when the send completes:
    whether it completed before its bound, not when this coroutine
    resumes, decides; the inline judgement here covers only a send
    cancelled before it ran, which never reached that step. A failed send
    is cancelled and abandoned (``_abandon``), never waited for to settle.
    Raises ``TimeoutError`` for a send past its bound; the send's own
    failure propagates.

    With ``record`` the send is a child of the stream's record. Without
    one (a refusal that creates no owner) the handler awaits it inline and
    owns it until it ends or is registered on abandonment (ING-EP-095).

    A cancellation of the caller while the send waits, before its bound
    is seen to pass, abandons the send the same way and propagates: the
    caller takes the shutdown path. Once the bound is seen to pass, the
    failure is established and final: a cancellation during the
    abandonment is withdrawn and ``TimeoutError`` raised all the same, so
    the caller takes the transport-failure path and sends nothing more.
    """
    clock = plugin.clock
    limit = clock.monotonic() + float(plugin.config.response_send_timeout)
    deadline = _send_bound(limit, plugin, rpc_deadline)
    if clock.monotonic() >= deadline:
        if not terminal:
            if inspect.iscoroutine(send):
                send.close()
            raise TimeoutError
        return await _last_attempt(plugin, record, send)
    # Whether the send completed before its bound passed, judged at the
    # moment it completes (in the send's own task, as its last step), not
    # when this coroutine resumes: a send complete in time is delivered
    # even if the bound passes (the clock, or the send bounds passing at
    # ``stop``), or another wake-up (the drain notice) resumes this
    # coroutine, before the send's done callbacks have run.
    in_time: list[bool] = []

    def judge() -> None:
        if not in_time:
            bound = _send_bound(limit, plugin, rpc_deadline)
            in_time.append(clock.monotonic() < bound)

    async def judged() -> T:
        try:
            return await send
        finally:
            judge()

    task = _start(record, judged())
    task.add_done_callback(_consume)
    try:
        while True:
            if task.done():
                judge()  # a send cancelled before it ran: judged now
                if in_time[0]:
                    return task.result()
                break
            if clock.monotonic() >= deadline:
                break
            waits: set[asyncio.Future[Any]] = {task}
            notice = _drain_notice(plugin)
            if notice is not None:
                waits.add(notice)
            try:
                await _until(plugin, waits, deadline)
            finally:
                if notice is not None:
                    notice.cancel()
            deadline = _send_bound(limit, plugin, rpc_deadline)
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
