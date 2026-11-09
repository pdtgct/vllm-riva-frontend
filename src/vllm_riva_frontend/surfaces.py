"""What the two streaming surfaces share.

The Riva gRPC servicer and the Realtime adapter drive the same stream owner
and registry record and apply the same locale and RIFF/WAVE header rules;
these helpers keep the two projections identical. Both surfaces read client
input only through ``StreamOwner.receive``, which applies the bound of the
owner's state (and the RPC deadline set on the owner) and raises
``lease.ShutdownObserved`` once the drain's trigger reached the owner; every
terminal outcome is the owner's one selected cause.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable
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
    projected = getattr(cause, "error", None)
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
