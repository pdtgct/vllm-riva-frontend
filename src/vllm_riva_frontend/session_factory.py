"""Discover the model plugin's session factory without importing a host.

The frontend does not construct engine sessions. A model plugin registers
one callable under ``vllm_riva.session_factories``. That callable receives
the core ``EngineClient`` and returns an object with ``async open``. When
the factory publishes ``served_cadences``, the configured streaming
cadence must be a member of that lookahead set.
"""

from __future__ import annotations

from importlib import metadata as importlib_metadata
from typing import cast

from vllm_riva_frontend.lease import SessionFactory

SESSION_FACTORY_GROUP = "vllm_riva.session_factories"


def load_session_factory(
    engine_client: object,
    *,
    name: str | None = None,
) -> SessionFactory:
    """Load the one session factory the model plugin registered.

    ``name`` selects a single entry point when more than one model plugin
    is installed. With no name, exactly one registration is required.
    """
    entries = _session_factory_entries()
    if name is not None:
        entries = tuple(entry for entry in entries if entry.name == name)
        if not entries:
            raise LookupError(
                f"no {SESSION_FACTORY_GROUP} entry point named {name!r}"
            )
    if not entries:
        raise LookupError(
            f"no {SESSION_FACTORY_GROUP} entry point is installed; "
            "the model plugin must supply the core session factory"
        )
    if len(entries) > 1:
        found = ", ".join(sorted(entry.name for entry in entries))
        hint = (
            "set session_factory to one entry-point name"
            if name is None
            else "duplicate entry-point names"
        )
        raise LookupError(
            f"multiple {SESSION_FACTORY_GROUP} entry points ({found}); {hint}"
        )
    builder = entries[0].load()
    if not callable(builder):
        raise TypeError(
            f"{SESSION_FACTORY_GROUP} entry point {entries[0].name!r} "
            "must be callable"
        )
    factory = builder(engine_client)
    if not callable(getattr(factory, "open", None)):
        raise TypeError("session factory must expose async open")
    return cast(SessionFactory, factory)


def require_served_cadence(factory: object, cadence: str) -> None:
    """Reject a streaming cadence outside the factory's lookahead set.

    Factories that do not publish ``served_cadences`` accept the configured
    cadence as-is. A published set is authoritative: the frontend does not
    substitute a hardcoded lookahead.
    """
    served = getattr(factory, "served_cadences", None)
    if served is None:
        return
    if isinstance(served, str) or not isinstance(
        served, (set, frozenset, list, tuple)
    ):
        raise TypeError(
            "session factory served_cadences must be a collection of "
            "cadence strings"
        )
    labels = set(served)
    if not labels or not all(isinstance(item, str) and item for item in labels):
        raise TypeError(
            "session factory served_cadences must be non-empty cadence strings"
        )
    if cadence not in labels:
        rendered = ", ".join(sorted(labels))
        raise ValueError(
            f"streaming_cadence {cadence} is not in the served lookahead "
            f"set: {rendered}"
        )


def _session_factory_entries() -> tuple[importlib_metadata.EntryPoint, ...]:
    """Return every installed session-factory entry point."""
    return tuple(importlib_metadata.entry_points(group=SESSION_FACTORY_GROUP))
