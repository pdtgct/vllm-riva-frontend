"""The ingress cap: one bounded count of transport owners, with a file sub-cap.

The cap counts active owners of ``StreamingRecognize``, ``Recognize``, the
Riva Realtime WebSocket and the file route together. A handler asks it once,
before it reads its first message or body, and either gets an ``Admission``
to hold until its last cleanup step or a refusal it answers at once.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

from vllm_riva_frontend.errors import FrontendError


class OwnerKind(StrEnum):
    """The four kinds of transport owner the cap counts."""

    stream = "stream"
    realtime = "realtime"
    recognize = "recognize"
    file_route = "file_route"


#: The owner kinds that also count against the file sub-cap.
FILE_KINDS = frozenset({OwnerKind.recognize, OwnerKind.file_route})


class Admission:
    """One admitted owner's place in the cap, released exactly once."""

    def __init__(self, cap: IngressCap, kind: OwnerKind) -> None:
        """Bind the admission to its cap and owner kind."""
        self._cap = cap
        self.kind = kind
        self._released = False

    @property
    def released(self) -> bool:
        """Whether ``release`` has run."""
        return self._released

    # @spec ING-ADM-013, ING-ADM-016
    def release(self) -> None:
        """Give the place back; later calls do nothing."""
        if self._released:
            return
        self._released = True
        self._cap._release(self.kind)


class IngressCap:
    """The finite ingress cap and the file-owner sub-cap inside it.

    Constructed closed: ``try_admit`` refuses until ``open`` is called, which
    the plugin's ``start`` does as its last step; ``drain`` closes it again.
    """

    def __init__(
        self,
        *,
        max_sessions: int,
        max_file_owners: int,
        serving: Callable[[], bool] | None = None,
    ) -> None:
        """Size the cap; ``serving`` is an extra predicate the plugin passes.

        ``serving`` reports whether core has not started draining, so an
        owner arriving between core setting ``state.draining`` and the
        plugin's ``drain`` closing the cap is refused too.
        """
        if type(max_sessions) is not int or max_sessions <= 0:
            raise ValueError("max_sessions must be a positive integer")
        if type(max_file_owners) is not int or max_file_owners < 0:
            raise ValueError("max_file_owners must be a non-negative integer")
        if max_file_owners >= max_sessions:
            raise ValueError("max_file_owners must be below max_sessions")
        self.max_sessions = max_sessions
        self.max_file_owners = max_file_owners
        self._serving = serving
        self._open = False
        self._count = 0
        self._file_count = 0

    @property
    def is_open(self) -> bool:
        """Whether admission is open."""
        return self._open

    @property
    def count(self) -> int:
        """Active owners of every kind."""
        return self._count

    @property
    def file_count(self) -> int:
        """Active file owners (``Recognize`` and the file route)."""
        return self._file_count

    # @spec ING-EP-035
    def open(self) -> None:
        """Open admission."""
        self._open = True

    def close(self) -> None:
        """Close admission; owners already admitted continue."""
        self._open = False

    def _serving_now(self) -> bool:
        return self._open and (self._serving is None or self._serving())

    # @spec ING-ADM-017, ING-ADM-011, ING-ADM-012, ING-ADM-010
    def try_admit(self, kind: OwnerKind | str) -> Admission | None:
        """Atomically check serving state and count, and admit one owner.

        Returns ``None`` when admission is closed or the cap (or, for a file
        owner, the sub-cap) is full; ``refusal(kind)`` names the answer. The
        check and the increment run without a suspension point, so two
        handlers never both take the last place. Nothing else is claimed:
        no residency, engine capacity or queue position.
        """
        owner_kind = OwnerKind(kind)
        if not self._serving_now() or self._count >= self.max_sessions:
            return None
        is_file = owner_kind in FILE_KINDS
        if is_file and self._file_count >= self.max_file_owners:
            return None
        self._count += 1
        if is_file:
            self._file_count += 1
        return Admission(self, owner_kind)

    # @spec ING-ADM-011
    def refusal(self, kind: OwnerKind | str) -> FrontendError:
        """Return the error a refused handler answers.

        ``service_unavailable`` while admission is closed or core is
        draining, otherwise retryable ``busy`` with ``authority: load_shed``.
        """
        OwnerKind(kind)
        if not self._serving_now():
            return FrontendError(
                "service_unavailable", "the frontend is not serving"
            )
        return FrontendError(
            "busy", "the ingress cap is full", authority="load_shed"
        )

    def _release(self, kind: OwnerKind) -> None:
        self._count -= 1
        if kind in FILE_KINDS:
            self._file_count -= 1
