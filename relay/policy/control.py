"""The policy control operations, as pure transitions of the pointer (ADR 0024).

    activate v   v active; the old active becomes previous; any candidate is cleared
    shadow v     v is the candidate, deciding in the background, never applied
    canary v p   v is the candidate, serving p% of requests
    promote      the candidate becomes active (a human's decision: ADR 0022)
    rollback     with a candidate: drop it (the active policy never changed);
                 without one: the previous version becomes active again

Pure functions, so every transition is tested without a database; the store applies one
inside a transaction together with its audit event.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Literal

CandidateMode = Literal["shadow", "canary"]


class ControlError(ValueError):
    """The operation makes no sense in the current state (e.g. promote with no candidate)."""


@dataclass(frozen=True)
class Pointer:
    active_version: str | None = None
    previous_version: str | None = None
    candidate_version: str | None = None
    candidate_mode: CandidateMode | None = None
    canary_pct: int = 0
    updated_at: datetime | None = None

    def stamped(self) -> "Pointer":
        return replace(self, updated_at=datetime.now(UTC))


# One operation, as a function of the current pointer (applied by the store under a lock).
type Transition = Callable[[Pointer], Pointer]


def activate(p: Pointer, version: str) -> Pointer:
    if version == p.active_version:
        raise ControlError(f"{version} is already active")
    return Pointer(active_version=version, previous_version=p.active_version).stamped()


def shadow(p: Pointer, version: str) -> Pointer:
    _not_active(p, version)
    return replace(p, candidate_version=version, candidate_mode="shadow", canary_pct=0).stamped()


def canary(p: Pointer, version: str, pct: int) -> Pointer:
    _not_active(p, version)
    if not 1 <= pct <= 100:
        raise ControlError("canary percentage must be between 1 and 100")
    return replace(p, candidate_version=version, candidate_mode="canary", canary_pct=pct).stamped()


def promote(p: Pointer) -> Pointer:
    if p.candidate_version is None:
        raise ControlError("nothing to promote: no shadow or canary is running")
    return Pointer(active_version=p.candidate_version, previous_version=p.active_version).stamped()


def rollback(p: Pointer) -> Pointer:
    if p.candidate_version is not None:
        return replace(p, candidate_version=None, candidate_mode=None, canary_pct=0).stamped()
    if p.previous_version is None:
        raise ControlError("nothing to roll back to: no previous version")
    return Pointer(active_version=p.previous_version, previous_version=None).stamped()


def _not_active(p: Pointer, version: str) -> None:
    if version == p.active_version:
        raise ControlError(f"{version} is already active; a candidate must be a different version")
