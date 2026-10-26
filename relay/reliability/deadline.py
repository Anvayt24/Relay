"""Request deadlines (PRD FR-5; ADR 0008).

A request's deadline is fixed when it arrives (tenant ``deadlines``, by priority) and stored on
the context as a ``time.monotonic()`` value. A wall clock can jump (NTP, DST), and a deadline
that jumps backwards never fires.

Two rules come out of it:
- **Every wait is clamped to what is left:** a first-token wait, a JSON call, a backoff
  sleep. No single step can overrun the request.
- **Don't start what can't finish:** an attempt with less than ``min_attempt_s`` left would
  only add load to a provider and still fail the client, so the dispatcher stops instead.

The deadline stops applying at the first byte of a stream. From there, only the idle timeout
bounds it (ADR 0008).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Deadline:
    at: float  # monotonic seconds

    def remaining(self, now: float) -> float:
        return max(0.0, self.at - now)

    def allows(self, now: float, needed_s: float) -> bool:
        """True if at least ``needed_s`` is left. ``needed_s`` = 0 still requires *some* time."""
        left = self.at - now
        return left > 0 and left >= needed_s

    def clamp(self, now: float, wait_s: float) -> float:
        """``wait_s``, shortened so it ends no later than the deadline."""
        return min(wait_s, self.remaining(now))
