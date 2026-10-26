"""A cap on retries per provider (PRD FR-4: "retries limited to <= 10% of requests per
provider per window to prevent retry storms").

A per-request rule ("retry once") is not enough on its own. If a provider starts failing
*every* call, "retry once" doubles its load at exactly the moment it is struggling. The
budget bounds that: over a sliding window, retries may be at most ``ratio`` of first attempts,
plus a small ``minimum`` so a quiet provider can still have a request or two retried.

Per replica, not shared (ADR 0008). Across N replicas the fleet's cap is N times this one,
which is still a fixed fraction of traffic, and checking it costs no round trip.
"""

import time
from collections import deque
from collections.abc import Callable

from prometheus_client import Counter

EXHAUSTED = Counter(
    "relay_retry_budget_exhausted_total",
    "Retries skipped because the provider's retry budget was spent.",
    ["provider"],
)


class RetryBudget:
    def __init__(
        self,
        *,
        ratio: float,
        minimum: int,
        window_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ratio = ratio
        self._minimum = minimum
        self._window_s = window_s
        self._clock = clock
        self._attempts: dict[str, deque[float]] = {}
        self._retries: dict[str, deque[float]] = {}

    def record_attempt(self, provider: str) -> None:
        """A first attempt (not a retry) was sent to ``provider``."""
        self._series(self._attempts, provider).append(self._clock())

    def try_spend(self, provider: str) -> bool:
        """May one more retry go to ``provider`` now? If yes, it is counted."""
        attempts = self._series(self._attempts, provider)
        retries = self._series(self._retries, provider)
        allowed = self._minimum + self._ratio * len(attempts)
        if len(retries) + 1 > allowed:
            EXHAUSTED.labels(provider=provider).inc()
            return False
        retries.append(self._clock())
        return True

    def _series(self, table: dict[str, deque[float]], provider: str) -> deque[float]:
        series = table.setdefault(provider, deque())
        cutoff = self._clock() - self._window_s
        while series and series[0] <= cutoff:
            series.popleft()
        return series
