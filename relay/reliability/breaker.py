"""Circuit breakers, one per (provider, model), per replica (PRD FR-5; ADR 0009).

A failing upstream costs every request that tries it a timeout or an error before it can fall
back. A breaker remembers recent failures and, past a threshold, stops calling that model
for a while, so requests go straight to a fallback, and the struggling provider gets a rest
instead of a constant stream of requests it will fail.

**States:**

    closed ──(≥ 50% of ≥ 10 counted calls failed in 30 s)──▶ open
    open ──(30 s pass)──▶ half-open
    half-open ──(3 probes succeed)──▶ closed
    half-open ──(any probe fails)──▶ open

- **closed:** every call is allowed and counted in a rolling window of one-second buckets.
- **open:** no call is allowed. ``retry_after_s()`` says when probing starts.
- **half-open:** exactly ``probes`` calls are allowed *at once*. Letting everything through at
  the first sign of recovery would hit a provider that may still be down with the full
  backlog.

**Permits.** ``acquire()`` returns a ``Permit`` (or None when the call isn't allowed), and the
caller hands the same permit back to ``record()`` with the outcome. That is how a half-open
breaker tells a probe from a call admitted while it was still closed, and how a probe slot is
released when a probe ends without a verdict (a 429, or a client cancel).

**What counts:** successes, plus failures in ``BREAKER_FAILURES``. Everything else (429s,
client errors) is ignored entirely, so it can neither trip the breaker nor dilute real
failures (see ``policy.py``).
"""

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

from prometheus_client import Counter, Gauge

from relay.config.schema import ReliabilityConfig
from relay.reliability.policy import BREAKER_FAILURES
from relay.schemas.errors import ErrorClass

STATE = Gauge(
    "relay_breaker_state",
    "Circuit breaker state: 0 closed, 1 open, 2 half-open.",
    ["provider", "model"],
)
TRANSITIONS = Counter(
    "relay_breaker_transitions_total",
    "Circuit breaker state changes.",
    ["provider", "model", "to"],
)


class BreakerState(IntEnum):
    CLOSED = 0
    OPEN = 1
    HALF_OPEN = 2


@dataclass(frozen=True)
class Permit:
    provider: str
    model: str
    probe: bool  # issued in half-open: its outcome decides whether the breaker closes


@dataclass
class _Bucket:
    second: int
    calls: int = 0
    failures: int = 0


class CircuitBreaker:
    def __init__(
        self,
        provider: str,
        model: str,
        config: ReliabilityConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.provider = provider
        self.model = model
        self._cfg = config
        self._clock = clock
        self._state = BreakerState.CLOSED
        self._opened_at = 0.0
        self._buckets: deque[_Bucket] = deque()
        self._probes_in_flight = 0
        self._probes_ok = 0
        STATE.labels(provider=provider, model=model).set(BreakerState.CLOSED)

    @property
    def state(self) -> BreakerState:
        # open → half-open happens lazily, when someone looks: no timer task needed.
        if (
            self._state is BreakerState.OPEN
            and self._clock() - self._opened_at >= self._cfg.breaker_open_s
        ):
            self._move(BreakerState.HALF_OPEN)
        return self._state

    def retry_after_s(self) -> float:
        """Seconds until an open breaker starts probing (0 if not open)."""
        if self.state is not BreakerState.OPEN:
            return 0.0
        return max(0.0, self._opened_at + self._cfg.breaker_open_s - self._clock())

    def acquire(self) -> Permit | None:
        state = self.state
        if state is BreakerState.CLOSED:
            return Permit(self.provider, self.model, probe=False)
        # Probes already in flight plus probes already succeeded: never more than `probes`
        # in one half-open round.
        if (
            state is BreakerState.HALF_OPEN
            and self._probes_in_flight + self._probes_ok < self._cfg.breaker_half_open_probes
        ):
            self._probes_in_flight += 1
            return Permit(self.provider, self.model, probe=True)
        return None

    def record(self, permit: Permit, error_class: ErrorClass | None) -> None:
        """The call made with ``permit`` ended. ``error_class`` None = success."""
        counted = error_class is None or error_class in BREAKER_FAILURES
        failed = error_class is not None and counted
        if permit.probe:
            self._record_probe(counted=counted, failed=failed)
            return
        # A call that started while closed but ends while open or half-open: its verdict is
        # stale (the breaker already decided), so it is dropped.
        if not counted or self._state is not BreakerState.CLOSED:
            return
        bucket = self._bucket()
        bucket.calls += 1
        if failed:
            bucket.failures += 1
        # Checked after *every* counted call, successes included: the call that brings the
        # window up to min_calls may be a success, with plenty of failures already in it.
        self._maybe_open()

    def release(self, permit: Permit) -> None:
        """The permit was not used (the call never happened): give back its probe slot."""
        if permit.probe:
            self._probes_in_flight = max(0, self._probes_in_flight - 1)

    def _record_probe(self, *, counted: bool, failed: bool) -> None:
        self._probes_in_flight = max(0, self._probes_in_flight - 1)
        if self._state is not BreakerState.HALF_OPEN or not counted:
            return  # no verdict: the slot is simply freed for another probe
        if failed:
            self._open()
            return
        self._probes_ok += 1
        if self._probes_ok >= self._cfg.breaker_half_open_probes:
            self._buckets.clear()
            self._move(BreakerState.CLOSED)

    def _maybe_open(self) -> None:
        calls = sum(b.calls for b in self._buckets)
        failures = sum(b.failures for b in self._buckets)
        if calls >= self._cfg.breaker_min_calls and failures >= (
            self._cfg.breaker_failure_ratio * calls
        ):
            self._open()

    def _open(self) -> None:
        self._opened_at = self._clock()
        self._buckets.clear()
        self._move(BreakerState.OPEN)

    def _move(self, to: BreakerState) -> None:
        self._state = to
        self._probes_in_flight = 0
        self._probes_ok = 0
        STATE.labels(provider=self.provider, model=self.model).set(to)
        TRANSITIONS.labels(provider=self.provider, model=self.model, to=to.name.lower()).inc()

    def _bucket(self) -> _Bucket:
        now = int(self._clock())
        # Drop buckets that fell out of the rolling window.
        while self._buckets and self._buckets[0].second <= now - self._cfg.breaker_window_s:
            self._buckets.popleft()
        if not self._buckets or self._buckets[-1].second != now:
            self._buckets.append(_Bucket(second=now))
        return self._buckets[-1]


class BreakerRegistry:
    """One breaker per (provider, model), created on first use."""

    def __init__(
        self, config: ReliabilityConfig, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._cfg = config
        self._clock = clock
        self._breakers: dict[tuple[str, str], CircuitBreaker] = {}

    def get(self, provider: str, model: str) -> CircuitBreaker:
        key = (provider, model)
        breaker = self._breakers.get(key)
        if breaker is None:
            breaker = self._breakers[key] = CircuitBreaker(provider, model, self._cfg, self._clock)
        return breaker

    def record(self, permit: Permit, error_class: ErrorClass | None) -> None:
        self.get(permit.provider, permit.model).record(permit, error_class)
