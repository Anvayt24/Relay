"""Rate limits (RPM/TPM token buckets) and the clock they run on.

Real providers return 429s because *you* pushed too hard, not at random. A simulator that
only injects random 429s cannot show that Relay's admission control (Phase 6) prevents
them, so the limits here behave like real ones:
- they refill over time;
- they are per API key, optionally per model as well;
- they answer a 429 with an exact ``retry_after``.

**Token bucket.** A bucket holds up to ``capacity`` tokens and refills at a steady rate (for
an RPM limit, ``rpm / 60`` per second). A request takes tokens; if too few are there, it is
refused, and ``retry_after`` = (missing tokens) / (refill rate) is exactly when it would fit.
Unlike a fixed "per calendar minute" window, a bucket has no burst at the window edge, and it
is what most providers describe.

**TPM is charged ``prompt_tokens + max_tokens`` up front**, because the provider cannot know
how long the reply will be until it ends, so it reserves the cap. That is why a request with
a huge ``max_tokens`` can be rate-limited while asking a one-word question. Relay's own
reservations (Phase 6, FR-4) copy the same idea.

**No locks needed.** Many requests share these buckets concurrently, but every method here is
synchronous: it never awaits. Under asyncio only one coroutine runs at a time, and it cannot
be interrupted between ``await``s, so each check-and-take happens atomically. Make one of
these methods async, and that guarantee is gone.
"""

import time
from dataclasses import dataclass
from typing import Literal, Protocol

from relay.simulator.scenario import Capacity


class Clock(Protocol):
    def now(self) -> float: ...


class MonotonicClock:
    # Monotonic, not wall time: an NTP adjustment must not refill (or drain) every bucket.
    def now(self) -> float:
        return time.monotonic()


class TokenBucket:
    def __init__(self, capacity: float, refill_per_s: float, now: float) -> None:
        self.capacity = capacity
        self.refill_per_s = refill_per_s
        self.tokens = capacity  # buckets start full
        self.updated = now

    def _refill(self, now: float) -> None:
        # Lazy refill: instead of a background task adding tokens every tick, compute how
        # many would have been added since the last look. Same result, no timers.
        elapsed = max(0.0, now - self.updated)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_s)
        self.updated = now

    def wait_for(self, n: float, now: float) -> float:
        """Seconds until ``n`` tokens are available (0 if they are now). Takes nothing.

        A request larger than the whole bucket gets a wait it can never satisfy. Real
        providers reject such a request as too large; here it simply stays rate-limited.
        """
        self._refill(now)
        missing = n - self.tokens
        return 0.0 if missing <= 0 else missing / self.refill_per_s

    def take(self, n: float, now: float) -> None:
        self._refill(now)
        self.tokens -= n

    def seconds_to_full(self) -> float:
        return (self.capacity - self.tokens) / self.refill_per_s


@dataclass(frozen=True)
class LimitState:
    """What the provider tells the client about its limits (the ``x-ratelimit-*`` headers)."""

    limit_requests: int
    limit_tokens: int
    remaining_requests: int
    remaining_tokens: int
    reset_requests_s: float  # until the bucket is full again
    reset_tokens_s: float
    retry_after_s: float  # 0 when the request was admitted
    scope: Literal["account", "model"]

    @property
    def admitted(self) -> bool:
        return self.retry_after_s == 0


class CapacityTracker:
    def __init__(self, capacity: Capacity, clock: Clock) -> None:
        self.capacity = capacity
        self.clock = clock
        # One (requests, tokens) pair of buckets per scope key, created on first use.
        self._buckets: dict[tuple[str, ...], tuple[TokenBucket, TokenBucket]] = {}

    def admit(
        self, api_key: str, model: str, prompt_tokens: int, max_tokens: int | None
    ) -> LimitState:
        """Charge one request; return the limit state, refused if ``retry_after_s > 0``.

        Both buckets are checked before either is charged. Charging RPM for a request that
        TPM then refuses would let refused requests eat the budget of admitted ones.
        """
        cap = self.capacity
        now = self.clock.now()
        requests, tokens = self._buckets_for(api_key, model, now)
        cost = prompt_tokens + (max_tokens if max_tokens is not None else cap.default_output_tokens)

        retry_after = max(requests.wait_for(1, now), tokens.wait_for(cost, now))
        if retry_after == 0:
            requests.take(1, now)
            tokens.take(cost, now)
        return LimitState(
            limit_requests=cap.rpm,
            limit_tokens=cap.tpm,
            remaining_requests=max(0, int(requests.tokens)),
            remaining_tokens=max(0, int(tokens.tokens)),
            reset_requests_s=requests.seconds_to_full(),
            reset_tokens_s=tokens.seconds_to_full(),
            retry_after_s=retry_after,
            scope=cap.scope,
        )

    def _buckets_for(self, api_key: str, model: str, now: float) -> tuple[TokenBucket, TokenBucket]:
        key = (api_key,) if self.capacity.scope == "account" else (api_key, model)
        if key not in self._buckets:
            cap = self.capacity
            self._buckets[key] = (
                TokenBucket(cap.rpm, cap.rpm / 60, now),
                TokenBucket(cap.tpm, cap.tpm / 60, now),
            )
        return self._buckets[key]
