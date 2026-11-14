"""Shared admission state: rate buckets, budget reservations and provider pauses.

One protocol, ``CapacityStore``, with three implementations:
- ``MemoryStore`` (this file): exact semantics in one process. Unit and property tests use
  it, and it is the fallback when Redis is down.
- ``RedisStore`` (``redis_store.py``): the same semantics, shared by every replica, each
  operation one atomic Lua script.
- ``ResilientStore`` (``resilient.py``): Redis, degrading to memory when Redis is unreachable.

The same contract test suite runs against Memory and Redis (``tests/admission``), which is
how we know the Lua scripts mean what the Python means.

**The operations:**
- ``take(key, capacity, refill_per_s, amount)``: a token bucket. Take ``amount`` if available
  and return 0; otherwise take nothing and return the seconds until it would be available.
- ``refund(key, capacity, amount)``: put unused tokens back (a TPM reservation that turned
  out larger than the real usage).
- ``reserve(budget, res_id, amount, limit, ttl_s)``: hold ``amount`` against a budget, but only
  if ``committed + outstanding + amount ≤ limit``. This is FR-4's invariant, checked and
  applied as one atomic step.
- ``settle(budget, res_id, actual)``: the request finished. Release its reservation and commit
  what it really cost. **Idempotent:** a second settle of the same id does nothing, so a retry
  or a duplicated finalize can never charge twice. ``actual=0`` is a release.
- ``pause(key, ttl_s)`` / ``paused_for(key)``: a shared "don't call this provider scope"
  flag. Pause only ever extends, never shortens.

**Crash safety.** A reservation carries a TTL. If the replica that made it dies before
settling, the reservation expires and its money returns to the budget, instead of leaking
forever. Expired reservations are swept at the start of every ``reserve``.

**Money is integer nano-USD** (see ``relay/config/pricing.py``). No floats near a budget.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol


class StoreUnavailable(Exception):
    """The shared store can't be reached. Budgets fail closed on this (PRD §8)."""


@dataclass(frozen=True)
class BudgetState:
    committed: int
    outstanding: int


class CapacityStore(Protocol):
    async def take(self, key: str, *, capacity: int, refill_per_s: float, amount: int) -> float: ...
    async def refund(self, key: str, *, capacity: int, amount: int) -> None: ...
    async def reserve(
        self, budget: str, res_id: str, *, amount: int, limit: int, ttl_s: float
    ) -> bool: ...
    async def settle(self, budget: str, res_id: str, *, actual: int) -> bool: ...
    async def budget_state(self, budget: str) -> BudgetState: ...
    async def pause(self, key: str, ttl_s: float) -> None: ...
    async def paused_for(self, key: str) -> float: ...
    async def start(self) -> None: ...
    async def close(self) -> None: ...


@dataclass
class _Bucket:
    tokens: float
    updated: float


@dataclass
class _Budget:
    committed: int = 0
    # res_id -> (amount, expires_at)
    outstanding: dict[str, tuple[int, float]] = field(default_factory=dict)
    settled: set[str] = field(default_factory=set)
    touched: float = 0.0  # last use, on the store's clock


# A budget unused for this long is dropped (daily keys: yesterday's is done after a day).
BUDGET_IDLE_S = 2 * 86_400


class MemoryStore:
    """In-process ``CapacityStore``. No method awaits anything, so under asyncio each one runs
    to completion without interleaving: atomic without a lock (see simulator/capacity.py)."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._budgets: dict[str, _Budget] = {}  # pruned by _budget()
        self._pauses: dict[str, float] = {}  # key -> until

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    # ---------------------------------------------------------------- rate buckets
    async def take(self, key: str, *, capacity: int, refill_per_s: float, amount: int) -> float:
        bucket = self._refilled(key, capacity, refill_per_s)
        if bucket.tokens >= amount:
            bucket.tokens -= amount
            return 0.0
        return (amount - bucket.tokens) / refill_per_s

    async def refund(self, key: str, *, capacity: int, amount: int) -> None:
        bucket = self._buckets.get(key)
        if bucket is not None:
            bucket.tokens = min(capacity, bucket.tokens + amount)

    def _refilled(self, key: str, capacity: int, refill_per_s: float) -> _Bucket:
        now = self._clock()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = self._buckets[key] = _Bucket(tokens=capacity, updated=now)
        bucket.tokens = min(capacity, bucket.tokens + (now - bucket.updated) * refill_per_s)
        bucket.updated = now
        return bucket

    # ---------------------------------------------------------------- budgets
    async def reserve(
        self, budget: str, res_id: str, *, amount: int, limit: int, ttl_s: float
    ) -> bool:
        state = self._swept(budget)
        outstanding = sum(a for a, _ in state.outstanding.values())
        if state.committed + outstanding + amount > limit:
            return False
        state.outstanding[res_id] = (amount, self._clock() + ttl_s)
        return True

    async def settle(self, budget: str, res_id: str, *, actual: int) -> bool:
        state = self._budget(budget)
        if res_id in state.settled:
            return False  # already settled: never charge twice
        state.settled.add(res_id)
        state.outstanding.pop(res_id, None)  # gone already if it expired; still commit
        state.committed += actual
        return True

    async def budget_state(self, budget: str) -> BudgetState:
        state = self._swept(budget)
        return BudgetState(state.committed, sum(a for a, _ in state.outstanding.values()))

    def _budget(self, budget: str) -> _Budget:
        """The budget's state. Budget keys are per day ("tenant:YYYYMMDD"), so yesterday's stop
        being used; when a new one appears, budgets untouched for two days are dropped (found in
        review: they, and every settled reservation id in them, were kept forever)."""
        now = self._clock()
        state = self._budgets.get(budget)
        if state is None:
            cutoff = now - BUDGET_IDLE_S
            self._budgets = {k: v for k, v in self._budgets.items() if v.touched > cutoff}
            state = self._budgets[budget] = _Budget()
        state.touched = now
        return state

    def _swept(self, budget: str) -> _Budget:
        state = self._budget(budget)
        now = self._clock()
        for res_id in [r for r, (_, exp) in state.outstanding.items() if exp <= now]:
            del state.outstanding[res_id]
        return state

    # ---------------------------------------------------------------- pauses
    async def pause(self, key: str, ttl_s: float) -> None:
        until = self._clock() + ttl_s
        self._pauses[key] = max(self._pauses.get(key, 0.0), until)

    async def paused_for(self, key: str) -> float:
        remaining = self._pauses.get(key, 0.0) - self._clock()
        return max(0.0, remaining)
