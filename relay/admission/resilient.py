"""Redis, with the PRD's degradation rules when Redis is down (PRD §8, degradation matrix).

- **Rate limits (RPM, TPM):** fall back to in-process buckets at limit / replica_count. With
  N replicas each enforcing 1/N, the fleet still stays at or under the configured limit. That
  is conservative: tenants get less, never more.
- **Provider pauses:** kept in-process too, so this replica still backs off. Pauses are always
  written locally as well, so one recorded just before the outage survives it.
- **Budgets: fail closed.** ``StoreUnavailable`` propagates, and the reservation stage refuses
  ``hard``-budget tenants with 503 ``budget_unavailable``: "overspend is worse than
  unavailability". ``soft`` tenants are served.

``healthy`` is the last operation's outcome, for ``/readyz``. Redis being down never makes
the gateway "not ready": it can still serve (most tenants) in degraded mode.

**Not asking a Redis that just failed.** After a failure the primary is skipped for
``retry_after_s`` (2 s), and every operation goes straight to its degraded behavior. Without
this (found in review), a blackholed Redis cost every request one socket timeout per call:
rpm, tpm, reserve, two pause reads per candidate, settle and refund, several seconds in all.
"""

import math
import time
from collections.abc import Callable

from prometheus_client import Counter

from relay.admission.store import BudgetState, CapacityStore, MemoryStore, StoreUnavailable

FALLBACKS = Counter(
    "relay_store_fallback_total",
    "Admission-store operations served by the in-process fallback because Redis was down.",
    ["operation"],
)


class ResilientStore:
    def __init__(
        self,
        primary: CapacityStore,
        *,
        replica_count: int = 1,
        fallback: MemoryStore | None = None,
        retry_after_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.primary = primary
        self.fallback = fallback or MemoryStore()
        self._replicas = max(1, replica_count)
        self._retry_after_s = retry_after_s
        self._clock = clock
        self._skip_until = 0.0
        self.healthy = True

    async def start(self) -> None:
        await self.primary.start()

    async def close(self) -> None:
        await self.primary.close()

    def _down(self, operation: str, *, failed: bool = True) -> None:
        self.healthy = False
        if failed:
            self._skip_until = self._clock() + self._retry_after_s
        FALLBACKS.labels(operation=operation).inc()

    def _skipping(self) -> bool:
        return self._clock() < self._skip_until

    def _primary(self) -> CapacityStore:
        """The primary, unless it failed moments ago (then StoreUnavailable at once)."""
        if self._skipping():
            raise StoreUnavailable("redis failed recently; not retrying yet")
        return self.primary

    # Rate limits: degrade to a conservative local share of the limit.
    async def take(self, key: str, *, capacity: int, refill_per_s: float, amount: int) -> float:
        try:
            wait = await self._primary().take(
                key, capacity=capacity, refill_per_s=refill_per_s, amount=amount
            )
        except StoreUnavailable:
            self._down("take", failed=not self._skipping())
            return await self.fallback.take(
                key,
                capacity=max(1, math.floor(capacity / self._replicas)),
                refill_per_s=refill_per_s / self._replicas,
                amount=amount,
            )
        self.healthy = True
        return wait

    async def refund(self, key: str, *, capacity: int, amount: int) -> None:
        try:
            await self._primary().refund(key, capacity=capacity, amount=amount)
        except StoreUnavailable:
            self._down("refund", failed=not self._skipping())
            await self.fallback.refund(
                key, capacity=max(1, math.floor(capacity / self._replicas)), amount=amount
            )

    # Budgets: fail closed. No fallback on purpose.
    async def reserve(
        self, budget: str, res_id: str, *, amount: int, limit: int, ttl_s: float
    ) -> bool:
        try:
            ok = await self._primary().reserve(
                budget, res_id, amount=amount, limit=limit, ttl_s=ttl_s
            )
        except StoreUnavailable:
            self._down("reserve", failed=not self._skipping())
            raise
        self.healthy = True
        return ok

    async def settle(self, budget: str, res_id: str, *, actual: int) -> bool:
        try:
            return await self._primary().settle(budget, res_id, actual=actual)
        except StoreUnavailable:
            # The reservation expires on its own (TTL), so nothing stays held. The spend is not
            # recorded, though: a bounded under-count, logged by the caller.
            self._down("settle", failed=not self._skipping())
            raise

    async def budget_state(self, budget: str) -> BudgetState:
        return await self.primary.budget_state(budget)

    # Pauses: always local, plus shared when possible.
    async def pause(self, key: str, ttl_s: float) -> None:
        await self.fallback.pause(key, ttl_s)
        try:
            await self._primary().pause(key, ttl_s)
        except StoreUnavailable:
            self._down("pause", failed=not self._skipping())

    async def paused_for(self, key: str) -> float:
        local = await self.fallback.paused_for(key)
        try:
            shared = await self._primary().paused_for(key)
        except StoreUnavailable:
            self._down("paused_for", failed=not self._skipping())
            return local
        return max(local, shared)
