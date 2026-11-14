"""ResilientStore's degraded mode (PRD §8): a Redis that just failed is not asked again for a
moment, so an outage costs one timeout, not one per call."""

import pytest

from relay.admission.resilient import ResilientStore
from relay.admission.store import MemoryStore, StoreUnavailable


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class DownRedis(MemoryStore):
    """Fails every call, and counts them (each one would be a socket timeout)."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def take(self, key, *, capacity, refill_per_s, amount):  # type: ignore[no-untyped-def]
        self.calls += 1
        raise StoreUnavailable("down")

    async def reserve(self, budget, res_id, *, amount, limit, ttl_s):  # type: ignore[no-untyped-def]
        self.calls += 1
        raise StoreUnavailable("down")

    async def paused_for(self, key):  # type: ignore[no-untyped-def]
        self.calls += 1
        raise StoreUnavailable("down")


async def test_after_a_failure_the_primary_is_skipped_for_a_while() -> None:
    clock, redis = Clock(), DownRedis()
    store = ResilientStore(redis, retry_after_s=2.0, clock=clock)
    assert await store.take("k", capacity=10, refill_per_s=1, amount=1) == 0  # local fallback
    assert redis.calls == 1
    await store.take("k", capacity=10, refill_per_s=1, amount=1)
    assert await store.paused_for("p") == 0
    with pytest.raises(StoreUnavailable):  # budgets still fail closed, at once
        await store.reserve("b", "r", amount=1, limit=10, ttl_s=5)
    assert redis.calls == 1  # none of those waited on Redis
    assert not store.healthy
    clock.now += 2.5
    await store.take("k", capacity=10, refill_per_s=1, amount=1)
    assert redis.calls == 2  # tried again after the pause


async def test_old_days_budgets_are_dropped() -> None:
    # Found in review: MemoryStore kept every day's budget, and every settled id in it, forever.
    clock = Clock()
    store = MemoryStore(clock=clock)
    assert await store.reserve("t:20261101", "r1", amount=1, limit=10, ttl_s=5)
    await store.settle("t:20261101", "r1", actual=1)
    clock.now += 3 * 86_400
    assert await store.reserve("t:20261104", "r2", amount=1, limit=10, ttl_s=5)
    assert set(store._budgets) == {"t:20261104"}
