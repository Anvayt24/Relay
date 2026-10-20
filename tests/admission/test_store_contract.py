"""One contract, two implementations: MemoryStore always, RedisStore under `-m integration`.

If both pass the same tests, the Lua scripts mean exactly what the Python means, and the
in-memory store (used by every unit and property test) is a faithful stand-in for production.

Time: the memory store gets a fake clock (instant, exact). Redis uses its own clock (TIME
inside the scripts), so for Redis `advance` really sleeps, and timing asserts allow slack.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

import pytest

from relay.admission.redis_store import RedisStore
from relay.admission.store import CapacityStore, MemoryStore
from relay.settings import Settings


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@dataclass
class Env:
    store: CapacityStore
    advance: Callable[[float], Awaitable[None]]
    slack: float  # timing tolerance in seconds


@pytest.fixture(params=["memory", pytest.param("redis", marks=pytest.mark.integration)])
async def env(request: pytest.FixtureRequest) -> AsyncIterator[Env]:
    if request.param == "memory":
        clock = FakeClock()

        async def advance(seconds: float) -> None:
            clock.t += seconds

        yield Env(MemoryStore(clock), advance, 1e-9)
        return
    url = Settings().redis_url
    if not url:
        pytest.fail("RELAY_REDIS_URL must be set for integration tests")
    # A random prefix per test: tests never see each other's keys, or production's.
    store = RedisStore(url, prefix=f"test:{uuid.uuid4().hex[:8]}:")
    yield Env(store, asyncio.sleep, 0.2)
    await store.close()


# ---------------------------------------------------------------- token buckets
async def test_bucket_takes_up_to_capacity_then_says_how_long(env: Env) -> None:
    for _ in range(5):
        assert await env.store.take("b", capacity=5, refill_per_s=10, amount=1) == 0
    wait = await env.store.take("b", capacity=5, refill_per_s=10, amount=1)
    assert wait == pytest.approx(0.1, abs=env.slack)  # 1 token at 10/s


async def test_a_refused_take_takes_nothing(env: Env) -> None:
    assert await env.store.take("b", capacity=10, refill_per_s=1, amount=8) == 0
    assert await env.store.take("b", capacity=10, refill_per_s=1, amount=5) > 0
    assert await env.store.take("b", capacity=10, refill_per_s=1, amount=2) == 0  # 2 still there


async def test_bucket_refills_over_time(env: Env) -> None:
    await env.store.take("b", capacity=4, refill_per_s=20, amount=4)
    await env.advance(0.25)  # +5 tokens, capped at 4
    assert await env.store.take("b", capacity=4, refill_per_s=20, amount=4) == 0


async def test_refund_returns_tokens_up_to_capacity(env: Env) -> None:
    await env.store.take("b", capacity=10, refill_per_s=0.001, amount=10)
    await env.store.refund("b", capacity=10, amount=4)
    assert await env.store.take("b", capacity=10, refill_per_s=0.001, amount=4) == 0
    await env.store.refund("b", capacity=10, amount=100)  # capped: can't exceed capacity
    assert await env.store.take("b", capacity=10, refill_per_s=0.001, amount=11) > 0


# ---------------------------------------------------------------- budgets
async def test_reserve_respects_the_limit_exactly(env: Env) -> None:
    assert await env.store.reserve("t:1", "a", amount=60, limit=100, ttl_s=60)
    assert await env.store.reserve("t:1", "b", amount=40, limit=100, ttl_s=60)  # = limit: ok
    assert not await env.store.reserve("t:1", "c", amount=1, limit=100, ttl_s=60)
    state = await env.store.budget_state("t:1")
    assert (state.committed, state.outstanding) == (0, 100)


async def test_settle_releases_and_commits_the_actual(env: Env) -> None:
    await env.store.reserve("t:1", "a", amount=60, limit=100, ttl_s=60)
    assert await env.store.settle("t:1", "a", actual=25)
    state = await env.store.budget_state("t:1")
    assert (state.committed, state.outstanding) == (25, 0)
    # The 35 that wasn't spent is available again.
    assert await env.store.reserve("t:1", "b", amount=75, limit=100, ttl_s=60)


async def test_settle_is_idempotent(env: Env) -> None:
    await env.store.reserve("t:1", "a", amount=50, limit=100, ttl_s=60)
    assert await env.store.settle("t:1", "a", actual=30)
    assert not await env.store.settle("t:1", "a", actual=30)  # a duplicate finalize
    assert (await env.store.budget_state("t:1")).committed == 30  # charged once


async def test_release_is_settle_with_zero(env: Env) -> None:
    await env.store.reserve("t:1", "a", amount=50, limit=100, ttl_s=60)
    await env.store.settle("t:1", "a", actual=0)
    state = await env.store.budget_state("t:1")
    assert (state.committed, state.outstanding) == (0, 0)


async def test_expired_reservations_free_their_money(env: Env) -> None:
    # The replica holding "a" died: after its TTL the money comes back by itself.
    await env.store.reserve("t:1", "a", amount=90, limit=100, ttl_s=0.2)
    assert not await env.store.reserve("t:1", "b", amount=20, limit=100, ttl_s=60)
    await env.advance(0.3)
    assert await env.store.reserve("t:1", "b", amount=20, limit=100, ttl_s=60)


async def test_late_settle_after_expiry_still_records_spend(env: Env) -> None:
    await env.store.reserve("t:1", "a", amount=50, limit=100, ttl_s=0.2)
    await env.advance(0.3)
    assert await env.store.settle("t:1", "a", actual=40)  # the money was really spent
    assert (await env.store.budget_state("t:1")).committed == 40


async def test_settle_without_reservation_commits(env: Env) -> None:
    # Soft-budget tenants over budget are served without a reservation; spend still counts.
    assert await env.store.settle("t:1", "never-reserved", actual=7)
    assert (await env.store.budget_state("t:1")).committed == 7


async def test_budgets_are_independent(env: Env) -> None:
    await env.store.reserve("a:1", "x", amount=100, limit=100, ttl_s=60)
    assert await env.store.reserve("b:1", "x", amount=100, limit=100, ttl_s=60)


# ---------------------------------------------------------------- pauses
async def test_pause_counts_down_and_only_extends(env: Env) -> None:
    assert await env.store.paused_for("p") == 0
    await env.store.pause("p", 5.0)
    await env.store.pause("p", 1.0)  # shorter: must not cut the 5 s pause
    assert await env.store.paused_for("p") == pytest.approx(5.0, abs=env.slack)
    await env.store.pause("p", 8.0)  # longer: extends
    assert await env.store.paused_for("p") == pytest.approx(8.0, abs=env.slack)


async def test_pause_expires(env: Env) -> None:
    await env.store.pause("p", 0.2)
    await env.advance(0.3)
    assert await env.store.paused_for("p") == 0
