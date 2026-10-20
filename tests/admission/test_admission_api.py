"""Admission, reservations, budgets and pauses through the HTTP API (PRD FR-4, FR-9)."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from relay.admission.redis_store import RedisStore
from relay.admission.resilient import ResilientStore
from relay.admission.store import MemoryStore
from relay.config.schema import RelayConfig
from relay.config.tenants import TenantLimits
from relay.providers.mock import MockBehavior, MockProvider
from relay.schemas.errors import ErrorClass
from relay.telemetry.log_writer import MemorySink

from ..conftest import TEST_CONFIG  # type: ignore[import-not-found]

CHAT = "/v1/chat/completions"


def _body(model: str = "relay/cheap", **extra: Any) -> dict[str, Any]:
    return {"model": model, "messages": [{"role": "user", "content": "hello there"}], **extra}


def _config(limits: dict[str, int] | None = None, **tenant: Any) -> RelayConfig:
    """The test config with the `test` tenant's limits/budget changed."""
    base = TEST_CONFIG.tenants["test"]
    update: dict[str, Any] = dict(tenant)
    if limits:
        update["limits"] = TenantLimits(**{**base.limits.model_dump(), **limits})
    tenants = {**TEST_CONFIG.tenants, "test": base.model_copy(update=update)}
    return TEST_CONFIG.model_copy(update={"tenants": tenants})


@asynccontextmanager
async def serving(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://relay.test") as c:
            yield c


# ---------------------------------------------------------------- rate limits
async def test_rpm_limit_is_a_retryable_429(make_app: Callable[..., FastAPI]) -> None:
    app = make_app(MockProvider(), config=_config({"rpm": 2}))
    async with serving(app) as client:
        codes = [(await client.post(CHAT, json=_body())).status_code for _ in range(2)]
        r = await client.post(CHAT, json=_body())
    assert codes == [200, 200]
    assert r.status_code == 429
    assert r.json()["error"]["code"] == ErrorClass.TENANT_RATE_LIMITED
    # Relay's own load refusal: the client *should* retry, after Retry-After. 2 per minute
    # refills one request every 30 s.
    assert r.headers["x-should-retry"] == "true"
    assert r.headers["retry-after"] == "30"


async def test_request_bigger_than_tpm_is_a_400_not_a_429(make_app: Callable[..., FastAPI]) -> None:
    app = make_app(MockProvider(), config=_config({"tpm": 500}))
    async with serving(app) as client:
        r = await client.post(CHAT, json=_body(max_tokens=1000))
    # Waiting can never help a request larger than the whole bucket: fix the request.
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "max_tokens"


async def test_tpm_limit(make_app: Callable[..., FastAPI]) -> None:
    # "hello there" estimates to 9 prompt tokens; with max_tokens=40 a request reserves 49.
    app = make_app(MockProvider(), config=_config({"tpm": 100}))
    async with serving(app) as client:
        first = await client.post(CHAT, json=_body(max_tokens=40))
        second = await client.post(CHAT, json=_body(max_tokens=40))
        third = await client.post(CHAT, json=_body(max_tokens=40))
    # The first used ~5 real tokens; the unused ~44 were refunded, so the second fits too.
    assert (first.status_code, second.status_code) == (200, 200)
    assert third.status_code == 200 or third.json()["error"]["code"] == "tenant_rate_limited"


# ---------------------------------------------------------------- concurrency + queue
async def test_queue_timeout_when_slots_are_busy(make_app: Callable[..., FastAPI]) -> None:
    import asyncio

    slow = MockProvider(MockBehavior(latency_s=0.3))
    app = make_app(slow, config=_config({"concurrency": 1}), queue_max_wait_interactive_s=0.05)
    async with serving(app) as client:
        first = asyncio.create_task(client.post(CHAT, json=_body()))
        await asyncio.sleep(0.05)
        second = await client.post(CHAT, json=_body())
        assert (await first).status_code == 200
    assert second.status_code == 429
    assert second.json()["error"]["code"] == ErrorClass.QUEUE_TIMEOUT
    assert second.headers["x-should-retry"] == "true"


async def test_batch_waits_longer_and_gets_served(make_app: Callable[..., FastAPI]) -> None:
    import asyncio

    sink = MemorySink()
    slow = MockProvider(MockBehavior(latency_s=0.2))
    app = make_app(slow, sink=sink, config=_config({"concurrency": 1}), queue_max_wait_batch_s=5)
    async with serving(app) as client:
        first = asyncio.create_task(client.post(CHAT, json=_body()))
        await asyncio.sleep(0.05)
        second = await client.post(CHAT, json=_body(), headers={"X-Relay-Priority": "batch"})
        await first
    assert second.status_code == 200
    queued = [row for row in sink.rows if row.priority == "batch"]
    assert queued[0].queue_wait_ms >= 100  # it waited for the first request's slot


async def test_queue_full(make_app: Callable[..., FastAPI]) -> None:
    import asyncio

    slow = MockProvider(MockBehavior(latency_s=0.3))
    app = make_app(slow, config=_config({"concurrency": 1}), queue_max_depth=0)
    async with serving(app) as client:
        first = asyncio.create_task(client.post(CHAT, json=_body()))
        await asyncio.sleep(0.05)
        second = await client.post(CHAT, json=_body())
        await first
    assert second.json()["error"]["code"] == ErrorClass.QUEUE_FULL


# ---------------------------------------------------------------- budgets
# mock/cheap at $0.10/$0.40 per 1M. With max_tokens=5, a request reserves its worst case:
# 9 (est. prompt) * 0.10 + 5 * 0.40 = 2.9 $/1M = 2,900 nano. It really costs
# 2 * 0.10 + 3 * 0.40 = 1.4 $/1M = 1,400 nano ("hello there" -> "mock: hello there").
BUDGET = Decimal("0.00001")  # 10,000 nano-USD


async def test_hard_budget_stops_before_overspending(make_app: Callable[..., FastAPI]) -> None:
    store = MemoryStore()
    app = make_app(
        MockProvider(), store=store, config=_config(budget_usd_daily=BUDGET, budget_mode="hard")
    )
    async with serving(app) as client:
        codes = [(await client.post(CHAT, json=_body(max_tokens=5))).status_code for _ in range(8)]
        refused = await client.post(CHAT, json=_body(max_tokens=5))
    # 6 x 1,400 = 8,400 committed; a 7th would need 8,400 + 2,900 > 10,000: refused, even
    # though its *real* cost (1,400) would have fit. Reservations are deliberately worst-case.
    assert codes == [200] * 6 + [429] * 2
    assert refused.json()["error"]["code"] == ErrorClass.BUDGET_EXCEEDED
    assert refused.headers["x-should-retry"] == "false"  # retrying won't help until midnight
    assert 0 < int(refused.headers["retry-after"]) <= 86_400
    [budget] = [k for k in store._budgets]  # one tenant-day
    state = await store.budget_state(budget)
    assert (state.committed, state.outstanding) == (8_400, 0)  # all settled, nothing held


async def test_soft_budget_serves_and_reports(make_app: Callable[..., FastAPI]) -> None:
    store = MemoryStore()
    app = make_app(
        MockProvider(), store=store, config=_config(budget_usd_daily=BUDGET, budget_mode="soft")
    )
    async with serving(app) as client:
        codes = [(await client.post(CHAT, json=_body(max_tokens=5))).status_code for _ in range(10)]
    assert codes == [200] * 10
    [budget] = list(store._budgets)
    assert (await store.budget_state(budget)).committed == 14_000  # over budget, but counted


async def test_failed_request_releases_its_reservation(make_app: Callable[..., FastAPI]) -> None:
    store = MemoryStore()
    failing = MockProvider(MockBehavior(error=ErrorClass.SERVER_ERROR))
    app = make_app(
        failing, store=store, config=_config(budget_usd_daily=BUDGET, budget_mode="hard")
    )
    async with serving(app) as client:
        r = await client.post(CHAT, json=_body(max_tokens=5))
    assert r.status_code == 502
    [budget] = list(store._budgets)
    state = await store.budget_state(budget)
    assert (state.committed, state.outstanding) == (0, 0)  # nothing billed, nothing held


async def test_broken_stream_settles_on_what_was_forwarded(
    make_app: Callable[..., FastAPI],
) -> None:
    store = MemoryStore()
    broken = MockProvider(MockBehavior(fail_after_chunks=2))
    app = make_app(broken, store=store, config=_config(budget_usd_daily=BUDGET, budget_mode="hard"))
    async with serving(app) as client:
        r = await client.post(CHAT, json=_body(max_tokens=5, stream=True))
    assert "stream broken" in r.text
    [budget] = list(store._budgets)
    state = await store.budget_state(budget)
    # No usage report arrived, but "mock:" and " hello" were forwarded and would be billed:
    # settled at the estimate instead of left outstanding. Forwarded text is counted with the
    # same byte estimator as reservations: "mock:" (5 bytes) = 2, " hello" (6 bytes) = 2.
    assert state.outstanding == 0
    assert state.committed == 9 * 100 + 4 * 400


# ---------------------------------------------------------------- provider pause
async def test_upstream_429_pauses_the_scope_for_everyone(make_app: Callable[..., FastAPI]) -> None:
    limited = MockProvider(MockBehavior(error=ErrorClass.RATE_LIMITED, retry_after_s=5))
    app = make_app(limited)
    async with serving(app) as client:
        first = await client.post(CHAT, json=_body())
        second = await client.post(CHAT, json=_body())
        other_model = await client.post(CHAT, json=_body("relay/strong"))
    assert first.json()["error"]["code"] == ErrorClass.RATE_LIMITED  # the provider's own 429
    assert second.json()["error"]["code"] == ErrorClass.PROVIDER_PAUSED
    assert second.headers["x-should-retry"] == "true"
    assert 5 <= int(second.headers["retry-after"]) <= 6  # 5 s x jitter in [1.0, 1.2]
    # mock's limit_scope is "model": only mock-cheap-v1 is paused. mock/strong still went
    # upstream (and got its own 429): 3 requests, 2 upstream calls.
    assert other_model.json()["error"]["code"] == ErrorClass.RATE_LIMITED
    assert limited.calls == 2


async def test_account_scope_pauses_every_model(make_app: Callable[..., FastAPI]) -> None:
    providers = {
        **TEST_CONFIG.providers,
        "mock": TEST_CONFIG.providers["mock"].model_copy(update={"limit_scope": "account"}),
    }
    config = TEST_CONFIG.model_copy(update={"providers": providers})
    limited = MockProvider(MockBehavior(error=ErrorClass.RATE_LIMITED, retry_after_s=5))
    async with serving(make_app(limited, config=config)) as client:
        await client.post(CHAT, json=_body())
        other_model = await client.post(CHAT, json=_body("relay/strong"))
    assert other_model.json()["error"]["code"] == ErrorClass.PROVIDER_PAUSED
    assert limited.calls == 1


# ---------------------------------------------------------------- Redis down
@pytest.fixture
def redis_down() -> ResilientStore:
    # Port 1: nothing listens. Every Redis call fails fast; the store degrades.
    return ResilientStore(RedisStore("redis://127.0.0.1:1/0", timeout_s=0.2))


async def test_redis_down_hard_budget_fails_closed(
    make_app: Callable[..., FastAPI], redis_down: ResilientStore
) -> None:
    app = make_app(MockProvider(), store=redis_down, config=_config(budget_mode="hard"))
    async with serving(app) as client:
        r = await client.post(CHAT, json=_body())
        ready = await client.get("/readyz")
    # "Overspend is worse than unavailability" (PRD §8).
    assert r.status_code == 503
    assert r.json()["error"]["code"] == ErrorClass.BUDGET_UNAVAILABLE
    assert ready.status_code == 200  # degraded, still serving others: never "not ready"
    assert ready.json()["dependencies"]["redis"] == "degraded"


async def test_redis_down_soft_budget_and_local_rate_limits(
    make_app: Callable[..., FastAPI], redis_down: ResilientStore
) -> None:
    app = make_app(MockProvider(), store=redis_down, config=_config({"rpm": 1}, budget_mode="soft"))
    async with serving(app) as client:
        first = await client.post(CHAT, json=_body())
        second = await client.post(CHAT, json=_body())
    assert first.status_code == 200  # soft budget: served without the shared store
    assert second.json()["error"]["code"] == ErrorClass.TENANT_RATE_LIMITED  # still limited
