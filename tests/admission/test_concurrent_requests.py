"""FR-4's acceptance under real concurrency: many requests at once, through the whole pipeline,
against a simulator that injects faults, with clients disconnecting at random.

Afterwards, whatever happened to each request:
- no reservation is left outstanding (every one was settled or released);
- no concurrency slot is leaked, and nobody is left waiting;
- committed spend ≤ budget + one request's worth of estimation error. That overshoot bound
  is FR-4's "final overshoot is bounded by token-estimation error".

The interleaving differs on every run (that's the point), so the assertions are invariants,
not exact counts. Seeds make the *choices* reproducible; the scheduling stays real.
"""

import asyncio
import random
from decimal import Decimal
from typing import Any

import httpx
import pytest

from relay.admission.store import MemoryStore
from relay.api.app import create_app
from relay.config.tenants import TenantLimits
from relay.providers.mock import MockProvider
from relay.simulator.engine import Simulator
from relay.simulator.quality import QualitySpec
from relay.simulator.scenario import ErrorMix, LatencyModel, ModelProfile, OutputLength, Scenario

from ..conftest import TEST_CONFIG  # type: ignore[import-not-found]
from ..conftest import test_settings as relay_settings  # type: ignore[import-not-found]

BUDGET_NANO = 400_000  # tight enough that some requests are refused for budget, not most


def _scenario() -> Scenario:
    profile = ModelProfile(
        latency=LatencyModel(ttft_median_s=0.005, ttft_p95_s=0.02, tokens_per_s=2000),
        length=OutputLength(median_tokens=4, p95_tokens=10),
        errors=ErrorMix(server_error=0.1, stream_broken=0.15, overloaded=0.05),
        quality=QualitySpec(base=1.0),
    )
    return Scenario(models={"mock-cheap-v1": profile, "mock-strong-v1": profile})


@pytest.mark.parametrize("seed", [0, 1, 2])
async def test_nothing_leaks_under_concurrent_chaos(seed: int) -> None:
    rng = random.Random(seed)
    tenant = TEST_CONFIG.tenants["test"].model_copy(
        update={
            "limits": TenantLimits(rpm=100_000, tpm=10_000_000, concurrency=4),
            "budget_usd_daily": Decimal(BUDGET_NANO) / 10**9,
            "budget_mode": "hard",
        }
    )
    config = TEST_CONFIG.model_copy(update={"tenants": {**TEST_CONFIG.tenants, "test": tenant}})
    store = MemoryStore()
    provider = MockProvider(simulator=Simulator.from_scenario(_scenario(), seed=seed))
    app = create_app(
        relay_settings(queue_max_wait_interactive_s=1.0, queue_max_wait_batch_s=1.0),
        providers={"mock": provider},
        config=config,
        capacity_store=store,
    )

    async def one(client: httpx.AsyncClient, i: int) -> None:
        body: dict[str, Any] = {
            "model": rng.choice(["relay/cheap", "relay/strong"]),
            "messages": [{"role": "user", "content": "hi " * rng.randint(1, 30)}],
            "max_tokens": rng.randint(5, 20),
            "stream": rng.random() < 0.5,
        }
        headers = {"X-Relay-Priority": rng.choice(["interactive", "batch"])}
        await client.post("/v1/chat/completions", json=body, headers=headers)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://relay.test") as client:
            tasks = [asyncio.create_task(one(client, i)) for i in range(80)]
            # Random clients hang up mid-request: the cancellation runs through the pipeline
            # exactly as a disconnect would.
            await asyncio.sleep(0.01)
            for task in rng.sample(tasks, 15):
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    queue = app.state.admission_queue
    assert queue.active("test") == 0, "a concurrency slot leaked"
    assert queue.waiting("test") == 0
    [budget] = list(store._budgets)
    state = await store.budget_state(budget)
    assert state.outstanding == 0, "a reservation was never settled or released"
    # The worst single request: ~100 prompt + 20 output tokens at mock-strong's $1/$4 per 1M.
    max_one_request = 100 * 1000 + 20 * 4000
    assert state.committed <= BUDGET_NANO + max_one_request


@pytest.mark.parametrize("seed", [0, 1, 2])
async def test_nothing_leaks_with_fallbacks_and_unequal_prices(seed: int) -> None:
    """Phase 7 version: a failing primary falls back to a pricier model, so reservations must
    be sized for the worst hop and still settle exactly (FR-4 + FR-5)."""
    from ..reliability.conftest import FALLBACK_CONFIG  # type: ignore[import-not-found]

    rng = random.Random(seed)
    tenant = FALLBACK_CONFIG.tenants["strict"].model_copy(
        update={
            "limits": TenantLimits(rpm=100_000, tpm=10_000_000, concurrency=4),
            "budget_usd_daily": Decimal(BUDGET_NANO) / 10**9,
            "budget_mode": "hard",
            "route": "deep",
        }
    )
    config = FALLBACK_CONFIG.model_copy(update={"tenants": {"strict": tenant}})
    store = MemoryStore()
    providers = {
        name: MockProvider(simulator=Simulator.from_scenario(_scenario(), seed=seed + i))
        for i, name in enumerate(["primary", "backup", "big", "capped"])
    }
    app = create_app(
        relay_settings(
            auth_disabled_tenant="strict",
            queue_max_wait_interactive_s=1.0,
            queue_max_wait_batch_s=1.0,
        ),
        providers=providers,  # type: ignore[arg-type]
        config=config,
        capacity_store=store,
    )

    async def one(client: httpx.AsyncClient) -> None:
        body: dict[str, Any] = {
            "model": "relay/auto",
            "messages": [{"role": "user", "content": "hi " * rng.randint(1, 30)}],
            "max_tokens": rng.randint(5, 20),
            "stream": rng.random() < 0.5,
        }
        await client.post("/v1/chat/completions", json=body)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://relay.test") as client:
            tasks = [asyncio.create_task(one(client)) for _ in range(80)]
            await asyncio.sleep(0.01)
            for task in rng.sample(tasks, 15):
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    queue = app.state.admission_queue
    assert queue.active("strict") == 0 and queue.waiting("strict") == 0
    [budget] = list(store._budgets)
    state = await store.budget_state(budget)
    assert state.outstanding == 0, "a reservation was never settled or released"
    # Worst hop: big/model at $2/$8 per 1M with ~100 prompt + 20 output tokens.
    assert state.committed <= BUDGET_NANO + 100 * 2000 + 20 * 8000
