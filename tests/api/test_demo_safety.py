"""Request limits, the per-IP limit and the global daily spend cap, end to end.

Every limit is off unless configured, so each test turns exactly one on and checks both sides
of the line.
"""

import copy
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from relay.admission.store import BudgetState, CapacityStore, MemoryStore, StoreUnavailable
from relay.api.app import create_app
from relay.auth.store import MemoryKeyStore
from relay.config.schema import RelayConfig
from relay.providers.mock import MockProvider
from relay.providers.registry import NO_KEYS
from relay.schemas.chat import ChatRequest, ChatResponse

from ..conftest import TEST_CONFIG  # type: ignore[import-not-found]
from ..conftest import test_settings as relay_settings  # type: ignore[import-not-found]

CHAT = "/v1/chat/completions"


def _body(content: str = "hi", **extra: Any) -> dict[str, Any]:
    return {"model": "relay/cheap", "messages": [{"role": "user", "content": content}], **extra}


def _config(
    *,
    limits: dict[str, Any] | None = None,
    request_limits: dict[str, Any] | None = None,
    global_budget: str | None = None,
    budget_usd_daily: str | None = None,
) -> RelayConfig:
    data = copy.deepcopy(TEST_CONFIG.model_dump(mode="json"))
    tenant = data["tenants"]["test"]
    tenant["limits"].update(limits or {})
    if request_limits is not None:
        tenant["request_limits"] = request_limits
    if budget_usd_daily is not None:
        tenant["budget_usd_daily"] = budget_usd_daily
        tenant["budget_mode"] = "hard"
    data["global_budget_usd_daily"] = global_budget
    return RelayConfig.model_validate(data)


class Recording(MockProvider):
    """The mock, remembering what it was asked."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[ChatRequest] = []

    async def complete(self, req: ChatRequest) -> ChatResponse:
        self.requests.append(req)
        return await super().complete(req)


@asynccontextmanager
async def serving(
    config: RelayConfig,
    *,
    provider: MockProvider | None = None,
    store: CapacityStore | None = None,
    client: tuple[str, int] = ("203.0.113.9", 5000),
    **settings: Any,
) -> AsyncIterator[tuple[httpx.AsyncClient, FastAPI]]:
    app = create_app(
        relay_settings(**settings),
        providers={"mock": provider or MockProvider()},
        config=config,
        capacity_store=store or MemoryStore(),
    )
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, client=client)
        async with httpx.AsyncClient(transport=transport, base_url="http://relay.test") as c:
            yield c, app


def _all_budgets(store: MemoryStore) -> dict[str, BudgetState]:
    # A test may read the store's own books.
    return {
        key: BudgetState(b.committed, sum(a for a, _ in b.outstanding.values()))
        for key, b in store._budgets.items()
    }


# ------------------------------------------------------------------ request limits
async def test_too_many_messages_is_a_400_naming_messages() -> None:
    config = _config(request_limits={"max_messages": 3})
    async with serving(config) as (c, _):
        many = {"model": "relay/cheap", "messages": [{"role": "user", "content": "x"}] * 4}
        refused = await c.post(CHAT, json=many)
        allowed = await c.post(CHAT, json={**many, "messages": many["messages"][:3]})
    assert refused.status_code == 400
    error = refused.json()["error"]
    assert (error["code"], error["param"]) == ("invalid_request", "messages")
    assert "3" in error["message"]
    assert allowed.status_code == 200


async def test_max_tokens_above_the_cap_is_refused_and_unset_gets_the_cap() -> None:
    provider = Recording()
    config = _config(request_limits={"max_tokens": 50})
    async with serving(config, provider=provider) as (c, _):
        over = await c.post(CHAT, json=_body(max_tokens=51))
        at_cap = await c.post(CHAT, json=_body(max_tokens=50))
        unset = await c.post(CHAT, json=_body())
    assert over.status_code == 400
    assert over.json()["error"]["param"] == "max_tokens"
    assert at_cap.status_code == unset.status_code == 200
    assert [r.max_tokens for r in provider.requests] == [50, 50]  # the cap is also the default


async def test_a_tenant_with_no_request_limits_is_unchanged() -> None:
    provider = Recording()
    async with serving(_config(), provider=provider) as (c, _):
        r = await c.post(CHAT, json=_body(max_tokens=100000))
    assert r.status_code == 200 and provider.requests[0].max_tokens == 100000


# ------------------------------------------------------------------ per-IP limit
async def test_one_address_is_limited_and_another_is_not() -> None:
    config = _config(limits={"per_ip_rpm": 2})
    async with serving(config, client=("203.0.113.9", 1)) as (first, app):
        assert [(await first.post(CHAT, json=_body())).status_code for _ in range(2)] == [200, 200]
        limited = await first.post(CHAT, json=_body())
        other = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("198.51.100.7", 1)),
            base_url="http://relay.test",
        )
        async with other:
            fresh = await other.post(CHAT, json=_body())
    assert limited.status_code == 429
    assert limited.json()["error"]["code"] == "client_rate_limited"
    assert float(limited.headers["retry-after"]) > 0
    assert limited.headers["x-should-retry"] == "true"  # waiting helps
    assert fresh.status_code == 200


async def test_x_forwarded_for_counts_only_behind_a_trusted_proxy() -> None:
    config = _config(limits={"per_ip_rpm": 1})
    spoof = {"X-Forwarded-For": "198.51.100.7"}
    # Not trusted: the header is ignored, so two "different" clients are one address.
    async with serving(config, client=("127.0.0.1", 1)) as (c, _):
        await c.post(CHAT, json=_body(), headers=spoof)
        second = await c.post(CHAT, json=_body(), headers={"X-Forwarded-For": "198.51.100.8"})
    assert second.status_code == 429
    # Trusted: each forwarded address gets its own allowance.
    async with serving(config, client=("127.0.0.1", 1), trusted_proxies="127.0.0.1") as (c, _):
        one = await c.post(CHAT, json=_body(), headers=spoof)
        two = await c.post(CHAT, json=_body(), headers={"X-Forwarded-For": "198.51.100.8"})
        again = await c.post(CHAT, json=_body(), headers=spoof)
    assert (one.status_code, two.status_code, again.status_code) == (200, 200, 429)


async def test_no_per_ip_limit_means_none() -> None:
    async with serving(_config()) as (c, _):
        codes = [(await c.post(CHAT, json=_body())).status_code for _ in range(5)]
    assert codes == [200] * 5


# ------------------------------------------------------------------ global daily cap
async def test_the_global_cap_refuses_and_gives_the_tenants_hold_back() -> None:
    store = MemoryStore()
    # One request's worst-case hold is a few thousand nano-USD: more than this whole cap.
    config = _config(global_budget="0.000001", budget_usd_daily="100")
    async with serving(config, store=store) as (c, _):
        r = await c.post(CHAT, json=_body())
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "budget_exceeded"
    assert "all tenants" in r.json()["error"]["message"]
    assert float(r.headers["retry-after"]) > 0
    # Nothing may stay held against the tenant, or a cap that refuses would starve it.
    for key, state in _all_budgets(store).items():
        assert (state.outstanding, state.committed) == (0, 0), key


async def test_a_request_under_the_cap_is_charged_to_both_budgets() -> None:
    store = MemoryStore()
    config = _config(global_budget="10", budget_usd_daily="100")
    async with serving(config, store=store) as (c, _):
        assert (await c.post(CHAT, json=_body())).status_code == 200
    states = _all_budgets(store)
    tenant_keys = [k for k in states if k.startswith("test:")]
    global_keys = [k for k in states if k.startswith("global:")]
    assert len(tenant_keys) == len(global_keys) == 1
    assert states[tenant_keys[0]].committed > 0
    assert states[global_keys[0]].committed == states[tenant_keys[0]].committed
    assert states[global_keys[0]].outstanding == states[tenant_keys[0]].outstanding == 0


async def test_no_global_cap_means_no_global_budget_is_touched() -> None:
    store = MemoryStore()
    async with serving(_config(), store=store) as (c, _):
        await c.post(CHAT, json=_body())
    assert not [k for k in _all_budgets(store) if k.startswith("global:")]


async def test_when_the_budget_store_is_down_the_global_cap_fails_closed() -> None:
    class GlobalDown(MemoryStore):
        async def reserve(self, budget: str, res_id: str, **kw: Any) -> bool:
            if budget.startswith("global:"):
                raise StoreUnavailable("down")
            return await super().reserve(budget, res_id, **kw)

    store = GlobalDown()
    config = _config(global_budget="10")  # the tenant is soft: it would be served unchecked
    async with serving(config, store=store) as (c, _):
        r = await c.post(CHAT, json=_body())
    assert r.status_code == 503 and r.json()["error"]["code"] == "budget_unavailable"
    assert all(s.outstanding == 0 for s in _all_budgets(store).values())


# ------------------------------------------------------------------ prod hides the docs
@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
async def test_api_docs_are_served_in_dev_and_hidden_in_prod(path: str) -> None:
    dev = create_app(relay_settings(), providers={"mock": MockProvider()})
    prod = create_app(
        relay_settings(env="prod", auth_disabled=False),
        provider_keys=NO_KEYS,
        key_store=MemoryKeyStore({}),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=dev), base_url="http://r") as c:
        assert (await c.get(path)).status_code == 200
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=prod), base_url="http://r") as c:
        assert (await c.get(path)).status_code == 404
