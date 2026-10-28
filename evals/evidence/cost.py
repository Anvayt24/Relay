"""M8: what evidence costs, as a share of what the route spends (PRD §5 M8, §10.2 item 3).

Measured, not computed: real traffic goes through the in-process gateway (the simulator as
every provider, list prices from the price table), with shadow sampling on at the shipped
default rates. The cost is then read back from where Relay records it:

- **routed spend:** the request log's list cost (``requests.list_cost_nano_usd``);
- **evidence spend:** each shadow pair's ``cost_nano_usd`` (the shadow call's list cost plus
  the judge call's estimated list cost).

Both at **list** price, because free tiers make billed spend about $0 and the ratio would be
undefined (M8's definition). The traffic mix is the router's: about 37% of requests served
cheap (the simulated router's measured share, evals/router), the rest strong. The bulk judge
is the cheap model (a cheap bulk judge is the PRD's design; §10.3).

Evidence is only gathered on cheap-served requests here (``cheap_vs_strong``), so its cost is
roughly ``rate x cheap share x (strong call + judge) / routed spend``; the report shows that
estimate next to the measurement.
"""

import asyncio
from typing import Any

import httpx

from relay.api.app import create_app
from relay.config.schema import RelayConfig
from relay.evidence.store import MemoryEvidenceStore
from relay.providers.mock import MockProvider
from relay.settings import Settings
from relay.telemetry.log_writer import MemorySink

CHEAP_SHARE = 0.37  # evals/router/sim/metrics.json: the router's measured cheap share
PROMPT = "Summarize the main idea of this paragraph in one sentence: {i} " + "lorem ipsum " * 20


def _config(sample_rate: float) -> RelayConfig:
    return RelayConfig.model_validate(
        {
            "price_table_version": "m8",
            "providers": {"mock": {"kind": "mock"}},
            "models": {
                # relay.yaml's simulator prices (list, USD per 1M tokens).
                "mock/cheap": {
                    "provider": "mock",
                    "provider_model": "mock-cheap-v1",
                    "tier": "cheap",
                    "price_in": 0.10,
                    "price_out": 0.40,
                },
                "mock/strong": {
                    "provider": "mock",
                    "provider_model": "mock-strong-v1",
                    "tier": "strong",
                    "price_in": 1.00,
                    "price_out": 4.00,
                },
            },
            "routes": {
                "default": {
                    "aliases": {"relay/cheap": "mock/cheap", "relay/strong": "mock/strong"},
                    "evidence": {
                        "judge": "mock/cheap",
                        "sample_rate": sample_rate,
                        "sample_rate_decided": sample_rate,
                        "shadow_budget_usd_per_day": 1000,
                    },
                }
            },
            "tenants": {
                "t": {
                    "limits": {"rpm": 10**6, "tpm": 10**9, "concurrency": 1000},
                    "budget_usd_daily": 10**6,
                    "budget_mode": "soft",
                    "allowed_providers": ["mock"],
                    "evidence_sampling": "on",
                }
            },
        }
    )


async def _measure(sample_rate: float, n_requests: int, seed: int) -> dict[str, Any]:
    sink, store = MemorySink(maxlen=10**6), MemoryEvidenceStore()
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, env="test", auth_disabled=True, auth_disabled_tenant="t"
    )
    app = create_app(
        settings,
        providers={"mock": MockProvider()},
        config=_config(sample_rate),
        log_sink=sink,
        evidence_store=store,
    )
    app.state.shadow_stage.reseed(seed)
    n_cheap = round(n_requests * CHEAP_SHARE)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://m8") as c:
            for i in range(n_requests):
                model = "relay/cheap" if i < n_cheap else "relay/strong"
                body = {
                    "model": model,
                    "messages": [{"role": "user", "content": PROMPT.format(i=i)}],
                }
                r = await c.post("/v1/chat/completions", json=body)
                r.raise_for_status()
    routed = sum(row.list_cost_nano_usd or 0 for row in sink.rows)
    evidence = sum(p.cost_nano_usd for p in store.pairs)
    return {
        "sample_rate": sample_rate,
        "requests": n_requests,
        "shadow_pairs": len(store.pairs),
        "routed_list_nano_usd": routed,
        "evidence_list_nano_usd": evidence,
        "share": round(evidence / routed, 4) if routed else None,
    }


def run(*, n_requests: int = 1000, seed: int = 0) -> dict[str, Any]:
    async def both() -> list[dict[str, Any]]:
        return [
            await _measure(0.05, n_requests, seed),  # EvidenceSpec default, card undecided
            await _measure(0.01, n_requests, seed),  # once decided
        ]

    undecided, decided = asyncio.run(both())
    return {
        "target": 0.10,
        "cheap_share": CHEAP_SHARE,
        "judge": "mock/cheap",
        "undecided_rate": undecided,
        "decided_rate": decided,
        "met": (undecided["share"] or 0) <= 0.10,
    }
