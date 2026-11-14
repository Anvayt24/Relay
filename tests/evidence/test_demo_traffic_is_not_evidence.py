"""A tenant with evidence_sampling off is never shadow-sampled, even on a route whose evidence
engine is on and sampling everything (public input is adversarial)."""

import copy

import httpx

from relay.api.app import create_app
from relay.config.schema import RelayConfig
from relay.evidence.store import MemoryEvidenceStore
from relay.providers.mock import MockProvider

from ..conftest import TEST_CONFIG  # type: ignore[import-not-found]
from ..conftest import test_settings as relay_settings  # type: ignore[import-not-found]


async def _pairs_after(evidence_sampling: str) -> int:
    data = copy.deepcopy(TEST_CONFIG.model_dump(mode="json"))
    data["routes"]["default"]["evidence"] = {
        "judge": "mock/cheap",
        "sample_rate": 1.0,
        "sample_rate_decided": 1.0,
    }
    data["tenants"]["test"]["evidence_sampling"] = evidence_sampling
    # Another tenant has it on, so the shadow stage exists: the point is the *tenant's* switch.
    data["tenants"]["limited"]["evidence_sampling"] = "on"
    store = MemoryEvidenceStore()
    app = create_app(
        relay_settings(),
        providers={"mock": MockProvider()},
        config=RelayConfig.model_validate(data),
        evidence_store=store,
    )
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://r") as c:
            for i in range(10):
                body = {
                    "model": "relay/cheap",
                    "messages": [{"role": "user", "content": f"question {i}"}],
                }
                assert (await c.post("/v1/chat/completions", json=body)).status_code == 200
        await app.state.shadow_stage.drain()
    return len(store.pairs)


async def test_a_tenant_with_evidence_off_produces_no_pairs() -> None:
    assert await _pairs_after("on") > 0  # the control: the same traffic is sampled when on
    assert await _pairs_after("off") == 0
