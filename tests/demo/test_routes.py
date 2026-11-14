"""The demo's HTTP surface: keyless chat for configured tenants only, the chaos
controls, and the evidence view. Everything is off unless configured."""

import copy
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from relay.admission.store import MemoryStore
from relay.api.app import create_app
from relay.auth.store import MemoryKeyStore
from relay.config.schema import RelayConfig
from relay.providers.mock import MockProvider
from relay.simulator.engine import Simulator
from relay.simulator.scenario import PRESETS

from ..conftest import TEST_CONFIG  # type: ignore[import-not-found]
from ..conftest import test_settings as relay_settings  # type: ignore[import-not-found]

BODY = {"model": "relay/cheap", "messages": [{"role": "user", "content": "hi"}]}
ROOT = Path(__file__).resolve().parents[2]


def _config(**demo_limits: Any) -> RelayConfig:
    data = copy.deepcopy(TEST_CONFIG.model_dump(mode="json"))
    template = data["tenants"]["test"]
    data["tenants"]["demo-sim"] = {
        **template,
        "request_limits": demo_limits.get("request_limits", {}),
        "limits": {**template["limits"], **demo_limits.get("limits", {})},
    }
    data["tenants"]["demo-chaos"] = {**template, "fallback_policy": "allow_degraded"}
    return RelayConfig.model_validate(data)


@asynccontextmanager
async def serving(
    *,
    config: RelayConfig | None = None,
    client: tuple[str, int] = ("203.0.113.9", 1),
    **settings: Any,
) -> AsyncIterator[httpx.AsyncClient]:
    provider = MockProvider(simulator=Simulator.from_scenario(PRESETS["ideal"]))
    # An explicit in-memory store: a RELAY_REDIS_URL in the environment must not leak in and
    # carry rate-limit state from one test (or one run) to the next.
    app = create_app(
        relay_settings(auth_disabled=False, key_pepper="p", **settings),
        providers={"mock": provider},
        config=config or _config(),
        key_store=MemoryKeyStore({}),
        capacity_store=MemoryStore(),
    )
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, client=client)
        async with httpx.AsyncClient(transport=transport, base_url="http://relay.test") as c:
            yield c


# ------------------------------------------------------------------ info
async def test_info_says_what_is_switched_on() -> None:
    async with serving() as c:
        off = (await c.get("/v1/demo/info")).json()
    assert off["chat"] == {"real": False, "sim": False} and off["chaos"] is False
    async with serving(demo_sim_tenant="demo-sim", sim_scenario="ideal", demo_chaos="on") as c:
        on = (await c.get("/v1/demo/info")).json()
    assert on["chat"]["sim"] is True and on["chaos"] is True


# ------------------------------------------------------------------ keyless chat
async def test_chat_is_404_unless_a_tenant_is_configured_for_it() -> None:
    async with serving() as c:
        for mode in ("sim", "real"):
            assert (await c.post(f"/v1/demo/chat?mode={mode}", json=BODY)).status_code == 404


async def test_a_configured_tenant_answers_without_a_key_with_the_decision_fields() -> None:
    async with serving(demo_sim_tenant="demo-sim") as c:
        r = await c.post("/v1/demo/chat?mode=sim", json=BODY)
        keyed = await c.post("/v1/chat/completions", json=BODY)  # the ordinary route
    assert r.status_code == 200 and r.json()["choices"]
    assert r.headers["x-relay-fallback-hops"] == "0" and "x-relay-strong-cost-usd" in r.headers
    assert keyed.status_code == 401  # the demo did not weaken the real endpoint


async def test_the_keyless_endpoint_cannot_choose_its_tenant() -> None:
    async with serving(demo_sim_tenant="demo-sim") as c:
        assert (await c.post("/v1/demo/chat?mode=test", json=BODY)).status_code == 400
        assert (await c.post("/v1/demo/chat?mode=chaos", json=BODY)).status_code == 404
        wrong = await c.post(
            "/v1/demo/chat?mode=chaos", json=BODY, headers={"X-Relay-Demo-Secret": "guess"}
        )
        assert wrong.status_code == 404


async def test_the_demo_tenants_limits_still_apply() -> None:
    config = _config(request_limits={"max_messages": 1}, limits={"per_ip_rpm": 2})
    async with serving(config=config, demo_sim_tenant="demo-sim") as c:
        two = {"model": "relay/cheap", "messages": [{"role": "user", "content": "x"}] * 2}
        assert (await c.post("/v1/demo/chat?mode=sim", json=two)).status_code == 400
        # The 400 above did not spend any of the address's allowance of 2.
        assert (await c.post("/v1/demo/chat?mode=sim", json=BODY)).status_code == 200
        assert (await c.post("/v1/demo/chat?mode=sim", json=BODY)).status_code == 200
        assert (await c.post("/v1/demo/chat?mode=sim", json=BODY)).status_code == 429


# ------------------------------------------------------------------ chaos
async def test_chaos_is_off_by_default_and_needs_a_simulator() -> None:
    async with serving() as c:
        assert (await c.post("/v1/demo/chaos", json={"action": "kill"})).status_code == 404
        assert (await c.get("/v1/demo/chaos/stats")).status_code == 404


async def test_a_chaos_action_breaks_the_primary_and_the_stats_report_both_clients() -> None:
    async with serving(demo_chaos="on", sim_scenario="ideal") as c:
        r = await c.post("/v1/demo/chaos", json={"action": "kill", "duration_s": 30})
        assert r.status_code == 200
        stats = (await c.get("/v1/demo/chaos/stats")).json()
    assert set(stats) >= {"relay", "naive", "enabled"} and stats["enabled"] is True
    assert {"n", "success_rate", "p50_s", "p95_s"} <= set(stats["relay"])


async def test_an_unknown_chaos_action_is_a_400() -> None:
    async with serving(demo_chaos="on", sim_scenario="ideal") as c:
        r = await c.post("/v1/demo/chaos", json={"action": "explode"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"


async def test_chaos_actions_are_limited_per_address() -> None:
    async with serving(demo_chaos="on", sim_scenario="ideal") as c:
        codes = [
            (await c.post("/v1/demo/chaos", json={"action": "clear"})).status_code for _ in range(8)
        ]
    assert codes[:5] == [200] * 5 and 429 in codes[5:]


async def test_chaos_needs_the_simulator_scenario_or_it_stays_off() -> None:
    async with serving(demo_chaos="on") as c:  # no sim_scenario: a scripted echo, not a simulator
        assert (await c.get("/v1/demo/info")).json()["chaos"] is False


# ------------------------------------------------------------------ evidence
async def test_the_evidence_view_serves_cards_and_the_judges_calibration() -> None:
    async with serving() as c:
        r = await c.get("/v1/demo/evidence")
    data = r.json()
    assert r.status_code == 200
    assert data["cards"] and data["cards"][0]["state"]
    sonnet = data["calibration"]["judges"]["anthropic/sonnet"]
    assert sonnet["status"] == "calibrated" and sonnet["rubric"] == ["v2"]
    assert 0.6 <= sonnet["kappa"]["value"] <= 0.7
    assert data["calibration"]["gate"]["kappa_min"] == 0.6
    assert "canary" in data["simulated"]


async def test_the_packaged_demo_data_matches_what_the_evaluations_produced() -> None:
    # relay/demo/data/ is a reduced copy (the image carries no evals/): it must never drift.
    from scripts.sync_demo_data import build, render

    for name, value in build().items():
        packaged = (ROOT / "relay" / "demo" / "data" / name).read_text(encoding="utf-8")
        assert packaged == render(value), f"{name} is stale: run python -m scripts.sync_demo_data"
        json.loads(packaged)  # and it is strict JSON


@pytest.mark.parametrize("path", ["/v1/demo/info", "/v1/demo/evidence"])
async def test_the_read_only_views_never_leak_secrets(path: str) -> None:
    async with serving(demo_sim_tenant="demo-sim") as c:
        text = (await c.get(path)).text.lower()
    for needle in ("pepper", "admin_token", "api_key", "secret"):
        assert needle not in text


async def test_each_mode_answers_only_as_its_own_tenant() -> None:
    # Configuring the simulator tenant must not make the real-model mode answer as it too.
    async with serving(demo_sim_tenant="demo-sim") as c:
        assert (await c.post("/v1/demo/chat?mode=sim", json=BODY)).status_code == 200
        assert (await c.post("/v1/demo/chat?mode=real", json=BODY)).status_code == 404
    async with serving(demo_tenant="demo-sim") as c:
        assert (await c.post("/v1/demo/chat?mode=real", json=BODY)).status_code == 200
        assert (await c.post("/v1/demo/chat?mode=sim", json=BODY)).status_code == 404


async def test_relays_side_of_the_comparison_really_goes_through_the_gateway() -> None:
    # The panel's own traffic enters as the chaos tenant, through the whole app, and succeeds.
    import asyncio

    async with serving(demo_chaos="on", sim_scenario="ideal") as c:
        await c.get("/v1/demo/chaos/stats")  # opens the panel: traffic starts
        await asyncio.sleep(0.8)
        stats = (await c.get("/v1/demo/chaos/stats")).json()
    assert stats["relay"]["n"] > 0 and stats["relay"]["success_rate"] == 1.0
