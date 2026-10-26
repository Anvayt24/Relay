"""YAML scenarios, runtime incidents and upstream-load stats (Phase 8: the chaos plumbing)."""

import httpx
import pytest

from relay.providers.registry import NO_KEYS, ProviderKeys
from relay.simulator.engine import Simulator
from relay.simulator.loader import scenario_from_dict
from relay.simulator.quality import WinRate
from relay.simulator.server import create_sim_app

CHAT = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}


def scenario() -> dict:
    return {
        "models": {
            "m": {
                "latency": {"ttft_median_s": 0.1, "ttft_p95_s": 0.2, "tokens_per_s": 100},
                "errors": {"rate_limited": 0.2, "retry_after_s": 2},
                "quality": {"base": 0.95, "shifts": [[10, 0.8]]},
            },
            "cheap": {"quality": {"base": {"reference": "m", "w": 0.9}}},
        },
        "capacity": {"rpm": 60, "tpm": 1000, "scope": "model"},
        "incidents": [{"start_s": 1, "end_s": 2, "errors": {"server_error": 1.0}}],
    }


def test_scenario_loads_from_a_dict() -> None:
    s = scenario_from_dict(scenario())
    assert s.models["m"].errors.rate_limited == 0.2
    assert s.models["m"].quality.level_at(10) == 0.8
    assert isinstance(s.models["cheap"].quality.base, WinRate)
    assert s.capacity is not None and s.capacity.scope == "model"
    assert s.incidents[0].errors is not None


def test_typos_are_errors_not_silently_ignored() -> None:
    bad = scenario()
    bad["models"]["m"]["errors"] = {"rate_limitd": 0.8}
    with pytest.raises(TypeError):
        scenario_from_dict(bad)
    bad = scenario()
    bad["models"]["m"]["colour"] = "red"
    with pytest.raises(ValueError, match="unknown key"):
        scenario_from_dict(bad)


def client(sim: Simulator, *, control: bool) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=create_sim_app(sim, control=control))
    return httpx.AsyncClient(transport=transport, base_url="http://sim.test")


async def test_runtime_incident_breaks_the_model_now_and_clear_heals_it() -> None:
    sim = Simulator.from_scenario(scenario_from_dict({"models": {"m": {}}}))
    async with client(sim, control=True) as c:
        assert (await c.post("/v1/chat/completions", json=CHAT)).status_code == 200
        r = await c.post("/sim/control", json={"duration_s": 60, "errors": {"server_error": 1.0}})
        assert r.status_code == 200
        assert (await c.post("/v1/chat/completions", json=CHAT)).status_code == 500
        await c.post("/sim/clear")
        assert (await c.post("/v1/chat/completions", json=CHAT)).status_code == 200


async def test_control_endpoints_are_off_unless_asked_for() -> None:
    sim = Simulator.from_scenario(scenario_from_dict({"models": {"m": {}}}))
    async with client(sim, control=False) as c:
        assert (await c.post("/sim/control", json={"duration_s": 1})).status_code == 404


async def test_stats_count_what_reached_the_simulator_and_reset_clears_them() -> None:
    sim = Simulator.from_scenario(scenario_from_dict({"models": {"m": {}}}))
    async with client(sim, control=True) as c:
        for _ in range(3):
            await c.post("/v1/chat/completions", json=CHAT)
        stats = (await c.get("/sim/stats")).json()
        assert stats["calls"] == 3 and stats["by_model"] == {"m": 3}
        assert sum(stats["arrivals"].values()) == 3
        await c.post("/sim/reset")
        assert (await c.get("/sim/stats")).json()["calls"] == 0


def test_provider_keys_read_declared_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIM_UPSTREAM_KEY", "abc")
    assert ProviderKeys(_env_file=None).get("SIM_UPSTREAM_KEY") == "abc"
    assert ProviderKeys(_env_file=None).get("SOMETHING_ELSE") is None
    assert NO_KEYS.get("SIM_UPSTREAM_KEY") is None  # tests that must not reach providers
