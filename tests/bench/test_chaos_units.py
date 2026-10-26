"""The chaos harness's own arithmetic and clients, without starting any process."""

import math

import httpx
import pytest

from bench.chaos import clients
from bench.chaos.metrics import (
    Result,
    burstiness,
    mean_and_spread,
    percentile,
    retry_amplification,
    summarize,
    wrong_rate,
)
from bench.chaos.runner import SCENARIOS, load, relay_config, sim_scenario
from relay.config.schema import RelayConfig
from relay.simulator.loader import scenario_from_dict


def test_percentile_is_nearest_rank_and_never_invents_a_value() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile([1, 2, 3, 4], 100) == 4
    assert percentile([5], 99) == 5
    assert math.isnan(percentile([], 50))


def test_summary_counts_only_successes_for_latency() -> None:
    results = [
        Result(True, 1.0),
        Result(True, 3.0, degraded=True),
        Result(False, 30.0, status=503),
    ]
    s = summarize(results)
    assert s["success_rate"] == pytest.approx(2 / 3)
    assert s["degraded"] == 1
    assert s["p95_s"] == 3.0  # the failure's 30 s wait is not a "latency"


def test_amplification_and_burstiness() -> None:
    assert retry_amplification(250, 100) == 2.5
    # 100 ms buckets 0..4. The busiest holds 8 calls; the offered load is 2 per bucket.
    assert burstiness({0: 2, 1: 8, 2: 1}, start_bucket=0, end_bucket=5, offered_per_bucket=2) == 4
    assert math.isnan(burstiness({}, start_bucket=0, end_bucket=0, offered_per_bucket=0))


def test_mean_and_spread_ignores_nan_and_handles_one_run() -> None:
    assert mean_and_spread([1.0, 3.0, math.nan]) == (2.0, pytest.approx(math.sqrt(2)))
    assert mean_and_spread([4.0]) == (4.0, 0.0)
    assert wrong_rate([True, True, False, False]) == 0.5


def test_every_scenario_builds_valid_simulator_and_relay_configs() -> None:
    for path in SCENARIOS.glob("*.yaml"):
        if path.stem == "base":
            continue
        for smoke in (False, True):
            scn = load(path.stem, smoke=smoke)
            for side in ("primary", "fallback"):
                scenario_from_dict(sim_scenario(scn, side))  # validates probabilities, quality
            RelayConfig.model_validate(relay_config(scn, 1, 2, "safe"))


def test_smoke_compresses_incident_times_with_the_duration() -> None:
    full, smoke = load("hard_outage"), load("hard_outage", smoke=True)
    inc_full = sim_scenario(full, "primary")["incidents"][0]
    inc_smoke = sim_scenario(smoke, "primary")["incidents"][0]
    assert inc_smoke["start_s"] / smoke.duration_s == pytest.approx(
        inc_full["start_s"] / full.duration_s
    )


async def test_naive_client_retries_five_times_without_jitter_then_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        if request.url.host == "primary":
            return httpx.Response(500, json={"error": {}})
        return httpx.Response(200, json={"id": "chatcmpl-sim-7-abc", "choices": [{}]})

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(clients.asyncio, "sleep", fake_sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        r = await clients.naive_retry_client(
            http, "http://primary", "http://backup", "p", "f", False, 0.0
        )
    assert r.ok and r.served_by == "fallback" and r.index == 7
    assert hosts == ["primary"] * 5 + ["backup"]
    assert sleeps == [0.5, 1.0, 2.0, 4.0]  # exactly the schedule: no jitter
