"""Compressed chaos runs asserting the documented behavior (PRD §11.3). Real subprocesses
(simulators + Relay), on localhost, no network. ``make chaos-smoke``; excluded from
``make check``.

Thresholds are deliberately loose: event interleaving is not deterministic (PRD §11.1), so
these assert behavior ("the outage is served via the fallback"), not exact numbers.
"""

from typing import Any

import pytest

from bench.chaos.runner import run_scenario

pytestmark = pytest.mark.chaos


def one(result: dict[str, Any], client: str) -> dict[str, Any]:
    [run] = result["clients"][client]
    assert isinstance(run, dict)
    return run


async def test_outage_is_absorbed_by_the_fallback() -> None:
    r = await run_scenario("hard_outage", ("relay", "direct"), 1, smoke=True)
    relay, direct = one(r, "relay"), one(r, "direct")
    assert relay["success_rate"] >= 0.97
    assert direct["success_rate"] < relay["success_rate"] - 0.1  # no gateway: the outage shows
    assert relay["silent_truncations"] == 0


async def test_429_storm_is_not_amplified_into_a_retry_spike() -> None:
    r = await run_scenario("storm_429", ("relay", "naive_retry"), 1, smoke=True)
    relay, naive = one(r, "relay"), one(r, "naive_retry")
    assert relay["success_rate"] >= 0.97
    assert relay["retry_amplification"] < 1.5 < naive["retry_amplification"]
    # The pause state stops Relay calling a limited primary at all (FR-4), so the primary
    # sees fewer synchronized spikes than from clients retrying on a fixed schedule. (Poisson
    # arrivals alone peak at about 3x the mean in a 100 ms bucket, so compare, don't cap.)
    assert relay["primary_burstiness"] < naive["primary_burstiness"]


async def test_sick_primary_trips_the_breaker_and_midstream_drops_are_never_silent() -> None:
    r = await run_scenario("midstream_drop", ("relay", "direct"), 1, smoke=True)
    relay, direct = one(r, "relay"), one(r, "direct")
    # Half the primary's streams break after the first byte. Those can't fail over (FR-5): a
    # direct client loses them. Relay loses a few too, until its breaker opens (stream_broken
    # counts, ADR 0009), then serves everything from the fallback.
    assert direct["success_rate"] < 0.7
    assert relay["success_rate"] > direct["success_rate"]
    assert relay["primary_calls"] >= 10  # the breaker needs its minimum calls to open
    assert relay["fallback_calls"] > 0
    assert relay["silent_truncations"] == 0  # a broken stream always says so


async def test_cold_start_strict_refuses_to_fail_over() -> None:
    r = await run_scenario("cold_start_strict", ("relay",), 1, smoke=True)
    relay = one(r, "relay")
    assert relay["fallback_calls"] == 0  # undecided edge + strict: never used
    assert relay["incident_success_rate"] < 0.2
    assert relay["success_rate"] < 0.9  # the outage is visible to clients, by design


async def test_headline_beats_the_naive_client() -> None:
    r = await run_scenario("m2_headline", ("relay", "naive_retry"), 1, smoke=True)
    relay, naive = one(r, "relay"), one(r, "naive_retry")
    assert relay["success_rate"] >= 0.97
    assert relay["success_rate"] >= naive["success_rate"]
    assert relay["p95_s"] < naive["p95_s"]
