"""The demo's chaos panel: break a simulated provider, and watch Relay and a naive
retry client side by side. Simulator only; bounded; stops by itself."""

import asyncio
import time

import pytest

from relay.demo.chaos import CHEAP, MAX_DURATION_S, STRONG, ChaosPanel
from relay.simulator.engine import Simulator
from relay.simulator.scenario import PRESETS


def simulator() -> Simulator:
    return Simulator.from_scenario(PRESETS["ideal"])


async def always_ok() -> tuple[bool, float]:
    return True, 0.0


def panel(sim: Simulator, **kw: object) -> ChaosPanel:
    values: dict[str, object] = {
        "send_relay": always_ok,
        "rate_hz": 200.0,
        "backoff_s": (0.0, 0.0, 0.0, 0.0),
        "idle_stop_s": 5.0,
    }
    return ChaosPanel(sim, **{**values, **kw})  # type: ignore[arg-type]


# ------------------------------------------------------------------ the controls
def test_kill_breaks_the_primary_only_and_clear_repairs_it() -> None:
    sim = simulator()
    p = panel(sim)
    p.apply("kill", 30)
    assert sim._runtime_incidents  # the primary now fails
    incident = sim._runtime_incidents[0]
    assert incident.models == (STRONG,) and incident.errors is not None
    assert incident.errors.server_error == 1.0
    p.apply("clear", 0)
    assert not sim._runtime_incidents


def test_slow_and_storm_are_incidents_on_the_primary() -> None:
    sim = simulator()
    p = panel(sim)
    p.apply("slow", 20)
    assert sim._runtime_incidents[0].latency_multiplier > 1
    p.apply("storm", 20)
    errors = sim._runtime_incidents[0].errors
    assert errors is not None and errors.rate_limited > 0.5
    assert all(i.models == (STRONG,) for i in sim._runtime_incidents)


def test_the_duration_is_capped_so_nobody_can_break_it_for_an_hour() -> None:
    sim = simulator()
    panel(sim).apply("kill", 3600)
    incident = sim._runtime_incidents[0]
    assert incident.end_s - incident.start_s == MAX_DURATION_S


def test_an_unknown_action_is_refused() -> None:
    with pytest.raises(ValueError, match="explode"):
        panel(simulator()).apply("explode", 10)


# ------------------------------------------------------------------ the comparison
async def test_a_dead_primary_costs_the_naive_client_its_backoff_but_not_relay() -> None:
    sim = simulator()
    p = panel(sim, backoff_s=(0.05, 0.05, 0.05, 0.05))
    p.apply("kill", 30)
    p.stats()  # opens the panel: traffic starts
    await asyncio.sleep(1.0)
    stats = p.stats()
    await p.stop()
    assert stats["relay"]["success_rate"] == 1.0
    assert stats["naive"]["n"] > 0
    # Four failed tries and their waits (4 x 50 ms; timers are coarse on some platforms, so
    # three quarters of it) come before the fallback answers.
    assert stats["naive"]["p50_s"] >= 0.15
    assert stats["relay"]["p50_s"] < stats["naive"]["p50_s"]


async def test_the_naive_client_ends_up_on_the_fallback_model() -> None:
    sim = simulator()
    seen: list[str] = []
    real_open = sim.open

    def spy(req):  # type: ignore[no-untyped-def]
        seen.append(req.model)
        return real_open(req)

    sim.open = spy  # type: ignore[method-assign]
    p = panel(sim)
    p.apply("kill", 30)
    p.stats()
    await asyncio.sleep(0.3)
    await p.stop()
    assert STRONG in seen and CHEAP in seen


# ------------------------------------------------------------------ bounded
async def test_traffic_is_bounded_and_stops_by_itself_when_nobody_is_watching() -> None:
    sim = simulator()
    p = panel(sim, rate_hz=100.0, idle_stop_s=0.15)
    p.stats()
    await asyncio.sleep(0.5)  # nobody polls again: the loop must stop on its own
    assert not p.running
    count = p.stats()["relay"]["n"] + p.stats()["naive"]["n"]
    await asyncio.sleep(0.2)
    assert p.stats()["relay"]["n"] + p.stats()["naive"]["n"] >= count  # (window) never grows
    await p.stop()


async def test_polling_keeps_it_running_and_stop_ends_it() -> None:
    p = panel(simulator(), idle_stop_s=0.3)
    p.stats()
    for _ in range(5):
        await asyncio.sleep(0.1)
        p.stats()
    assert p.running
    await p.stop()
    assert not p.running


async def test_only_samples_inside_the_window_are_reported() -> None:
    now = [time.monotonic()]
    p = panel(simulator(), window_s=10.0, rate_hz=1.0)
    p._clock = lambda: now[0]  # type: ignore[method-assign]
    p._record("relay", True, 0.1)
    now[0] += 11
    p._record("relay", False, 0.2)
    stats = p.stats()
    await p.stop()
    assert stats["relay"]["n"] == 1 and stats["relay"]["success_rate"] == 0.0


async def test_requests_in_flight_are_capped() -> None:
    gate = asyncio.Event()
    started = 0

    async def slow_relay() -> tuple[bool, float]:
        nonlocal started
        started += 1
        await gate.wait()
        return True, 0.0

    p = panel(simulator(), send_relay=slow_relay, rate_hz=500.0, max_in_flight=7)
    p.stats()
    await asyncio.sleep(0.3)
    assert started <= 7
    # Not just the running ones: requests that would only wait are not even created, so a
    # dead primary cannot make the panel's own backlog grow without bound.
    assert len(p._inflight) <= 7
    gate.set()
    await p.stop()
