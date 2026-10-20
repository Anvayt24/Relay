from collections.abc import Callable
from typing import Any

import pytest

from relay.schemas.chat import ChatRequest
from relay.schemas.errors import ErrorClass, ProviderError
from relay.simulator.engine import CapacityExceeded, Simulator
from relay.simulator.plan import MockBehavior
from relay.simulator.quality import QualitySpec
from relay.simulator.scenario import (
    Capacity,
    ErrorMix,
    Incident,
    ModelProfile,
    OutputLength,
    Scenario,
)

MODEL = "m"


def _sim(clock: Any = None, **scenario: Any) -> Simulator:
    profile = scenario.pop(
        "profile", ModelProfile(length=OutputLength(5, 10), quality=QualitySpec(base=0.5))
    )
    return Simulator.from_scenario(
        Scenario(models={MODEL: profile}, **scenario), seed=9, clock=clock
    )


async def test_complete_and_stream_agree(make_request: Callable[..., ChatRequest]) -> None:
    sim = _sim()
    req = make_request(MODEL)
    response = await sim.open(req, index=3).complete()
    chunks = [c async for c in sim.open(req, index=3).stream()]

    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == response.choices[0].message.content
    assert response.id.startswith("chatcmpl-sim-3-")
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[-1].usage == response.usage


async def test_indexes_count_up_unless_given(make_request: Callable[..., ChatRequest]) -> None:
    sim = _sim()
    req = make_request(MODEL)
    assert [sim.open(req).plan.index for _ in range(3)] == [0, 1, 2]
    assert sim.open(req, index=42).plan.index == 42
    assert sim.open(req).plan.index == 3  # an explicit index does not advance the counter


async def test_unknown_model_is_model_not_found(make_request: Callable[..., ChatRequest]) -> None:
    with pytest.raises(ProviderError) as exc:
        _sim().open(make_request("nope"))
    assert exc.value.error_class is ErrorClass.MODEL_NOT_FOUND


async def test_stream_breaks_mid_reply(make_request: Callable[..., ChatRequest]) -> None:
    profile = ModelProfile(length=OutputLength(5, 10), errors=ErrorMix(stream_broken=1.0))
    call = _sim(profile=profile).open(make_request(MODEL))
    k = call.plan.fail_after_chunks
    assert k is not None
    received = 0
    with pytest.raises(ProviderError) as exc:
        async for chunk in call.stream():
            received += chunk.has_content
    assert exc.value.error_class is ErrorClass.STREAM_BROKEN
    assert received == k


async def test_injected_429_carries_retry_after(make_request: Callable[..., ChatRequest]) -> None:
    profile = ModelProfile(errors=ErrorMix(rate_limited=1.0, retry_after_s=4))
    with pytest.raises(ProviderError) as exc:
        await _sim(profile=profile).open(make_request(MODEL)).complete()
    assert exc.value.error_class is ErrorClass.RATE_LIMITED
    assert exc.value.signal is not None
    assert exc.value.signal.retry_after_s == 4


async def test_capacity_429_is_raised_at_open(
    make_request: Callable[..., ChatRequest], clock: Any
) -> None:
    sim = _sim(clock, capacity=Capacity(rpm=1, tpm=10_000))
    sim.open(make_request(MODEL))
    with pytest.raises(CapacityExceeded) as exc:
        sim.open(make_request(MODEL))
    assert exc.value.error_class is ErrorClass.RATE_LIMITED
    assert exc.value.signal is not None
    assert exc.value.signal.limit_scope == "account"
    assert exc.value.limits.retry_after_s == pytest.approx(60.0)


async def test_incident_windows_start_at_the_first_request(
    make_request: Callable[..., ChatRequest], clock: Any
) -> None:
    outage = Incident(start_s=10, end_s=20, errors=ErrorMix(server_error=1.0))
    sim = _sim(clock, incidents=[outage])
    clock.advance(100)  # time before the first request does not count
    assert sim.open(make_request(MODEL)).plan.error is None  # t = 0
    clock.advance(10)
    assert sim.open(make_request(MODEL)).plan.error is ErrorClass.SERVER_ERROR  # t = 10
    clock.advance(10)
    assert sim.open(make_request(MODEL)).plan.error is None  # t = 20, window closed


def test_truth_matches_the_plan(make_request: Callable[..., ChatRequest]) -> None:
    sim = _sim()
    for i in range(20):
        call = sim.open(make_request(MODEL), index=i, cohort="c")
        assert call.plan.correct == sim.truth(MODEL, "c", i)


def test_scripted_simulator_has_no_truth() -> None:
    with pytest.raises(LookupError):
        Simulator.scripted(MockBehavior()).truth(MODEL, "default", 0)
