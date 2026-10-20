import statistics
from collections import Counter
from collections.abc import Callable
from dataclasses import replace

import pytest

from relay.schemas.chat import ChatRequest
from relay.schemas.errors import ErrorClass
from relay.simulator.plan import MockBehavior, Plan, SampledPlanner, ScriptedPlanner
from relay.simulator.quality import QualitySpec
from relay.simulator.scenario import (
    ErrorMix,
    Incident,
    LatencyModel,
    ModelProfile,
    OutputLength,
    Scenario,
)

N = 20_000
MODEL = "m"


def _scenario(errors: ErrorMix | None = None, length: OutputLength | None = None) -> Scenario:
    return Scenario(
        models={
            MODEL: ModelProfile(
                latency=LatencyModel(
                    ttft_median_s=0.5, ttft_p95_s=1.5, tokens_per_s=50, tokens_per_s_jitter=0.2
                ),
                length=length or OutputLength(median_tokens=2, p95_tokens=4),
                errors=errors or ErrorMix(),
                quality=QualitySpec(base=0.8),
            )
        }
    )


def _plans(
    planner: SampledPlanner, req: ChatRequest, n: int = N, incident: Incident | None = None
) -> list[Plan]:
    return [
        planner.plan(req, model=MODEL, index=i, cohort="default", incident=incident)
        for i in range(n)
    ]


def test_same_seed_and_index_give_the_same_plan(make_request: Callable[..., ChatRequest]) -> None:
    req = make_request(MODEL)
    a = SampledPlanner(_scenario(), seed=11)
    b = SampledPlanner(_scenario(), seed=11)
    assert _plans(a, req, 100) == _plans(b, req, 100)
    assert _plans(a, req, 100) != _plans(SampledPlanner(_scenario(), seed=12), req, 100)


def test_raising_the_error_rate_leaves_latency_and_quality_untouched(
    make_request: Callable[..., ChatRequest],
) -> None:
    # Common random numbers: two scenarios that differ only in errors see identical
    # latency, length and correctness, request by request.
    req = make_request(MODEL)
    calm = _plans(SampledPlanner(_scenario(), seed=1), req, 500)
    stormy = _plans(SampledPlanner(_scenario(ErrorMix(server_error=0.5)), seed=1), req, 500)
    assert any(p.error for p in stormy)
    for a, b in zip(calm, stormy, strict=True):
        assert (a.first_chunk_delay_s, a.chunk_delay_s, a.reply_words, a.correct) == (
            b.first_chunk_delay_s,
            b.chunk_delay_s,
            b.reply_words,
            b.correct,
        )


def test_ttft_matches_the_configured_median_and_p95(
    make_request: Callable[..., ChatRequest],
) -> None:
    ttfts = sorted(
        p.first_chunk_delay_s for p in _plans(SampledPlanner(_scenario(), 0), make_request(MODEL))
    )
    assert statistics.median(ttfts) == pytest.approx(0.5, rel=0.05)
    assert ttfts[int(0.95 * N)] == pytest.approx(1.5, rel=0.05)


def test_token_rate_stays_within_the_jitter(make_request: Callable[..., ChatRequest]) -> None:
    delays = [
        p.chunk_delay_s for p in _plans(SampledPlanner(_scenario(), 0), make_request(MODEL), 2000)
    ]
    assert min(delays) >= 1 / (50 * 1.2) - 1e-12
    assert max(delays) <= 1 / (50 * 0.8) + 1e-12


def test_error_frequencies_match_the_mix(make_request: Callable[..., ChatRequest]) -> None:
    mix = ErrorMix(rate_limited=0.05, server_error=0.02, stream_broken=0.03)
    plans = _plans(SampledPlanner(_scenario(mix), seed=4), make_request(MODEL))
    counts = Counter(p.error for p in plans)
    for error_class, p in [
        (ErrorClass.RATE_LIMITED, 0.05),
        (ErrorClass.SERVER_ERROR, 0.02),
        (ErrorClass.STREAM_BROKEN, 0.03),
    ]:
        sd = (N * p * (1 - p)) ** 0.5
        assert abs(counts[error_class] - N * p) < 4 * sd, error_class


def test_error_details_follow_the_class(make_request: Callable[..., ChatRequest]) -> None:
    mix = ErrorMix(rate_limited=0.3, timeout=0.3, stream_broken=0.3, retry_after_s=2.5, hang_s=7)
    for p in _plans(SampledPlanner(_scenario(mix), seed=5), make_request(MODEL), 300):
        if p.error is ErrorClass.RATE_LIMITED:
            assert p.retry_after_s == 2.5
        if p.error is ErrorClass.TIMEOUT:
            # The hang is the only wait: no first token ever arrives.
            assert (p.hang_s, p.latency_s, p.first_chunk_delay_s) == (7, 0, 0)
        if p.error is ErrorClass.STREAM_BROKEN:
            assert p.fail_after_chunks is not None
            assert 0 <= p.fail_after_chunks < len(p.reply_words)
        if p.error is None:
            assert (p.retry_after_s, p.hang_s, p.fail_after_chunks) == (None, 0, None)


def test_max_tokens_caps_the_reply(make_request: Callable[..., ChatRequest]) -> None:
    long_replies = _scenario(length=OutputLength(median_tokens=50, p95_tokens=100))
    plans = _plans(SampledPlanner(long_replies, 0), make_request(MODEL, max_tokens=10), 200)
    assert all(len(p.reply_words) <= 10 for p in plans)
    assert {p.finish_reason for p in plans} == {"length"}


def test_an_incident_overrides_errors_and_scales_latency(
    make_request: Callable[..., ChatRequest],
) -> None:
    planner = SampledPlanner(_scenario(), seed=6)
    req = make_request(MODEL)
    outage = Incident(start_s=0, end_s=1, errors=ErrorMix(server_error=1.0), latency_multiplier=3)
    normal = _plans(planner, req, 50)
    during = _plans(planner, req, 50, incident=outage)
    assert all(p.error is ErrorClass.SERVER_ERROR for p in during)
    for a, b in zip(normal, during, strict=True):
        assert b.first_chunk_delay_s == pytest.approx(3 * a.first_chunk_delay_s)


def test_correct_follows_q(make_request: Callable[..., ChatRequest]) -> None:
    plans = _plans(SampledPlanner(_scenario(), seed=7), make_request(MODEL))
    assert sum(bool(p.correct) for p in plans) / N == pytest.approx(0.8, abs=0.015)


def test_scripted_planner_reproduces_the_phase_1_mock(
    make_request: Callable[..., ChatRequest],
) -> None:
    behavior = MockBehavior(latency_s=0.2, error=ErrorClass.OVERLOADED, retry_after_s=3.0)
    plan = ScriptedPlanner(behavior).plan(
        make_request(MODEL, "hello there"), model=MODEL, index=4, cohort="default", incident=None
    )
    assert plan == Plan(
        index=4,
        reply_words=("mock:", "hello", "there"),
        finish_reason="stop",
        latency_s=0.2,
        error=ErrorClass.OVERLOADED,
        retry_after_s=3.0,
    )
    assert replace(plan, index=0).correct is None  # scripted plans carry no ground truth
