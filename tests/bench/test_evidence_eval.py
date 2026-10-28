"""The evidence experiments (evals/evidence): their building blocks behave as described, and
the committed metrics cover what the PRD asks for. ``make eval`` re-runs the reduced gate and
compares it with the committed numbers; these tests are the fast, focused checks."""

import json
from pathlib import Path

import numpy as np
import pytest

from evals.evidence import canary, coverage, detection, sensitivity
from evals.evidence.sim import JUDGES, cell_probs, judge_stream
from relay.simulator.judge import MockJudge

METRICS = Path("evals/evidence/metrics.json")


def test_vectorized_judge_matches_the_simulators_mock_judge() -> None:
    # Same model, two implementations: the loss rates must agree.
    params = JUDGES["mediocre"]
    mock = MockJudge(seed=1, params=params)
    n = 20_000
    # True losses: candidate wrong, reference right.
    sim_on_loss = np.mean(
        [mock.verdict(cand_correct=False, ref_correct=True, index=i) == "loss" for i in range(n)]
    )
    sim_on_ok = np.mean(
        [mock.verdict(cand_correct=True, ref_correct=True, index=i) == "loss" for i in range(n)]
    )
    rng = np.random.default_rng(0)
    vec_on_loss = 1 - judge_stream(rng, np.zeros(n), params).mean()
    vec_on_ok = 1 - judge_stream(rng, np.ones(n), params).mean()
    assert vec_on_loss == pytest.approx(sim_on_loss, abs=0.015)
    assert vec_on_ok == pytest.approx(sim_on_ok, abs=0.01)


def test_cell_probabilities_sum_to_one_and_mean_w() -> None:
    p = cell_probs(0.9, JUDGES["good"])
    assert sum(p.as_array()) == pytest.approx(1.0)
    assert p.p11 + p.p10 == pytest.approx(0.9)  # the human (truth) mean is w


def test_a_coverage_cell_is_deterministic() -> None:
    a = coverage.coverage_cell(n=50, big_n=500, w=0.9, judge="good", runs=100, n_boot=50, seed=3)
    b = coverage.coverage_cell(n=50, big_n=500, w=0.9, judge="good", runs=100, n_boot=50, seed=3)
    assert a == b
    assert 0.5 < a["wald"] <= 1.0


def test_fixed_threshold_alarms_when_the_rolling_rate_drops() -> None:
    verdicts = np.ones((1, detection.WARMUP + detection.HORIZON), dtype=bool)
    verdicts[0, detection.WARMUP + 100 :] = False  # everything goes bad at monitored 100
    at = detection.fixed_threshold_alarm(verdicts, window=50)
    # 5 points below a baseline of 1.0 = 0.95: crossed after 3 bad samples in a 50-window.
    assert at[0] == 102


def test_e1_alarms_only_at_a_look() -> None:
    verdicts = np.ones((1, detection.WARMUP + detection.HORIZON), dtype=bool)
    verdicts[0, detection.WARMUP + 100 :] = False
    labeled = np.array([[180, 0, 0, 20]])  # a perfect judge on 200 labels
    at = detection.e1_alarm(verdicts, labeled, block=250, z=2.0, lam=1.0)
    assert at[0] == 249  # the end of the first block: never inside it


def test_a_monitor_that_never_alarms_is_censored_at_the_horizon() -> None:
    change = np.array([100, 200])
    r = detection._delays(np.array([detection.HORIZON, 300]), change)
    assert r["detected_within_horizon"] == 0.5
    assert r["false_alarms_before_change"] == 0


def test_ppi_protects_against_a_loss_blind_judge() -> None:
    r = sensitivity.run(runs=300)
    assert r["false_safe"]["loss_blind"]["judge_only"] > 0.5
    assert r["false_safe"]["loss_blind"]["ppi"] < 0.05


def test_a_bad_canary_is_never_promoted_and_a_good_one_never_rolled_back() -> None:
    r = canary.run(runs=10, seed=1)
    assert r["bad"]["promote_recommended"] == 0.0
    assert r["good"]["rolled_back"] == 0.0


def test_committed_metrics_cover_the_prd() -> None:
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    cells = m["coverage"]["cells"]
    # M6 must include the small-n and w near 0.95 regimes.
    assert any(c["n_human"] == 50 and c["w"] == 0.95 for c in cells)
    assert m["coverage"]["runs_per_cell"] >= 1000
    # M5: monitors tuned to the same false-alarm target, on >= 1,000 streams.
    det = m["detection"]
    assert det["streams"] >= 1000
    for name in ("fixed_threshold", "e1_matched"):
        assert det["monitors"][name]["false_alarm_rate"] <= det["false_alarm_target"]
    # Every target has a met flag (reported whether or not it is met).
    assert {"met"} <= set(m["coverage"]) and {"met"} <= set(m["detection"])
    assert {"met"} <= set(m["cost"])
    assert set(m["gate"]) >= {"coverage_wald_worst", "detection_e1_median", "m8_share"}


async def test_the_real_check_budget_is_a_clean_provider_error() -> None:
    # Past the budget a call fails like a provider quota (the gateway answers 429), never an
    # exception that kills the run and loses what was gathered.
    from evals.evidence.real_check import Counting
    from relay.providers.mock import MockProvider
    from relay.schemas.chat import ChatRequest
    from relay.schemas.errors import ErrorClass, ProviderError

    counted = Counting(MockProvider(), max_calls=1)  # type: ignore[arg-type]
    req = ChatRequest.model_validate({"model": "m", "messages": [{"role": "user", "content": "x"}]})
    await counted.complete(req)
    assert counted.exhausted
    with pytest.raises(ProviderError) as e:
        await counted.complete(req)
    assert e.value.error_class is ErrorClass.QUOTA_EXHAUSTED
    assert counted.calls == 1
