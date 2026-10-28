"""Route cards (PRD §10.8; ADRs 0019-0021): the state machine, looks, pooling and actions."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from relay.evidence.cards import (
    ALL,
    OTHER,
    CardSpec,
    JudgedPair,
    action,
    calibration_by_judge,
    compute_cards,
    pooling_failures,
    render_markdown,
)
from relay.evidence.looks import LookSchedule

NOW = datetime(2026, 10, 28, 12, tzinfo=UTC)


def make_pairs(
    *,
    n_judge: int,
    n_human: int,
    w: float,
    comparison: str = "cheap_vs_strong",
    cohorts: tuple[str, ...] = ("c1",),
    sensitivity: float = 0.9,
    false_loss: float = 0.03,
    seed: int = 0,
    judge: str = "judge-a",
    label_age: timedelta = timedelta(days=1),
    route: str = "default",
) -> list[JudgedPair]:
    """Pairs whose truth has win-or-tie rate ``w`` and a judge with known errors. The first
    ``n_human`` carry a human label equal to the truth."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_judge + n_human):
        ok = bool(rng.random() < w)
        says_loss = rng.random() < (sensitivity if not ok else false_loss)
        labeled = i < n_human
        out.append(
            JudgedPair(
                pair_id=f"{seed}-{i}",
                route=route,
                comparison=comparison,
                cohort=cohorts[i % len(cohorts)],
                judge_label="loss" if says_loss else "tie",
                judge_model=judge,
                judge_prompt_version="v1",
                ts=NOW - timedelta(hours=1) + timedelta(seconds=i),
                policy_version="p1",
                human_label=("tie" if ok else "loss") if labeled else None,
                human_ts=NOW - label_age if labeled else None,
            )
        )
    return out


def calibration_set(judge: str = "judge-a", sensitivity: float = 0.9, false_loss: float = 0.03):  # type: ignore[no-untyped-def]
    """The judge's M7 numbers from a separate, loss-rich labeled set, as in practice: the
    calibration set is stratified so it holds enough losses (at w = 0.98, 200 labels hold only
    about four, too few for kappa or loss recall: the kappa paradox)."""
    cal = make_pairs(
        n_judge=0,
        n_human=400,
        w=0.7,
        judge=judge,
        sensitivity=sensitivity,
        false_loss=false_loss,
        seed=99,
        comparison="calibration",
    )
    return calibration_by_judge(cal)


def cards_for(
    pairs: list[JudgedPair],
    spec: CardSpec | None = None,
    calibration=None,
    **kw: object,  # type: ignore[no-untyped-def]
) -> dict:
    spec = spec or CardSpec(looks=LookSchedule(every=500, max_looks=4), min_cohort_n=50)
    calibration = calibration if calibration is not None else calibration_set()
    out = compute_cards(pairs, spec, calibration=calibration, now=NOW, **kw)  # type: ignore[arg-type]
    return {c.cohort: c for c in out}


def test_no_human_labels_means_uncalibrated_and_judge_only() -> None:
    cards = cards_for(make_pairs(n_judge=1000, n_human=0, w=0.97))
    card = cards[ALL]
    assert card.state == "uncalibrated"
    assert card.recommended_action == "none"
    assert card.ci is None
    assert card.estimate is not None and "judge-only" in (card.reason or "")


def test_a_loss_blind_judge_fails_the_gate() -> None:
    pairs = make_pairs(n_judge=1000, n_human=300, w=0.85, sensitivity=0.3, false_loss=0.02)
    blind = calibration_set(sensitivity=0.3, false_loss=0.02)
    assert cards_for(pairs, calibration=blind)[ALL].state == "uncalibrated"


def test_old_labels_make_the_rectifier_stale() -> None:
    pairs = make_pairs(n_judge=1000, n_human=200, w=0.97, label_age=timedelta(days=45))
    card = cards_for(pairs)[ALL]
    assert card.state == "stale"
    assert card.recommended_action == "none"


def test_before_the_first_look_nothing_is_decided() -> None:
    card = cards_for(make_pairs(n_judge=300, n_human=200, w=0.99))[ALL]
    assert card.state == "undecided"
    assert card.look == 0
    assert card.recommended_action == "collect_more"
    assert card.samples_needed == 200


@pytest.mark.parametrize(
    ("w", "state", "action_"),
    [(0.98, "safe", "none"), (0.80, "unsafe", "raise_threshold")],
)
def test_clear_cases_are_decided(w: float, state: str, action_: str) -> None:
    card = cards_for(make_pairs(n_judge=2000, n_human=200, w=w))[ALL]
    assert card.state == state
    assert card.recommended_action == action_
    assert card.look == 4
    assert card.ci is not None and card.ci[0] <= card.estimate <= card.ci[1]  # type: ignore[operator]


def test_a_rate_at_the_floor_stays_undecided_with_a_power_estimate() -> None:
    card = cards_for(make_pairs(n_judge=600, n_human=200, w=0.90, seed=3))[ALL]
    assert card.state == "undecided"
    assert card.look == 1
    assert card.samples_needed is None or card.samples_needed > 0


def test_after_the_last_look_undecided_is_final() -> None:
    spec = CardSpec(looks=LookSchedule(every=200, max_looks=2), min_cohort_n=50)
    card = cards_for(make_pairs(n_judge=1000, n_human=200, w=0.90, seed=4), spec)[ALL]
    if card.state == "undecided":
        assert card.look == 2
        assert "last pre-registered look" in (card.reason or "")
        assert card.samples_needed is None


def test_the_decision_uses_only_the_look_sample() -> None:
    # 1,400 judged pairs, looks every 500: the decision at look 2 uses exactly 1,000 of them.
    pairs = make_pairs(n_judge=1400, n_human=200, w=0.97)
    cards = cards_for(pairs)
    assert cards[ALL].look == 2
    assert cards[ALL].n_judge == 1400
    assert sum(c.n_judge for c in cards.values() if c.cohort == ALL) == 1400


def test_a_state_only_changes_at_a_new_look() -> None:
    pairs = make_pairs(n_judge=1200, n_human=200, w=0.97)
    first = cards_for(pairs)
    assert first[ALL].state == "safe"
    # Pretend the last run, at the same look, said undecided: new data within the look (more
    # human labels, say) must not turn one look into a second test.
    prev = {c.key: replace(c, state="undecided") for c in first.values()}
    again = cards_for(pairs, previous=prev)
    assert again[ALL].state == "undecided"
    # At the next look the decision is taken afresh.
    more = make_pairs(n_judge=1600, n_human=200, w=0.97)
    assert cards_for(more, previous=prev)[ALL].state == "safe"


def test_pairs_from_another_judge_are_excluded() -> None:
    a = make_pairs(n_judge=1000, n_human=200, w=0.97, judge="judge-a")
    b = make_pairs(n_judge=100, n_human=0, w=0.5, judge="judge-b", seed=9)
    card = cards_for(a + b)[ALL]
    assert card.judge == "judge-a@v1"
    assert card.n_excluded == 100
    assert card.state == "safe"


def test_small_cohorts_merge_into_other() -> None:
    pairs = make_pairs(n_judge=1000, n_human=200, w=0.97, cohorts=("big",) * 18 + ("s1", "s2"))
    cards = cards_for(pairs, CardSpec(looks=LookSchedule(every=500, max_looks=4), min_cohort_n=100))
    assert set(cards) == {ALL, "big", OTHER}  # s1 and s2 have about 50 each


def test_a_cohort_whose_judge_bias_differs_fails_the_pooling_test() -> None:
    # Cohort "odd": the judge misses every loss there, so its residual (human - judge) is
    # negative where the rest of the labels say the judge is fine.
    good = make_pairs(n_judge=0, n_human=150, w=0.97, cohorts=("ok",), seed=1)
    odd = make_pairs(
        n_judge=0, n_human=150, w=0.80, cohorts=("odd",), sensitivity=0.0, false_loss=0.0, seed=2
    )
    # With two cohorts "this one differs from the rest" is symmetric, so both are flagged:
    # the test can say the bias isn't shared, not which cohort is to blame.
    assert "odd" in pooling_failures(good + odd, 0.05)
    # Without enough labels in a cohort it isn't tested at all (no power, ADR 0021).
    assert "odd" not in pooling_failures(good + odd[:40], 0.05)


def test_a_pooling_failure_blocks_a_safe_state() -> None:
    good = make_pairs(n_judge=600, n_human=150, w=0.97, cohorts=("ok",), seed=1)
    odd = make_pairs(
        n_judge=600, n_human=150, w=0.80, cohorts=("odd",), sensitivity=0.0, false_loss=0.0, seed=2
    )
    pairs = [replace(p, ts=p.ts + timedelta(milliseconds=k)) for k, p in enumerate(good + odd)]
    cards = cards_for(pairs, CardSpec(looks=LookSchedule(every=500, max_looks=2), min_cohort_n=50))
    assert cards["odd"].state != "safe"


def test_band_promotion_needs_every_cohort_safe() -> None:
    pairs = make_pairs(
        n_judge=2000, n_human=200, w=0.99, comparison="band:0.7-0.82", cohorts=("a", "b")
    )
    cards = cards_for(pairs)
    assert all(c.state == "safe" for c in cards.values())
    assert cards[ALL].recommended_action == "promote"
    assert cards["a"].recommended_action == "collect_more"  # promotion is a route-level call


def test_bootstrap_interval_option() -> None:
    spec = CardSpec(
        looks=LookSchedule(every=500, max_looks=4),
        min_cohort_n=50,
        interval="bootstrap",
        n_boot=400,
    )
    card = cards_for(make_pairs(n_judge=2000, n_human=200, w=0.98), spec)[ALL]
    assert card.interval == "bootstrap"
    assert card.state == "safe"


@pytest.mark.parametrize(
    ("comparison", "state", "promotable", "expected"),
    [
        ("fallback:a->b", "unsafe", False, "disable_edge"),
        ("canary:v2", "unsafe", False, "rollback"),
        ("canary:v2", "safe", True, "promote"),
        ("band:0.7-0.8", "unsafe", False, "do_not_promote"),
        ("band:0.7-0.8", "safe", False, "collect_more"),
        ("cheap_vs_strong", "safe", True, "none"),
        ("fallback:a->b", "uncalibrated", True, "none"),
        ("fallback:a->b", "stale", True, "none"),
        ("fallback:a->b", "undecided", True, "collect_more"),
    ],
)
def test_actions_follow_prd_10_7(
    comparison: str, state: str, promotable: bool, expected: str
) -> None:
    assert action(comparison, state, promotable) == expected  # type: ignore[arg-type]


def test_cards_serialize_and_render() -> None:
    cards = list(cards_for(make_pairs(n_judge=1000, n_human=200, w=0.97)).values())
    json.dumps([c.as_dict() for c in cards])
    table = render_markdown(cards)
    assert "cheap_vs_strong" in table and "| all |" in table


def test_routes_and_comparisons_get_separate_cards() -> None:
    a = make_pairs(n_judge=600, n_human=100, w=0.97, route="r1")
    b = make_pairs(n_judge=600, n_human=100, w=0.97, route="r2", comparison="fallback:x->y", seed=5)
    out = compute_cards(a + b, CardSpec(min_cohort_n=50), calibration=calibration_set(), now=NOW)
    assert {(c.route, c.comparison) for c in out} == {
        ("r1", "cheap_vs_strong"),
        ("r2", "fallback:x->y"),
    }
