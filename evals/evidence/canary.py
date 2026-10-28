"""Canary decisions (PRD FR-11, §10.7): is a bad canary rolled back, and a good one recommended?

A canary bundle serves a share of traffic; the shadow stage judges its cheap-routed answers
against the strong model (comparison ``canary:<version>``); the worker computes that card at
pre-registered looks. This experiment runs the **real** card code (``compute_cards``) on
simulated canaries, look by look, exactly as the worker would:

- a **bad** canary (true w = 0.84, below the 0.90 floor) should be rolled back;
- a **good** canary (true w = 0.97) should end ``safe`` with ``promote`` recommended, and is
  never promoted by the engine itself (ADR 0022).

Human labels for the canary's comparison (the rectifier) are a small labeled sample of canary
pairs: PRD §10.4 asks for a label refresh after a policy change, and a canary is one.
"""

from datetime import UTC, datetime, timedelta
from statistics import median
from typing import Any

import numpy as np

from evals.evidence.sim import JUDGES, judge_stream
from relay.evidence.cards import (
    ALL,
    CardSpec,
    JudgedPair,
    RouteCard,
    calibration_by_judge,
    compute_cards,
)
from relay.evidence.looks import LookSchedule

FLOOR = 0.90
LOOK_EVERY = 250
MAX_LOOKS = 4
N_HUMAN = 100
CASES = {"bad": 0.84, "good": 0.97}
T0 = datetime(2026, 10, 28, tzinfo=UTC)


def _pairs(rng: np.random.Generator, w: float, n: int, start: int, human: bool) -> list[JudgedPair]:
    truly_ok = rng.random(n) < w
    judge = JUDGES["good"]
    says_loss = rng.random(n) < np.where(truly_ok, judge.false_loss_rate, judge.sensitivity)
    return [
        JudgedPair(
            pair_id=f"c{start + i}",
            route="default",
            comparison="canary:v2",
            cohort="all-traffic",
            judge_label="loss" if says_loss[i] else "tie",
            judge_model="judge",
            judge_prompt_version="v1",
            ts=T0 + timedelta(seconds=start + i),
            policy_version="v2",
            human_label=("tie" if truly_ok[i] else "loss") if human else None,
            human_ts=T0 if human else None,
        )
        for i in range(n)
    ]


def _calibration(rng: np.random.Generator) -> dict[str, Any]:
    # The judge's M7 numbers from a stratified, loss-rich set (as in practice).
    truly_ok = rng.random(400) < 0.7
    verdict_ok = judge_stream(rng, np.where(truly_ok, 1.0, 0.0), JUDGES["good"])
    pairs = [
        JudgedPair(
            f"cal{i}",
            "",
            "calibration",
            "x",
            "tie" if verdict_ok[i] else "loss",
            "judge",
            "v1",
            T0,
            human_label="tie" if truly_ok[i] else "loss",
            human_ts=T0,
        )
        for i in range(400)
    ]
    return calibration_by_judge(pairs)


def one_canary(rng: np.random.Generator, w: float, calibration: Any) -> tuple[str, int]:
    """Run looks until the card decides; (outcome, judged pairs used)."""
    spec = CardSpec(
        floor=FLOOR, looks=LookSchedule(every=LOOK_EVERY, max_looks=MAX_LOOKS), min_cohort_n=50
    )
    labeled = _pairs(rng, w, N_HUMAN, 0, human=True)
    stream = _pairs(rng, w, LOOK_EVERY * MAX_LOOKS, N_HUMAN, human=False)
    previous: dict[tuple[str, str, str], RouteCard] = {}
    now = T0 + timedelta(days=1)
    for look in range(1, MAX_LOOKS + 1):
        cards = compute_cards(
            labeled + stream[: look * LOOK_EVERY],
            spec,
            calibration=calibration,
            now=now,
            previous=previous,
        )
        previous = {c.key: c for c in cards}
        card = next(c for c in cards if c.cohort == ALL)
        if card.recommended_action == "rollback":
            return "rolled_back", look * LOOK_EVERY
        if card.recommended_action == "promote":
            return "promote_recommended", look * LOOK_EVERY
    return "undecided", MAX_LOOKS * LOOK_EVERY


def run(*, runs: int = 200, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    calibration = _calibration(rng)
    out: dict[str, Any] = {
        "runs": runs,
        "floor": FLOOR,
        "look_every": LOOK_EVERY,
        "max_looks": MAX_LOOKS,
        "n_human": N_HUMAN,
    }
    for name, w in CASES.items():
        results = [one_canary(rng, w, calibration) for _ in range(runs)]
        decided = [n for outcome, n in results if outcome != "undecided"]
        out[name] = {
            "w": w,
            "rolled_back": round(sum(o == "rolled_back" for o, _ in results) / runs, 4),
            "promote_recommended": round(
                sum(o == "promote_recommended" for o, _ in results) / runs, 4
            ),
            "undecided": round(sum(o == "undecided" for o, _ in results) / runs, 4),
            "median_pairs_to_decision": median(decided) if decided else None,
        }
    return out
