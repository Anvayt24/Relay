"""The labeling tool and the calibration report."""

import json
import random
from pathlib import Path
from typing import Any

import httpx
import pytest

from relay.evidence.calibrate import calibrate, w_self
from relay.evidence.label import (
    RELABEL_FRACTION,
    build_queue,
    candidate_label,
    create_app,
)
from relay.evidence.pairs import EvalPair, read_jsonl, write_jsonl
from relay.evidence.validate import JUDGES, simulate


def pair(
    i: int, *, cohort: str = "c", judge: str | None = "tie", swapped: bool = False
) -> EvalPair:
    return EvalPair(
        id=f"p{i:03d}",
        prompt_hash=f"h{i}",
        prompt=f"prompt {i}",
        cohort=cohort,
        candidate_model="cand",
        reference_model="ref",
        candidate_response=f"CANDIDATE-{i}",
        reference_response=f"REFERENCE-{i}",
        judge_label=judge,
        judge_model="judge",
        judge_prompt_version="v1",
        position_swapped=swapped,
        text_retained=True,
        ts="2026-01-01T00:00:00+00:00",
        source="pilot",
    )


# ---------------------------------------------------------------- queue
def test_queue_interleaves_cohorts_and_adds_blind_relabels() -> None:
    pairs = [pair(i, cohort="a") for i in range(30)] + [
        pair(100 + i, cohort="b") for i in range(30)
    ]
    queue = build_queue(pairs, seed=1)
    originals = [q for q in queue if not q.is_relabel]
    relabels = [q for q in queue if q.is_relabel]
    assert len(originals) == 60
    assert len(relabels) == round(60 * RELABEL_FRACTION)
    # Stratified: the first ten items are not all one cohort.
    cohorts = {p.id: p.cohort for p in pairs}
    assert {cohorts[q.pair_id] for q in originals[:10]} == {"a", "b"}
    # A relabel always comes after its original, with a gap.
    for r in relabels:
        assert queue.index(r) > next(i for i, q in enumerate(queue) if q.pair_id == r.pair_id)
        assert queue.index(r) - next(i for i, q in enumerate(queue) if q.pair_id == r.pair_id) >= 1


def test_queue_is_deterministic_for_a_seed() -> None:
    pairs = [pair(i) for i in range(40)]
    assert build_queue(pairs, seed=5) == build_queue(pairs, seed=5)
    assert build_queue(pairs, seed=5) != build_queue(pairs, seed=6)


@pytest.mark.parametrize(
    ("choice", "swapped", "label"),
    [
        ("neither", False, "tie"),
        ("a_worse", False, "loss"),  # candidate is on the left (A): A worse = loss
        ("b_worse", False, "win"),
        ("a_worse", True, "win"),  # candidate on the right (B): A (the reference) is worse
        ("b_worse", True, "loss"),
    ],
)
def test_button_to_candidate_label(choice: str, swapped: bool, label: str) -> None:
    assert candidate_label(choice, shown_swapped=swapped) == label


# ---------------------------------------------------------------- the web app
async def test_labeling_flow_is_blind_resumable_and_logs_candidate_labels(tmp_path: Path) -> None:
    pairs = [pair(i) for i in range(12)]
    pairs_path, labels_path = tmp_path / "pairs.jsonl", tmp_path / "labels.jsonl"
    write_jsonl(pairs, pairs_path)

    async def session(n: int) -> list[dict[str, Any]]:
        app = create_app(pairs_path, labels_path, annotator="me", seed=2)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://label.test"
        ) as c:
            seen = []
            for _ in range(n):
                item = (await c.get("/api/next")).json()
                if item.get("finished"):
                    break
                # Blind: the page gets the prompt and two texts, never a model name.
                assert set(item) == {"pair_id", "is_relabel", "prompt", "a", "b", "progress"}
                assert "cand" not in json.dumps(item).replace("CANDIDATE", "")
                seen.append(item)
                await c.post(
                    "/api/label",
                    json={
                        "pair_id": item["pair_id"],
                        "choice": "a_worse",
                        "is_relabel": item["is_relabel"],
                    },
                )
            return seen

    first = await session(5)
    assert len(first) == 5
    # Restarting continues where it stopped: nothing is asked twice.
    second = await session(100)
    ids = [(i["pair_id"], i["is_relabel"]) for i in first + second]
    assert len(ids) == len(set(ids))
    rows = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(ids)
    assert {r["label"] for r in rows} <= {"loss", "win"}  # "a_worse" every time
    # Some were shown swapped (the candidate on the right), so labels differ from the button.
    assert {r["shown_swapped"] for r in rows} == {True, False}


# ---------------------------------------------------------------- calibration gate
def labels_for(pairs: list[EvalPair], humans: list[str]) -> list[dict[str, Any]]:
    return [
        {"pair_id": p.id, "label": h, "annotator": "me", "is_relabel": False}
        for p, h in zip(pairs, humans, strict=True)
    ]


def test_too_few_labels_is_insufficient_not_calibrated() -> None:
    pairs = [pair(i, judge="loss") for i in range(20)]
    report = calibrate(pairs, labels_for(pairs, ["loss"] * 20))
    assert report["status"] == "insufficient_n"  # perfect agreement on 20 pairs proves nothing


def test_a_good_judge_passes_the_gate_and_a_loss_blind_one_fails() -> None:
    rng = random.Random(0)
    humans = ["loss" if rng.random() < 0.3 else "tie" for _ in range(300)]

    def judged(recall: float, false_alarm: float) -> list[EvalPair]:
        out = []
        for i, h in enumerate(humans):
            says_loss = rng.random() < (recall if h == "loss" else false_alarm)
            out.append(pair(i, judge="loss" if says_loss else "tie"))
        return out

    good = judged(0.92, 0.04)
    blind = judged(0.40, 0.04)
    assert calibrate(good, labels_for(good, humans))["status"] == "calibrated"
    report = calibrate(blind, labels_for(blind, humans))
    assert report["status"] == "uncalibrated"
    assert report["loss_recall"]["value"] < 0.7
    # Missing most losses still leaves high accuracy (losses are only 30% here), which is
    # why the gate is on loss recall and kappa, not accuracy.
    assert report["accuracy"]["value"] > 0.75


def test_self_consistency_and_position_bias_are_reported() -> None:
    pairs = [pair(i, judge="loss" if i % 4 == 0 else "tie", swapped=i % 2 == 0) for i in range(80)]
    humans = ["loss" if i % 4 == 0 else "tie" for i in range(80)]
    labels = labels_for(pairs, humans)
    # A blind re-label of pairs 0..19 (a mix of losses and ties) that flips two of them.
    for i in range(20):
        flipped = i in (0, 1)
        again = ("tie" if humans[i] == "loss" else "loss") if flipped else humans[i]
        labels.append(
            {"pair_id": f"p{i:03d}", "label": again, "annotator": "me", "is_relabel": True}
        )
    report = calibrate(pairs, labels)
    assert report["annotator_self_kappa"]["n"] == 20
    assert 0 < report["annotator_self_kappa"]["kappa"] < 1  # consistent, but not perfectly
    assert set(report["position_bias"]) == {"candidate_as_A", "candidate_as_B"}


def test_w_self_is_the_share_of_win_or_tie() -> None:
    pairs = [pair(i, judge="loss" if i < 3 else "tie") for i in range(10)]
    assert w_self(pairs) == {"n": 10, "w_self": 0.7}


# ---------------------------------------------------------------- validated against the simulator
@pytest.mark.parametrize(("name", "sensitivity", "false_loss"), JUDGES)
def test_calibration_recovers_the_simulated_judges_true_accuracy(
    name: str, sensitivity: float, false_loss: float
) -> None:
    r = simulate(name, sensitivity, false_loss, n=600, w=0.8, seed=11)
    assert r["recall_ci_contains_configured"], r
    expected = "calibrated" if name == "good" else "uncalibrated"
    assert r["status"] == expected


def test_pairs_round_trip_through_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    original = [pair(1), pair(2, judge=None)]
    write_jsonl(original, path)
    assert read_jsonl(path) == original
    assert original[1].judge_loss is None and original[0].judge_loss is False
