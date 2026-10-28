"""Turning judged pairs into the real router dataset (no model, no network)."""

from evals.router.real_build import gold_by_split, rows_from_pairs
from relay.evidence.pairs import EvalPair, prompt_hash


def pair(i: int, label: str | None) -> EvalPair:
    prompt = f"question {i}"
    return EvalPair(
        id=f"real-{i:04d}",
        prompt_hash=prompt_hash(prompt),
        prompt=prompt,
        cohort="lang=latin,len=short,code=false,tools=false",
        candidate_model="groq/gpt-oss-20b",
        reference_model="groq/gpt-oss-120b",
        candidate_response="c",
        reference_response="r",
        judge_label=label,
        judge_model="judge",
        judge_prompt_version="v1",
        position_swapped=False,
        text_retained=True,
        ts="2026-01-01T00:00:00+00:00",
        source="real",
    )


def test_abstentions_are_dropped_not_guessed() -> None:
    rows, dropped = rows_from_pairs([pair(0, "win"), pair(1, None), pair(2, "loss")], {})
    assert dropped == 1
    assert [r["y_judge"] for r in rows] == [1, 0]  # win-or-tie = 1, loss = 0


def test_there_is_no_ground_truth_without_human_labels() -> None:
    rows, _ = rows_from_pairs([pair(0, "tie"), pair(1, "loss")], {})
    assert all(r["y_true"] is None for r in rows)
    assert not any(gold_by_split(rows, ["val", "test"]))  # no gold -> no PPI


def test_human_labeled_pairs_in_held_out_splits_become_the_gold_set() -> None:
    pairs = [pair(0, "tie"), pair(1, "loss"), pair(2, "win")]
    human = {"real-0000": "tie", "real-0001": "loss"}  # pair 2 has no human label
    rows, _ = rows_from_pairs(pairs, human)
    assert [r["y_true"] for r in rows] == [1, 0, None]
    # Gold = human-labeled AND held out; a labeled training prompt is not gold.
    assert gold_by_split(rows, ["train", "test", "val"]) == [False, True, False]
