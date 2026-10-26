"""The cache evaluation code, its committed baseline, and the config that depends on it."""

import json
import math
from pathlib import Path

import pytest

from evals.cache import evaluate
from relay.config.loader import load_config

METRICS = Path(evaluate.HERE) / "metrics.json"


def pair(sim: float, label: int, kind: str = "near_miss", guard_ok: bool = True) -> dict:
    return {"sim": sim, "label": label, "kind": kind, "guard_ok": guard_ok, "split": "tune"}


def test_threshold_is_the_smallest_with_enough_precise_hits() -> None:
    # 40 true paraphrases above 0.90, wrong pairs only below it: 0.90 is the first threshold
    # where precision reaches 99% with at least MIN_HITS hits.
    rows = [pair(0.95, 1) for _ in range(40)] + [pair(0.80, 0) for _ in range(40)]
    rows += [pair(0.85, 0)]
    t = evaluate.choose_threshold(evaluate.curve(rows, guard=True))
    assert t is not None and 0.85 < t <= 0.95


def test_too_few_hits_never_qualifies() -> None:
    rows = [pair(0.99, 1) for _ in range(evaluate.MIN_HITS - 1)] + [pair(0.2, 0)]
    assert evaluate.choose_threshold(evaluate.curve(rows, guard=True)) is None


def test_the_lexical_guard_removes_hits_with_different_numbers_or_names() -> None:
    wrong = [pair(0.97, 0, guard_ok=False) for _ in range(10)]
    right = [pair(0.97, 1) for _ in range(40)]
    assert len(evaluate.hits_at(wrong + right, 0.9, guard=False)) == 50
    assert len(evaluate.hits_at(wrong + right, 0.9, guard=True)) == 40


def test_the_committed_baseline_matches_a_fresh_run() -> None:
    fresh = evaluate.headline(evaluate.run())
    committed = evaluate.headline(json.loads(METRICS.read_text(encoding="utf-8")))
    assert fresh == committed  # otherwise: `make eval-cache` and commit metrics.json


def test_the_frozen_data_is_what_the_manifest_says() -> None:
    import hashlib

    manifest = json.loads((evaluate.HERE / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((evaluate.HERE / name).read_bytes()).hexdigest() == digest
    assert manifest["pairs"] >= 1500  # PRD FR-6: roughly 1,500+ labeled pairs


def test_no_prompt_or_fact_is_on_both_sides_of_the_split() -> None:
    rows = evaluate.load()
    prompts = {
        s: {p for r in rows if r["split"] == s for p in (r["a"], r["b"])} for s in ("tune", "test")
    }
    assert not prompts["tune"] & prompts["test"]


def test_the_chosen_threshold_reaches_the_target_on_test() -> None:
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    test = metrics["guard_on"]["test"]
    assert test["precision"] >= evaluate.TARGET_PRECISION  # M3: point estimate >= 99%
    assert not math.isnan(test["precision_lower_95"])


@pytest.mark.parametrize("route", ["default", "live"])
def test_shipped_routes_are_not_set_below_the_measured_threshold(route: str) -> None:
    measured = json.loads(METRICS.read_text(encoding="utf-8"))["guard_on"]["threshold"]
    config = load_config(None)
    cache = config.routes[route].cache
    assert cache.lexical_guard  # the threshold was measured with the guard on
    assert cache.semantic_threshold >= measured
