"""The router evaluation: the world, the data hygiene, the committed baseline, and the gate."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evals.router import build as build_mod
from evals.router import data as data_mod
from evals.router import evaluate, train, world
from evals.router.data import Dataset
from relay.config.loader import load_config
from relay.routing.artifact import load as load_artifact

SIM = data_mod.HERE / "sim"


@pytest.fixture(scope="module")
def ds() -> Dataset:
    return data_mod.load("sim")


@pytest.fixture(scope="module")
def artifact() -> Any:
    return load_artifact(evaluate.artifact_path("sim"))


@pytest.fixture(scope="module")
def metrics(ds: Dataset, artifact: Any) -> dict[str, Any]:
    return evaluate.evaluate_world(ds, artifact)


# ---------------------------------------------------------------- the world
def test_the_world_is_deterministic_and_large_enough() -> None:
    a, b = world.generate(0), world.generate(0)
    assert [i.text for i in a] == [i.text for i in b]
    assert len(a) >= 2500
    assert len({i.text for i in a}) == len(a)  # exact duplicates were dropped
    assert {i.family for i in a} == {
        "trivia", "definition", "translation", "conversion", "arithmetic", "code",
        "reasoning", "advice",
    }  # fmt: skip


def test_difficulty_depends_on_visible_structure() -> None:
    by_family: dict[str, list[float]] = {}
    for it in world.generate(0):
        by_family.setdefault(it.family, []).append(it.w)
    mean = {f: sum(v) / len(v) for f, v in by_family.items()}
    assert mean["trivia"] > mean["conversion"] > mean["arithmetic"] > mean["reasoning"]
    assert mean["trivia"] > mean["code"] > mean["reasoning"]
    # Within a family, more digits / more sub-questions is harder (the learnable signal).
    # arithmetic instances are "arithmetic:<i>-<ops>-<digits>-<first number>"
    arith = [
        (int(it.instance.split("-")[2]), it.w)
        for it in world.generate(0)
        if it.family == "arithmetic"
    ]
    low = [w for d, w in arith if d <= 2]
    high = [w for d, w in arith if d >= 5]
    assert sum(low) / len(low) > sum(high) / len(high) + 0.1


def test_paraphrases_share_their_hidden_difficulty() -> None:
    items = [i for i in world.generate(0) if i.instance == "trivia:capital-France"]
    assert len(items) == 4
    assert len({world.hidden_w(0, i) for i in items}) == 1  # one question, one difficulty


def test_noise_is_invisible_but_bounded() -> None:
    it = world.generate(0)[0]
    shifts = {round(world.hidden_w(s, it), 3) for s in range(40)}
    assert len(shifts) > 10  # the noise really varies with the seed
    assert all(0.0 < w < 1.0 for w in shifts)


def test_realized_outcomes_match_the_hidden_probability() -> None:
    """Across many prompts of one family, the share of win-or-tie outcomes tracks mean w."""
    items = [i for i in world.generate(0) if i.family == "trivia"] + [
        i for i in world.generate(0) if i.family == "reasoning"
    ]
    for family in ("trivia", "reasoning"):
        subset = [i for i in items if i.family == family]
        outcomes, ws = [], []
        for k, it in enumerate(subset):
            y_true, _, w = world.labels(0, it, k)
            outcomes.append(y_true)
            ws.append(w)
        assert abs(sum(outcomes) / len(outcomes) - sum(ws) / len(ws)) < 0.08


# ---------------------------------------------------------------- data hygiene
def test_the_frozen_data_is_what_the_manifest_says() -> None:
    manifest = json.loads((SIM / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["sha256"].items():
        data = (SIM / name).read_bytes()
        if not name.endswith(".npy"):
            data = data.replace(b"\r\n", b"\n")  # what Linux CI checks out
        assert hashlib.sha256(data).hexdigest() == digest, name


def test_no_prompt_instance_or_cluster_is_on_both_sides_of_a_split(ds: Dataset) -> None:
    groups: dict[str, dict[Any, set[str]]] = {"text": {}, "instance": {}, "cluster": {}}
    for r in ds.rows:
        for key in groups:
            groups[key].setdefault(r[key], set()).add(r["split"])
    for key, mapping in groups.items():
        leaked = [v for v, splits in mapping.items() if len(splits) > 1]
        assert not leaked, f"{key} spans splits: {leaked[:3]}"


def test_near_duplicates_are_merged_into_clusters(ds: Dataset) -> None:
    n_clusters = len({r["cluster"] for r in ds.rows})
    n_instances = len({r["instance"] for r in ds.rows})
    assert n_clusters <= n_instances < len(ds.rows)  # paraphrases collapse into one cluster


def test_clustering_merges_by_cosine_and_by_instance() -> None:
    vecs = np.array([[1.0, 0.0], [0.99, math.sqrt(1 - 0.99**2)], [0.0, 1.0], [0.0, 1.0]])
    ids = build_mod.cluster(["a", "b", "c", "d"], vecs, threshold=0.95)
    assert ids[0] == ids[1]  # cosine 0.99 > 0.95
    assert ids[2] == ids[3]  # identical vectors
    assert ids[0] != ids[2]
    same_instance = build_mod.cluster(["x", "x", "y"], np.eye(3)[:, :2].copy(), threshold=0.95)
    assert same_instance[0] == same_instance[1] != same_instance[2]


def test_splits_are_roughly_60_20_20_and_gold_is_disjoint_from_nothing_else(ds: Dataset) -> None:
    n = len(ds.rows)
    counts = {s: int(ds.mask(s).sum()) for s in data_mod.SPLITS}
    assert 0.5 < counts["train"] / n < 0.7
    assert 0.1 < counts["val"] / n < 0.3 and 0.1 < counts["test"] / n < 0.3
    gold = ds.gold
    assert not gold[ds.mask("train")].any()  # gold exists only in held-out splits
    assert gold[ds.mask("val")].sum() == gold[ds.mask("test")].sum() == 150
    assert np.isfinite(ds.y_true).all()  # truth is known for every row in the simulation


def test_labels_look_like_a_noisy_judge_of_the_truth(ds: Dataset) -> None:
    truth, judge = ds.y_true, ds.y_judge
    losses = truth == 0
    # Sensitivity 0.90: the judge catches about 90% of true losses; false-loss rate 0.03.
    assert 0.85 < (judge[losses] == 0).mean() < 0.95
    assert (judge[~losses] == 0).mean() < 0.07


# ---------------------------------------------------------------- evaluation arithmetic
def test_auc_known_values() -> None:
    assert evaluate.auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert evaluate.auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == 0.0
    assert evaluate.auc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 1, 0, 1])) == 0.5
    assert math.isnan(evaluate.auc(np.array([0.1, 0.2]), np.array([1, 1])))


def test_policy_terms_count_strong_routed_requests_as_ties() -> None:
    cheap = np.array([True, True, False, False])
    labels = np.array([1.0, 0.0, 0.0, 1.0])
    # Cheap-routed: their own outcome (1, 0). Strong-routed: 1 regardless of the label.
    np.testing.assert_array_equal(evaluate.policy_terms(cheap, labels), [1.0, 0.0, 1.0, 1.0])


def test_the_threshold_is_the_most_aggressive_one_that_clears_the_floor() -> None:
    """A tiny synthetic dataset where the answer is known: scores perfectly ordered by label."""
    n = 400
    rng = np.random.default_rng(0)
    y = (rng.random(n) < 0.8).astype(float)
    scores = np.where(y == 1, 0.9, 0.1) + rng.normal(0, 0.01, n)
    rows = [
        {
            "split": "val", "family": "f", "is_gold": i < 100, "y_true": y[i], "y_judge": y[i],
            "w": None, "text": "t", "cluster": i, "instance": str(i), "id": str(i),
        }
        for i in range(n)
    ]  # fmt: skip
    synthetic = Dataset("syn", rows, np.zeros((n, 2)), np.zeros((n, 1)), {})
    out = evaluate.select_threshold(synthetic, scores)
    # Losses all score about 0.1 and wins about 0.9, with a perfect judge: the most aggressive
    # safe threshold sits between the two clusters (any lower sends losses cheap).
    assert out["met"] and 0.05 < out["threshold"] < 0.95


def test_a_threshold_that_cannot_clear_the_floor_sends_nothing_cheap() -> None:
    n = 300
    y = np.zeros(n)  # the cheap model is always wrong
    # Shuffled, so the first 100 rows (the gold subset) are a random sample of scores. PPI needs
    # that: gold drawn from the lowest scores would all be "sent strong" and look perfect.
    scores = np.random.default_rng(1).permutation(np.linspace(0, 1, n))
    rows = [
        {
            "split": "val", "family": "f", "is_gold": i < 100, "y_true": y[i], "y_judge": y[i],
            "w": None, "text": "t", "cluster": i, "instance": str(i), "id": str(i),
        }
        for i in range(n)
    ]  # fmt: skip
    synthetic = Dataset("syn", rows, np.zeros((n, 2)), np.zeros((n, 1)), {})
    out = evaluate.select_threshold(synthetic, scores)
    assert out["threshold"] > 0.95  # only the very highest scores could ever go cheap


# ---------------------------------------------------------------- the router's results
def test_the_router_beats_random_and_never_beats_the_oracle(metrics: dict[str, Any]) -> None:
    test = metrics["test"]
    for point in test["curve"]:
        if 0.1 < point["share_cheap"] < 0.95:
            # At the same share sent cheap, the router's quality exceeds random routing's.
            assert point["quality"] > point["random_quality_at_share"]
    assert test["baselines"]["always_strong"]["quality"] == 1.0
    assert test["baselines"]["oracle"]["quality"] == 1.0
    # The oracle sends more traffic cheap than any honest policy can at the same quality.
    assert test["baselines"]["oracle"]["share_cheap"] >= test["headline"]["cheap_share"]
    assert test["auc"] > 0.7  # it learned something real


def test_the_router_learned_the_structure_of_the_world(metrics: dict[str, Any]) -> None:
    fam = metrics["test"]["families"]
    # Trivia is the easiest family and reasoning the hardest: the router sends the former
    # cheap far more often than the latter.
    assert fam["trivia"]["router_cheap_share"] > fam["reasoning"]["router_cheap_share"] + 0.4


def test_the_threshold_is_chosen_on_validation_not_test(ds: Dataset, artifact: Any) -> None:
    scores = artifact.score_matrix(ds.embeddings, ds.scalars)
    chosen = evaluate.select_threshold(ds, scores)["threshold"]
    assert chosen == artifact.threshold  # the artifact stores the validation-chosen value


def test_the_headline_separates_truth_judge_and_ppi(metrics: dict[str, Any]) -> None:
    h = metrics["test"]["headline"]
    assert h["quality_basis"] == "ground_truth"
    assert "ppi" in h and h["ppi"]["n_labeled"] == 150
    # The judge misses real losses, so judge-only is optimistic or equal vs the truth here.
    assert h["judge_only_quality"] <= h["quality"] + 0.02
    # PPI's interval should contain the known truth.
    assert h["ppi"]["ci_low"] - 0.02 <= h["quality"] <= h["ppi"]["ci_high"] + 0.02


def test_the_committed_baseline_equals_a_fresh_evaluation(metrics: dict[str, Any]) -> None:
    committed = json.loads((SIM / "metrics.json").read_text(encoding="utf-8"))
    fresh_h, old_h = evaluate.headline_numbers(metrics), evaluate.headline_numbers(committed)
    assert fresh_h == old_h  # otherwise: `make eval-router` and commit metrics.json


def test_the_gate_flags_a_regression_in_either_direction_it_protects() -> None:
    base = {"threshold": 0.8, "cheap_share": 0.4, "quality_lower_95": 0.97, "auc": 0.8}
    assert evaluate.compare(base, dict(base)) == []
    assert evaluate.compare(base, {**base, "cheap_share": 0.35})  # saves less
    assert evaluate.compare(base, {**base, "quality_lower_95": 0.95})  # less safe
    assert evaluate.compare(base, {**base, "auc": 0.7})  # worse ranking
    assert evaluate.compare(base, {**base, "threshold": 0.7})  # a decision changed
    assert evaluate.compare(base, {**base, "cheap_share": 0.5}) == []  # improvements pass


# ---------------------------------------------------------------- artifact vs code
def test_retraining_reproduces_the_committed_artifacts_predictions(
    ds: Dataset, artifact: Any
) -> None:
    retrained, _ = train.train(ds, version=artifact.version)
    committed = artifact.score_matrix(ds.embeddings, ds.scalars)
    fresh = retrained.score_matrix(ds.embeddings, ds.scalars)
    agree = ((committed >= artifact.threshold) == (fresh >= artifact.threshold)).mean()
    assert agree >= 0.995
    np.testing.assert_allclose(fresh, committed, atol=5e-3)


def test_the_artifact_matches_the_dataset_it_claims_to_be_trained_on(artifact: Any) -> None:
    manifest = json.loads((SIM / "manifest.json").read_text(encoding="utf-8"))
    assert artifact.trained_on["dataset_sha256"] == manifest["sha256"]
    assert artifact.embedding_model == manifest["embedding_model"]


def test_the_shipped_route_is_not_set_more_aggressive_than_the_measured_threshold(
    artifact: Any,
) -> None:
    spec = load_config(None).routes["default"].router
    assert spec is not None
    assert spec.threshold >= artifact.threshold  # lower = sends more cheap = less safe
    assert spec.artifact.endswith("sim-world.json")
    assert Path(__file__).parents[2].joinpath("relay", "config", spec.artifact).exists()


# ---------------------------------------------------------------- the real-data dataset
REAL = data_mod.HERE / "real"


@pytest.fixture(scope="module")
def real() -> Dataset:
    return data_mod.load("real")


def test_the_real_dataset_matches_its_manifest_and_is_labelled_uncalibrated() -> None:
    manifest = json.loads((REAL / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["sha256"].items():
        data = (REAL / name).read_bytes()
        if not name.endswith(".npy"):
            data = data.replace(b"\r\n", b"\n")
        assert hashlib.sha256(data).hexdigest() == digest, name
    assert "UNCALIBRATED" in manifest["world"]  # the label every consumer sees
    assert manifest["judge_model"] != manifest["reference_model"]  # a different family judged
    assert manifest["human_labels"] == 0 and manifest["abstentions_dropped"] == 0


def test_the_real_dataset_has_no_ground_truth_and_no_cross_split_leakage(real: Dataset) -> None:
    assert not real.has_truth and not real.gold.any()
    for key in ("text", "cluster"):
        seen: dict[Any, set[str]] = {}
        for r in real.rows:
            seen.setdefault(r[key], set()).add(r["split"])
        assert not [v for v, s in seen.items() if len(s) > 1], key


def test_the_real_report_is_computed_on_judge_labels_and_says_so() -> None:
    metrics = json.loads((REAL / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["test"]["headline"]["quality_basis"] == "judge_only"
    assert "ppi" not in metrics["test"]["headline"]  # no gold labels, so no PPI
    assert "Real data, uncalibrated judge" in (REAL / "report.md").read_text(encoding="utf-8")


def test_the_real_baseline_equals_a_fresh_evaluation(real: Dataset) -> None:
    fresh = evaluate.headline_numbers(
        evaluate.evaluate_world(real, load_artifact(evaluate.artifact_path("real")))
    )
    committed = evaluate.headline_numbers(
        json.loads((REAL / "metrics.json").read_text(encoding="utf-8"))
    )
    assert fresh == committed


def test_the_real_router_is_not_wired_into_any_shipped_route() -> None:
    # No route may use an artifact whose judge is uncalibrated (ADR 0017 / PRD §10.3).
    for name, route in load_config(None).routes.items():
        if route.router is not None:
            assert "real" not in route.router.artifact, name
