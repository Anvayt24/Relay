"""Semantic-cache evaluation (PRD E-B, M3): how safe is each similarity threshold?

    python -m evals.cache.evaluate              # write metrics.json, report.md, pr_curve.png
    python -m evals.cache.evaluate --check      # fail if the result regressed vs metrics.json

Reads only the frozen files from `build.py` (pairs, prompts, embeddings): no model, no
network, fully deterministic, so CI runs it on every pull request.

**What is measured.** A pair is (a cached prompt, a new prompt, label). A cache *hit* is
predicted when cosine similarity >= threshold (and, with the lexical guard on, the numbers and
names in the two prompts match). Then:
- **precision** = of the hits, the share that were truly acceptable. Wrong hits serve a wrong
  answer, so this is the number the cache is tuned for (target >= 99%, PRD G4/M3);
- **hit rate** = hits / all pairs. It reflects how this set was built (roughly 1 in 3 pairs is
  a true paraphrase), *not* production traffic: the report says so.

**Protocol (no peeking).** The threshold is chosen on the *tune* split only: the smallest one
with precision >= 99% there (and a minimum number of hits). It is then *reported* on the *test*
split, which the choice never saw, with a one-sided 95% Clopper-Pearson lower bound on
precision and a bootstrap interval on the hit rate. The split is by fact (see `build.py`), so
no prompt appears on both sides.

Both configurations are reported: without the lexical guard, and with it (the production
setting). The guard exists because embeddings call "15% of 200" and "15% of 300" near-identical.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from relay.cache.keys import guard_signature
from relay.evidence.stats import clopper_pearson_lower

HERE = Path(__file__).parent
TARGET_PRECISION = 0.99
MIN_HITS = 30  # a "99% precision" from 10 hits is not a measurement
THRESHOLDS = [round(0.50 + 0.0025 * i, 4) for i in range(201)]  # 0.50 .. 1.00
BOOTSTRAP = 2000
SEED = 0


def load() -> list[dict[str, Any]]:
    prompts: list[str] = json.loads((HERE / "prompts.json").read_text(encoding="utf-8"))
    vectors = np.load(HERE / "embeddings.npy")
    index = {p: i for i, p in enumerate(prompts)}
    rows = []
    for line in (HERE / "pairs.jsonl").read_text(encoding="utf-8").splitlines():
        pair = json.loads(line)
        sim = float(vectors[index[pair["a"]]] @ vectors[index[pair["b"]]])
        pair["sim"] = sim
        pair["guard_ok"] = guard_signature(pair["a"]) == guard_signature(pair["b"])
        rows.append(pair)
    return rows


def hits_at(rows: list[dict[str, Any]], t: float, guard: bool) -> list[dict[str, Any]]:
    return [r for r in rows if r["sim"] >= t and (r["guard_ok"] or not guard)]


def curve(rows: list[dict[str, Any]], guard: bool) -> list[dict[str, float]]:
    positives = sum(r["label"] for r in rows)
    out = []
    for t in THRESHOLDS:
        hits = hits_at(rows, t, guard)
        tp = sum(r["label"] for r in hits)
        out.append(
            {
                "threshold": t,
                "hits": len(hits),
                "precision": tp / len(hits) if hits else math.nan,
                "recall": tp / positives if positives else math.nan,
                "hit_rate": len(hits) / len(rows),
            }
        )
    return out


def choose_threshold(tune_curve: list[dict[str, float]]) -> float | None:
    for point in tune_curve:  # ascending thresholds: the first that qualifies is the smallest
        if point["hits"] >= MIN_HITS and point["precision"] >= TARGET_PRECISION:
            return point["threshold"]
    return None


def bootstrap_hit_rate(rows: list[dict[str, Any]], t: float, guard: bool) -> list[float]:
    rng = random.Random(SEED)
    flags = [r["sim"] >= t and (r["guard_ok"] or not guard) for r in rows]
    n = len(flags)
    rates = sorted(sum(flags[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP))
    return [rates[int(0.025 * BOOTSTRAP)], rates[int(0.975 * BOOTSTRAP) - 1]]


def evaluate_config(tune: list[dict[str, Any]], test: list[dict[str, Any]], guard: bool) -> dict:
    tune_curve = curve(tune, guard)
    t = choose_threshold(tune_curve)
    result: dict[str, Any] = {"lexical_guard": guard, "threshold": t, "tune_curve": tune_curve}
    if t is None:
        result["test"] = None
        return result
    hits = hits_at(test, t, guard)
    tp = sum(r["label"] for r in hits)
    wrong = [r for r in hits if not r["label"]]
    positives = sum(r["label"] for r in test)
    result["tune"] = _row(next(p for p in tune_curve if p["threshold"] == t))
    result["test"] = {
        "pairs": len(test),
        "hits": len(hits),
        "true_hits": tp,
        "precision": tp / len(hits) if hits else None,
        "precision_lower_95": clopper_pearson_lower(tp, len(hits)) if hits else None,
        "hit_rate": len(hits) / len(test),
        "hit_rate_ci_95": bootstrap_hit_rate(test, t, guard),
        "recall_of_positives": tp / positives,
        "wrong_hits_by_kind": {
            kind: sum(r["kind"] == kind for r in wrong)
            for kind in sorted({r["kind"] for r in wrong})
        },
    }
    return result


def _row(point: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in point.items() if k != "threshold"}


def run() -> dict[str, Any]:
    rows = load()
    tune = [r for r in rows if r["split"] == "tune"]
    test = [r for r in rows if r["split"] == "test"]
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    return {
        "what": "semantic cache precision and hit rate by similarity threshold (simulated labels)",
        "dataset": {
            "pairs": len(rows),
            "tune_pairs": len(tune),
            "test_pairs": len(test),
            "positive_share": sum(r["label"] for r in rows) / len(rows),
            "embedding_model": manifest["embedding_model"],
            "label_note": manifest["label"],
            "sha256": manifest["sha256"],
        },
        "target_precision": TARGET_PRECISION,
        "guard_off": evaluate_config(tune, test, False),
        "guard_on": evaluate_config(tune, test, True),
    }


def headline(metrics: dict[str, Any]) -> dict[str, float]:
    """The numbers the CI gate protects: the production configuration's test result."""
    on = metrics["guard_on"]
    test = on["test"] or {}
    return {
        "threshold": on["threshold"] if on["threshold"] is not None else math.inf,
        "precision_lower_95": test.get("precision_lower_95") or 0.0,
        "hit_rate": test.get("hit_rate") or 0.0,
    }


def wrong_kinds(cfg: dict[str, Any]) -> Any:
    return cfg["test"]["wrong_hits_by_kind"] if cfg["test"] else "-"


def write_report(metrics: dict[str, Any]) -> None:
    def row(cfg: dict[str, Any]) -> str:
        t = cfg["test"]
        name = "on" if cfg["lexical_guard"] else "off"
        if t is None:
            return f"| {name} | none reaches 99% on tune | | | | |"
        lo, hi = t["hit_rate_ci_95"]
        return (
            f"| {name} | {cfg['threshold']:.4f} | {t['precision'] * 100:.2f}% "
            f"(lower 95%: {t['precision_lower_95'] * 100:.2f}%) | {t['hit_rate'] * 100:.1f}% "
            f"({lo * 100:.1f}-{hi * 100:.1f}) | {t['recall_of_positives'] * 100:.1f}% |"
        )

    d = metrics["dataset"]
    lines = [
        "# Semantic cache evaluation",
        "",
        f"{d['pairs']} labeled pairs ({d['tune_pairs']} tune / {d['test_pairs']} test), "
        f"embedding model `{d['embedding_model']}`. Labels are **constructed** (same fact in a "
        "different wording = acceptable; one fact changed or unrelated = not), not "
        "human-judged. The hit rate reflects how the set was built (about "
        f"{d['positive_share'] * 100:.0f}% of pairs are true paraphrases), not production "
        "traffic.",
        "",
        "Threshold chosen on **tune** as the smallest with precision >= 99%; everything below is "
        "**test**, which that choice never saw.",
        "",
        "| lexical guard | threshold | test precision | test hit rate (95% CI) | recall |",
        "|---|---|---|---|---|",
        row(metrics["guard_off"]),
        row(metrics["guard_on"]),
        "",
        "The production configuration is the guard **on**. Wrong hits on test, by kind: "
        f"guard off {wrong_kinds(metrics['guard_off'])}, "
        f"guard on {wrong_kinds(metrics['guard_on'])}.",
        "",
    ]
    (HERE / "report.md").write_text("\n".join(lines), encoding="utf-8")


def plot(metrics: dict[str, Any]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for key, label in (("guard_off", "guard off"), ("guard_on", "guard on")):
        c = [p for p in metrics[key]["tune_curve"] if p["hits"] >= MIN_HITS]
        ax.plot([p["recall"] for p in c], [p["precision"] for p in c], label=label)
    ax.axhline(TARGET_PRECISION, color="grey", ls="--", lw=1)
    ax.set_xlabel("recall of true paraphrases (tune)")
    ax.set_ylabel("precision (tune)")
    ax.set_title("Cache precision vs recall as the threshold varies")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "pr_curve.png", dpi=130)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.cache.evaluate", description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare with the committed metrics")
    args = parser.parse_args()
    metrics = run()
    path = HERE / "metrics.json"
    if args.check:
        if not path.exists():
            print("evals/cache/metrics.json is missing: run `make eval-cache` and commit it")
            return 1
        old, new = headline(json.loads(path.read_text(encoding="utf-8"))), headline(metrics)
        problems = []
        if new["precision_lower_95"] < old["precision_lower_95"] - 1e-9:
            problems.append(
                f"precision lower bound {old['precision_lower_95']:.4f} "
                f"-> {new['precision_lower_95']:.4f}"
            )
        if new["hit_rate"] < old["hit_rate"] - 1e-9:
            problems.append(f"hit rate {old['hit_rate']:.4f} -> {new['hit_rate']:.4f}")
        if new["threshold"] != old["threshold"]:
            problems.append(f"threshold {old['threshold']} -> {new['threshold']}")
        if problems:
            print("cache eval regressed vs the committed baseline: " + "; ".join(problems))
            print("If intended, run `make eval-cache` and commit the new metrics.json.")
            return 1
        print(f"cache eval OK: {new}")
        return 0
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(metrics)
    plotted = plot(metrics)
    print(json.dumps(headline(metrics), indent=2))
    print("wrote metrics.json, report.md" + (", pr_curve.png" if plotted else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
