"""Judge calibration: how far can this judge be trusted? (PRD §10.3, M7)

    python -m relay.evidence.calibrate --pairs pairs.jsonl --labels labels.jsonl \\
        --out calibration.json

Joins each judged pair with the human's label and reports, on the **binary** label
(loss vs win-or-tie): the confusion matrix, Cohen's kappa, accuracy, and per-class recall,
each with a seeded bootstrap interval. **Loss recall** is the one that matters most: a judge
that misses real losses makes a bad route look safe.

**The gate (M7).** The judge may drive decisions only if kappa >= 0.6 AND loss recall >= 0.7.
Otherwise the status is ``uncalibrated`` and route cards must say so. With too few labeled
pairs the status is ``insufficient_n``: a verdict from 30 pairs isn't one. (The PRD's PPI
guard asks n >= 50 per rectifier scope; the same floor is used here.)

Also reported, because they qualify the numbers:
- **kappa paradox:** losses are rare, so kappa can be low while raw accuracy is high. All
  three are printed.
- **annotator self-consistency:** kappa between your label and your blind re-label of the
  same pair. The judge can't agree with you better than you agree with yourself.
- **position bias:** the judge's loss rate by which side the candidate was shown on. A gap
  means the order randomization isn't cancelling a real bias.
- **w_self** (``--self-pairs``): for pairs of the reference model against itself, the share
  judged win-or-tie. It is the ceiling on ``w`` any candidate can reach; if it's near the
  quality floor (0.90), no route can pass and the rubric or the floor needs fixing.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

from relay.evidence import stats
from relay.evidence.pairs import EvalPair, file_sha256, read_jsonl

KAPPA_MIN = 0.6
LOSS_RECALL_MIN = 0.7
MIN_N = 50


def read_labels(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def calibrate(
    pairs: list[EvalPair], labels: list[dict[str, Any]], *, seed: int = 0, n_boot: int = 2000
) -> dict[str, Any]:
    by_id = {p.id: p for p in pairs}
    first: dict[str, str] = {}
    relabel: dict[str, str] = {}
    for row in labels:
        (relabel if row["is_relabel"] else first).setdefault(row["pair_id"], row["label"])

    human: list[bool] = []
    judge: list[bool] = []
    abstained = 0
    for pair_id, label in first.items():
        p = by_id.get(pair_id)
        if p is None:
            continue
        if p.judge_loss is None:
            abstained += 1
            continue
        human.append(label == "loss")
        judge.append(p.judge_loss)

    c = stats.confusion(human, judge)
    kappa = stats.cohen_kappa(c)
    recall = stats.loss_recall(c)
    out: dict[str, Any] = {
        "n": c.n,
        "abstained": abstained,
        "confusion": {"tp": c.tp, "fn": c.fn, "fp": c.fp, "tn": c.tn},
        "human_loss_rate": (c.tp + c.fn) / c.n if c.n else math.nan,
        "judge_loss_rate": (c.tp + c.fp) / c.n if c.n else math.nan,
        "kappa": _with_ci(kappa, human, judge, stats.cohen_kappa, seed, n_boot),
        "accuracy": _with_ci(stats.accuracy(c), human, judge, stats.accuracy, seed, n_boot),
        "loss_recall": _with_ci(recall, human, judge, stats.loss_recall, seed, n_boot),
        "win_recall": _with_ci(stats.win_recall(c), human, judge, stats.win_recall, seed, n_boot),
        "annotator_self_kappa": _self_kappa(first, relabel),
        "position_bias": _position_bias(pairs),
        "judge_models": sorted({p.judge_model for p in pairs}),
        "judge_prompt_versions": sorted({p.judge_prompt_version for p in pairs}),
        "thresholds": {"kappa_min": KAPPA_MIN, "loss_recall_min": LOSS_RECALL_MIN, "n_min": MIN_N},
        "note": "kappa is low when losses are rare even if accuracy is high (kappa paradox).",
    }
    out["status"] = _status(c.n, kappa, recall)
    return out


def _status(n: int, kappa: float, recall: float) -> str:
    if n < MIN_N:
        return "insufficient_n"
    ok = not math.isnan(kappa) and not math.isnan(recall)
    return (
        "calibrated" if ok and kappa >= KAPPA_MIN and recall >= LOSS_RECALL_MIN else "uncalibrated"
    )


def _with_ci(
    value: float, human: list[bool], judge: list[bool], fn: Any, seed: int, n_boot: int
) -> dict[str, float]:
    if not human:
        return {"value": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    lo, hi = stats.bootstrap_ci(human, judge, fn, n_boot=n_boot, seed=seed)
    return {"value": value, "ci_low": lo, "ci_high": hi}


def _self_kappa(first: dict[str, str], relabel: dict[str, str]) -> dict[str, float]:
    shared = [k for k in relabel if k in first]
    if len(shared) < 2:
        return {"n": len(shared), "kappa": math.nan}
    a = [first[k] == "loss" for k in shared]
    b = [relabel[k] == "loss" for k in shared]
    return {"n": len(shared), "kappa": stats.cohen_kappa(stats.confusion(a, b))}


def _position_bias(pairs: list[EvalPair]) -> dict[str, float]:
    out = {}
    for swapped in (False, True):
        group = [
            p.judge_loss
            for p in pairs
            if p.position_swapped == swapped and p.judge_loss is not None
        ]
        out["candidate_as_B" if swapped else "candidate_as_A"] = (
            sum(group) / len(group) if group else math.nan
        )
    return out


def w_self(pairs: list[EvalPair]) -> dict[str, float]:
    judged = [p for p in pairs if p.judge_loss is not None]
    if not judged:
        return {"n": 0, "w_self": math.nan}
    return {"n": len(judged), "w_self": sum(not p.judge_loss for p in judged) / len(judged)}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m relay.evidence.calibrate", description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--labels", help="human labels JSONL (omit with --self-pairs)")
    parser.add_argument("--self-pairs", action="store_true", help="report w_self instead")
    parser.add_argument("--out")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    pairs_path = Path(args.pairs)
    pairs = read_jsonl(pairs_path)
    if args.self_pairs:
        report: dict[str, Any] = w_self(pairs)
    else:
        if not args.labels:
            raise SystemExit("--labels is required (or use --self-pairs)")
        report = calibrate(pairs, read_labels(Path(args.labels)), seed=args.seed)
        report["pairs_sha256"] = file_sha256(pairs_path)
    text = json.dumps(report, indent=2, allow_nan=True)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
