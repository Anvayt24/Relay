"""The judge's calibration on the public pairs: the M7 numbers, per judge and per source.

    python -m evals.judge.public.report     # writes calibration.json and report.md

- **Per judge, never pooled.** If a fallback judge took over part of the run, each judge is
  calibrated on its own pairs only (see run.py).
- **Per source.** MT-Bench votes are preferences between 2023-era models; LLMBar's are
  objective instruction-following verdicts, half of them built to fool judges. A judge can
  pass one and fail the other, and the pooled number would hide it.
- **Human-human agreement.** On MT-Bench pairs voted on by two annotators: how often do two
  humans agree on the binary label (loss vs win-or-tie)? That is the ceiling: a judge can't
  be expected to agree with *a* human more than humans agree with each other.

Calibrated on public data, the result says how far the judge can be trusted on *this* kind
of data. Relay's own traffic (gpt-oss answers, not 2023 models) may differ (PRD §10.4,
stationarity), and the report says so.
"""

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from relay.evidence import stats
from relay.evidence.calibrate import KAPPA_MIN, LOSS_RECALL_MIN, MIN_N, calibrate, read_labels
from relay.evidence.pairs import EvalPair, file_sha256, read_jsonl

HERE = Path(__file__).parent

SOURCES: dict[str, tuple[str, ...]] = {
    "mtbench": ("mtbench",),
    "llmbar": ("llmbar-natural", "llmbar-adversarial"),
    "llmbar-natural": ("llmbar-natural",),
    "llmbar-adversarial": ("llmbar-adversarial",),
}


def human_agreement(
    labels: Sequence[Mapping[str, Any]], extra_votes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Binary agreement (loss vs win-or-tie) between a pair's label and another annotator's
    vote on the same pair: one comparison per double-voted pair."""
    first = {str(row["pair_id"]): str(row["label"]) for row in labels}
    seen: set[str] = set()
    a: list[bool] = []
    b: list[bool] = []
    for vote in extra_votes:
        pair_id = str(vote["pair_id"])
        if pair_id in first and pair_id not in seen:
            seen.add(pair_id)
            a.append(first[pair_id] == "loss")
            b.append(vote["label"] == "loss")
    if not a:
        return {"n": 0, "kappa": math.nan, "accuracy": math.nan}
    c = stats.confusion(a, b)
    return {"n": c.n, "kappa": stats.cohen_kappa(c), "accuracy": stats.accuracy(c)}


def build(
    pairs: Sequence[EvalPair],
    labels: Sequence[Mapping[str, Any]],
    *,
    extra_votes: Sequence[Mapping[str, Any]],
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    label_rows = [dict(row) for row in labels]
    judges: dict[str, Any] = {}
    for judge in sorted({p.judge_model for p in pairs if p.judge_model}):
        mine = [p for p in pairs if p.judge_model == judge]
        entry: dict[str, Any] = {
            "rubric": sorted({p.judge_prompt_version for p in mine}),
            "all": calibrate(mine, label_rows, seed=seed, n_boot=n_boot),
            "by_source": {},
        }
        for name, sources in SOURCES.items():
            subset = [p for p in mine if p.source in sources]
            if subset:
                entry["by_source"][name] = calibrate(subset, label_rows, seed=seed, n_boot=n_boot)
        judges[judge] = entry
    return {
        "what": "judge calibration on public human-labeled pairs (MT-Bench human judgments, "
        "LLMBar); see evals/judge/public/SOURCES.md",
        "pairs": len(pairs),
        "unjudged": sum(not p.judge_model for p in pairs),
        "judges": judges,
        "human_agreement_mtbench": human_agreement(labels, extra_votes),
        "gate": {"kappa_min": KAPPA_MIN, "loss_recall_min": LOSS_RECALL_MIN, "n_min": MIN_N},
        "caveat": "Calibrated on public data (2023-era models and author-built adversarial "
        "pairs), not on Relay's own traffic. MT-Bench votes are preferences, while the rubric "
        "asks whether one answer is materially worse.",
    }


def _num(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Judge calibration on public data",
        "",
        result["caveat"],
        "",
        f"Pairs: {result['pairs']} ({result['unjudged']} not judged). Gate (M7): kappa >= "
        f"{KAPPA_MIN} and loss recall >= {LOSS_RECALL_MIN} on >= {MIN_N} pairs.",
        "",
        "| judge | rubric | data | n | kappa [95% CI] | loss recall [95% CI] | accuracy | status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for judge, entry in result["judges"].items():
        rows = [("all", entry["all"]), *entry["by_source"].items()]
        for name, r in rows:
            k, lr = r["kappa"], r["loss_recall"]
            lines.append(
                f"| {judge} | {'/'.join(entry['rubric'])} | {name} | {r['n']} | "
                f"{_num(k['value'])} [{_num(k['ci_low'])}, "
                f"{_num(k['ci_high'])}] | {_num(lr['value'])} [{_num(lr['ci_low'])}, "
                f"{_num(lr['ci_high'])}] | {_num(r['accuracy']['value'])} | {r['status']} |"
            )
    h = result["human_agreement_mtbench"]
    lines += [
        "",
        f"Human-human agreement (MT-Bench pairs with two votes, binary label): n = {h['n']}, "
        f"kappa = {_num(h['kappa'])}, accuracy = {_num(h['accuracy'])}. A judge is not "
        "expected to beat this.",
    ]
    for judge, entry in result["judges"].items():
        bias = entry["all"]["position_bias"]
        lines.append(f"\nPosition bias ({judge}): {json.dumps(bias)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m evals.judge.public.report")
    parser.add_argument(
        "--runs",
        nargs="*",
        help="judged run files (default: every runs/*.jsonl); one judge per file",
    )
    parser.add_argument("--labels", default=str(HERE / "labels.jsonl"))
    parser.add_argument("--extra", default=str(HERE / "extra_votes.jsonl"))
    args = parser.parse_args()
    runs = [Path(r) for r in args.runs] if args.runs else sorted((HERE / "runs").glob("*.jsonl"))
    pairs = [p for run in runs for p in read_jsonl(run)]
    result = build(pairs, read_labels(Path(args.labels)), extra_votes=read_labels(Path(args.extra)))
    result["sha256"] = {
        "labels": file_sha256(Path(args.labels)),
        "extra_votes": file_sha256(Path(args.extra)),
        **{f"runs/{run.name}": file_sha256(run) for run in runs},
    }
    (HERE / "calibration.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (HERE / "report.md").write_text(markdown(result), encoding="utf-8")
    print(markdown(result))


if __name__ == "__main__":
    main()
