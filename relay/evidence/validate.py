"""Validate the calibration machinery against a judge whose accuracy is *known*.

    python -m relay.evidence.validate --out evals/judge/sim_validation.json     (make eval-judge)

Real calibration can't tell you whether the calibration code is right: you don't know the
judge's true accuracy, that's what you're measuring. The simulator can. It decides, per
request, whether each model's answer is secretly correct, so the true outcome of every pair
(loss, win or tie) is known exactly, and its ``MockJudge`` reports that truth through a
configured error rate (a sensitivity: how often it catches a true loss; a false-loss rate).

So: build a few hundred pairs from a scenario, give the judge a known sensitivity, let a
"perfect annotator" (the truth) play the human, run ``calibrate``, and check that

1. the recovered **loss recall** matches the configured sensitivity (inside its interval);
2. the **gate** flips where it should: a good judge is ``calibrated``, a judge that misses
   half the losses is ``uncalibrated``.

This is the PRD §10.9 / M7 machinery tested end to end, with ground truth (it is a
simulation, and says so).
"""

import argparse
import json
from pathlib import Path
from typing import Any

from relay.evidence.calibrate import calibrate
from relay.evidence.pairs import EvalPair, prompt_hash
from relay.simulator.judge import JudgeParams, MockJudge
from relay.simulator.quality import WinRate, outcome
from relay.simulator.scenario import CHEAP, STRONG, ModelProfile, Scenario

# (name, sensitivity, false_loss_rate): a good judge, a mediocre one, a loss-blind one.
JUDGES = (
    ("good", 0.90, 0.03),
    ("mediocre", 0.75, 0.10),
    ("loss_blind", 0.45, 0.02),
)


def scenario(w: float) -> Scenario:
    from relay.simulator.quality import QualitySpec

    return Scenario(
        models={
            STRONG: ModelProfile(quality=QualitySpec(base=0.95)),
            CHEAP: ModelProfile(quality=QualitySpec(base=WinRate(reference=STRONG, w=w))),
        }
    )


def simulate(
    name: str, sensitivity: float, false_loss: float, *, n: int, w: float, seed: int
) -> dict[str, Any]:
    scn = scenario(w)
    judge = MockJudge(seed, JudgeParams(sensitivity=sensitivity, false_loss_rate=false_loss))
    pairs: list[EvalPair] = []
    labels: list[dict[str, Any]] = []
    true_losses = 0
    for i in range(n):
        cand = scn.correct(seed, CHEAP, "default", i)
        ref = scn.correct(seed, STRONG, "default", i)
        truth = outcome(cand, ref)
        true_losses += truth == "loss"
        verdict = judge.verdict(cand_correct=cand, ref_correct=ref, index=i, candidate=CHEAP)
        pair_id = f"sim-{i:04d}"
        pairs.append(
            EvalPair(
                id=pair_id,
                prompt_hash=prompt_hash(pair_id),
                prompt="",
                cohort="lang=latin,len=short,code=false,tools=false",
                candidate_model=CHEAP,
                reference_model=STRONG,
                candidate_response="",
                reference_response="",
                judge_label="loss" if verdict == "loss" else "tie",
                judge_model=f"mock-judge-{name}",
                judge_prompt_version="sim",
                position_swapped=i % 2 == 0,
                text_retained=False,
                ts="2026-01-01T00:00:00+00:00",
                source="sim",
            )
        )
        labels.append(
            {
                "pair_id": pair_id,
                "label": "loss" if truth == "loss" else "tie",
                "annotator": "truth",
                "is_relabel": False,
            }
        )
    report = calibrate(pairs, labels, seed=seed)
    recall = report["loss_recall"]
    return {
        "judge": name,
        "configured_sensitivity": sensitivity,
        "configured_false_loss_rate": false_loss,
        "n": n,
        "true_loss_rate": true_losses / n,
        "recovered_loss_recall": recall,
        "recovered_kappa": report["kappa"],
        "recall_ci_contains_configured": recall["ci_low"] <= sensitivity <= recall["ci_high"],
        "status": report["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m relay.evidence.validate", description=__doc__)
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument(
        "--w", type=float, default=0.80, help="true win-or-tie rate of the candidate"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args()
    results = [simulate(name, s, f, n=args.n, w=args.w, seed=args.seed) for name, s, f in JUDGES]
    report = {
        "what": "simulator validation of the judge calibration (ground truth known)",
        "results": results,
    }
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    for r in results:
        rc = r["recovered_loss_recall"]
        print(
            f"{r['judge']:10s} configured {r['configured_sensitivity']:.2f} recovered "
            f"{rc['value']:.2f} [{rc['ci_low']:.2f}, {rc['ci_high']:.2f}] -> {r['status']}"
        )


if __name__ == "__main__":
    main()
