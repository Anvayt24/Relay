"""Judge noise sensitivity: what the gate does with and without the PPI correction
(PRD §10.9 item 3).

One decision, many times over: is this candidate's win-or-tie rate above the 0.90 floor? The
gate sees ``N`` judge-only pairs and ``n`` human-labeled pairs and says ``safe`` (lower bound
>= floor), ``unsafe`` (upper bound < floor) or ``undecided``. Two gates:

- **judge-only:** the judge's own win-or-tie rate, with a binomial interval. It ignores the
  judge's errors, so a judge that misses losses makes a bad candidate look safe;
- **PPI:** the same judge, corrected by the human labels (PPI++).

Each is run against a candidate that is truly **bad** (w = 0.86, below the floor: "safe" is
the dangerous error) and truly **good** (w = 0.96: "safe" is the right answer), for each judge.

**Drift.** The last judge is calibrated (good) when the human labels are collected, then drifts
to loss-blind before the judge-only pairs arrive. The rectifier is stale: it corrects for a
bias the judge no longer has. That is the failure the card's ``stale`` state and the
"refresh the human labels after a change" rule (§10.4) exist for, and this experiment shows
its size.
"""

from typing import Any

import numpy as np
from scipy.stats import norm

from evals.evidence.sim import JUDGES, judge_only_ok, labeled_counts
from relay.evidence.stats import ppi_from_counts
from relay.simulator.judge import JudgeParams

FLOOR = 0.90
CASES = {"bad": 0.86, "good": 0.96}


def _states(lo: np.ndarray, hi: np.ndarray) -> dict[str, float]:
    return {
        "safe": round(float(np.mean(lo >= FLOOR)), 4),
        "unsafe": round(float(np.mean(hi < FLOOR)), 4),
        "undecided": round(float(np.mean((lo < FLOOR) & (hi >= FLOOR))), 4),
    }


def gate(
    *,
    w: float,
    label_judge: JudgeParams,
    judge: JudgeParams,
    n: int,
    big_n: int,
    runs: int,
    alpha: float,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    z = float(norm.ppf(1 - alpha))  # one-sided, each side
    cells = labeled_counts(rng, n, w, label_judge, runs)
    k = judge_only_ok(rng, big_n, w, judge, runs)
    theta, se, _ = ppi_from_counts(cells[:, 0], cells[:, 1], cells[:, 2], cells[:, 3], k, big_n)
    p = k / big_n
    se_j = np.sqrt(p * (1 - p) / big_n)
    return {
        "ppi": _states(theta - z * se, theta + z * se),
        "judge_only": _states(p - z * se_j, p + z * se_j),
    }


def run(
    *, n: int = 200, big_n: int = 2000, runs: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    judges: dict[str, tuple[JudgeParams, JudgeParams]] = {
        name: (params, params) for name, params in JUDGES.items()
    }
    judges["drifted"] = (JUDGES["good"], JUDGES["loss_blind"])  # labeled good, then drifted
    out: dict[str, Any] = {"floor": FLOOR, "n_human": n, "n_judge": big_n, "runs": runs}
    for name, (at_labeling, later) in judges.items():
        out[name] = {
            case: gate(
                w=w,
                label_judge=at_labeling,
                judge=later,
                n=n,
                big_n=big_n,
                runs=runs,
                alpha=alpha,
                rng=rng,
            )
            for case, w in CASES.items()
        }
    # The headline: how often a truly bad candidate is declared safe.
    out["false_safe"] = {
        name: {
            "ppi": out[name]["bad"]["ppi"]["safe"],
            "judge_only": out[name]["bad"]["judge_only"]["safe"],
        }
        for name in judges
    }
    return out
