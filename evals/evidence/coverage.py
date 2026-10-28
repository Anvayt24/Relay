"""M6: do nominal 95% PPI intervals really cover the truth 95% of the time? (PRD §10.9 item 2)

For every cell of a grid (human labels n, judge-only pairs N, true w, judge quality), draw many
independent datasets from the simulator's model, build the interval each way, and count how
often it contains the true ``w``. The PRD's target is >= 93% everywhere, *including* the
regimes where Wald-type intervals are known to struggle: small n and w near 0.95 (few losses,
so the variance estimate is itself noisy).

Two intervals are compared:
- **Wald** (PPI++ with the pseudo-count floor, ``relay.evidence.stats.ppi_mean_ci``);
- **percentile bootstrap** (``ppi_bootstrap_ci``): Appendix A's fallback.

The result picks the interval the cards use (ADR 0019): Wald if it meets the target in every
cell, else the bootstrap if *it* does, else whichever covers more in its worst cell, with the
shortfall reported, never hidden.
"""

from itertools import product
from typing import Any

import numpy as np
from scipy.stats import norm

from evals.evidence.sim import JUDGES, judge_only_ok, labeled_counts
from relay.evidence.stats import ppi_from_counts

TARGET = 0.93


def coverage_cell(
    *, n: int, big_n: int, w: float, judge: str, runs: int, n_boot: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    params = JUDGES[judge]
    cells = labeled_counts(rng, n, w, params, runs)  # (runs, 4)
    k = judge_only_ok(rng, big_n, w, params, runs)  # (runs,)
    z = float(norm.ppf(0.975))

    theta, se, _ = ppi_from_counts(cells[:, 0], cells[:, 1], cells[:, 2], cells[:, 3], k, big_n)
    lo, hi = np.maximum(0.0, theta - z * se), np.minimum(1.0, theta + z * se)
    wald = float(np.mean((lo <= w) & (w <= hi)))

    # Bootstrap every run at once: resample its own empirical cells and its own judge rate.
    probs = cells / n
    boot_cells = rng.multinomial(n, probs, size=(n_boot, runs))  # (B, runs, 4)
    boot_k = rng.binomial(big_n, k / big_n, size=(n_boot, runs))
    reps, _, _ = ppi_from_counts(
        boot_cells[..., 0],
        boot_cells[..., 1],
        boot_cells[..., 2],
        boot_cells[..., 3],
        boot_k,
        big_n,
    )
    b_lo, b_hi = np.quantile(reps, [0.025, 0.975], axis=0)
    boot = float(np.mean((b_lo <= w) & (w <= b_hi)))
    return {
        "n_human": n,
        "n_judge": big_n,
        "w": w,
        "judge": judge,
        "wald": round(wald, 4),
        "bootstrap": round(boot, 4),
        "wald_width": round(float(np.mean(hi - lo)), 4),
        "bootstrap_width": round(float(np.mean(b_hi - b_lo)), 4),
    }


def run(
    *,
    n_humans: tuple[int, ...] = (50, 100, 250),
    n_judges: tuple[int, ...] = (500, 2000),
    ws: tuple[float, ...] = (0.80, 0.90, 0.95),
    judges: tuple[str, ...] = ("good", "mediocre", "loss_blind"),
    runs: int = 1000,
    n_boot: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    cells = [
        coverage_cell(n=n, big_n=big_n, w=w, judge=j, runs=runs, n_boot=n_boot, seed=seed + i)
        for i, (n, big_n, w, j) in enumerate(product(n_humans, n_judges, ws, judges))
    ]
    worst = {m: min(c[m] for c in cells) for m in ("wald", "bootstrap")}
    if worst["wald"] >= TARGET:
        chosen = "wald"
    elif worst["bootstrap"] >= TARGET:
        chosen = "bootstrap"
    else:
        chosen = max(worst, key=lambda m: worst[m])
    small_n_high_w = [c for c in cells if c["n_human"] == min(n_humans) and c["w"] == max(ws)]
    return {
        "target": TARGET,
        "runs_per_cell": runs,
        "bootstrap_resamples": n_boot,
        "cells": cells,
        "worst": worst,
        "cells_below_target": {m: sum(c[m] < TARGET for c in cells) for m in ("wald", "bootstrap")},
        "chosen_interval": chosen,
        "met": worst[chosen] >= TARGET,
        "small_n_high_w_worst": {
            m: min(c[m] for c in small_n_high_w) for m in ("wald", "bootstrap")
        },
    }
