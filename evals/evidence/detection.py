"""M5: how fast is a silent regression caught, at a matched false-alarm rate? (PRD §10.9 item 1)

The setting: a model keeps answering with 200s, but its win-or-tie rate drops from 0.95 to 0.85
(below the 0.90 floor). Judged samples arrive one by one. Three monitors watch the stream:

- **none:** never alarms (the status quo without evidence);
- **fixed threshold:** alarm when the judge's win-or-tie rate over the last ``W`` samples falls
  5 points below its baseline (the rate in a warm-up period). The common dashboard alert;
- **E1:** Relay's card at pre-registered looks. Every ``L`` judged samples it computes a PPI
  interval on that block (judge verdicts, corrected by human labels collected before the
  change) and alarms when the upper bound is below the floor, i.e. the card turns ``unsafe``.

**Why E1 here fixes lambda at 1 (classic PPI).** PPI++ tunes lambda on the labeled set,
assuming the labeled and judge-only pairs come from the *same* distribution. Monitoring for a
regression breaks that on purpose: the human labels are from before the change. With few
losses at w = 0.95 the tuned lambda is about 0.3, so the estimate leans on the stale human mean
and barely moves when the judge's rate falls (``e1_ppi_pp`` below shows it). With lambda = 1
the estimate follows the judge, and only the rectifier (the judge's bias, which here changes
little with w) is stale. This is PRD §10.4's stationarity warning, measured.

**Matched false-alarm rate.** A monitor can always be made faster by making it jumpier, so the
comparison is only fair at equal false alarms. Each monitor's knob (the fixed threshold's
window ``W``; E1's z multiplier) is tuned on *null* streams (no change, w = 0.95 throughout) to
the largest false-alarm rate <= 5% over a 5,000-sample horizon. E1 is also reported at its
untuned, pre-registered Bonferroni setting (alpha / number of looks), which is valid by
construction.

**Delay** is counted in judged samples from the change to the alarm, on separate *change*
streams whose change point is uniform in [0, 1000) so neither monitor benefits from alignment
with a block boundary. A stream that never alarms within the horizon is censored at the
horizon; the median handles that as long as most streams alarm.
"""

from typing import Any

import numpy as np
from scipy.stats import norm

from evals.evidence.sim import JUDGES, judge_stream, labeled_counts
from relay.evidence.stats import ppi_from_counts

W_BEFORE, W_AFTER, FLOOR = 0.95, 0.85, 0.90
DROP = 0.05  # the fixed threshold: 5 points below baseline
WARMUP = 1000  # samples used for the fixed threshold's baseline (and human labels exist)
HORIZON = 5000  # monitored samples per stream
FA_TARGET = 0.05
N_HUMAN = 200  # human labels, collected before the change
WINDOWS = tuple(range(50, 2001, 25))
# Negative z is allowed: matching a 5% false-alarm rate can require alarming while the
# estimate is still slightly above the floor (a threshold, no longer a confidence bound).
Z_GRID = tuple(np.round(np.arange(-2.0, 4.01, 0.05), 2))


def _streams(rng: np.random.Generator, n_streams: int, change_at: np.ndarray | None) -> np.ndarray:
    """Judge verdicts (True = win or tie), shape (streams, WARMUP + HORIZON). With
    ``change_at`` (monitored index per stream) w drops from W_BEFORE to W_AFTER there."""
    t = np.arange(WARMUP + HORIZON)[None, :]
    w = np.full((n_streams, WARMUP + HORIZON), W_BEFORE)
    if change_at is not None:
        w = np.where(t >= WARMUP + change_at[:, None], W_AFTER, w)
    return judge_stream(rng, w, JUDGES["good"])


def _first(alarm: np.ndarray) -> np.ndarray:
    """Index of the first True per row, or HORIZON when there is none."""
    hit = alarm.any(axis=1)
    return np.where(hit, alarm.argmax(axis=1), HORIZON)


def fixed_threshold_alarm(verdicts: np.ndarray, window: int) -> np.ndarray:
    """First monitored index at which the rolling judge rate is DROP below the baseline."""
    baseline = verdicts[:, :WARMUP].mean(axis=1, keepdims=True)
    csum = np.concatenate(
        [np.zeros((len(verdicts), 1)), np.cumsum(verdicts, axis=1, dtype=float)], axis=1
    )
    # Rolling mean of the last `window` samples ending at each monitored index.
    end = np.arange(WARMUP, WARMUP + HORIZON) + 1
    rolling = (csum[:, end] - csum[:, end - window]) / window
    return _first(rolling < baseline - DROP)


def e1_alarm(
    verdicts: np.ndarray, labeled: np.ndarray, block: int, z: float, lam: float | None
) -> np.ndarray:
    """First monitored index at which a block's PPI upper bound is below the floor. The alarm
    is raised at the end of the block (a look), never inside it."""
    monitored = verdicts[:, WARMUP:]
    n_blocks = HORIZON // block
    k = monitored[:, : n_blocks * block].reshape(len(verdicts), n_blocks, block).sum(axis=2)
    theta, se, _ = ppi_from_counts(
        labeled[:, 0, None],
        labeled[:, 1, None],
        labeled[:, 2, None],
        labeled[:, 3, None],
        k,
        block,
        lam=lam,
    )
    alarm_block = (theta + z * se) < FLOOR  # (streams, blocks)
    first_block = _first(alarm_block)
    return np.where(first_block < HORIZON, (first_block + 1) * block - 1, HORIZON)


def _delays(alarm_at: np.ndarray, change_at: np.ndarray) -> dict[str, Any]:
    early = alarm_at < change_at  # alarmed before anything changed: a false alarm
    detected = (alarm_at < HORIZON) & ~early
    delay = np.where(detected, alarm_at - change_at, HORIZON - change_at)
    valid = delay[~early]
    return {
        "median_delay": float(np.median(valid)),
        "p25_delay": float(np.percentile(valid, 25)),
        "p75_delay": float(np.percentile(valid, 75)),
        "detected_within_horizon": round(float(np.mean(detected)), 4),
        "false_alarms_before_change": int(early.sum()),
    }


def run(
    *, n_streams: int = 1000, block: int = 250, alpha: float = 0.05, seed: int = 0
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    null = _streams(rng, n_streams, None)
    change_at = rng.integers(0, 1000, size=n_streams)
    changed = _streams(rng, n_streams, change_at)
    # Human labels collected before the change (the rectifier), one set per stream.
    null_labels = labeled_counts(rng, N_HUMAN, W_BEFORE, JUDGES["good"], n_streams)
    change_labels = labeled_counts(rng, N_HUMAN, W_BEFORE, JUDGES["good"], n_streams)

    def fa(alarm_at: np.ndarray) -> float:
        return float(np.mean(alarm_at < HORIZON))

    # Fixed threshold: the smallest window whose false-alarm rate is within target.
    fixed_fa = {w: fa(fixed_threshold_alarm(null, w)) for w in WINDOWS}
    window = next((w for w in WINDOWS if fixed_fa[w] <= FA_TARGET), WINDOWS[-1])
    fixed = {
        "window": window,
        "false_alarm_rate": round(fixed_fa[window], 4),
        **_delays(fixed_threshold_alarm(changed, window), change_at),
    }

    looks = HORIZON // block
    z_bonf = float(norm.ppf(1 - alpha / looks))

    def e1(lam: float | None) -> tuple[dict[str, Any], dict[str, Any]]:
        # Matched: the smallest z (most sensitive) whose false-alarm rate is within target.
        rates = {z: fa(e1_alarm(null, null_labels, block, z, lam)) for z in Z_GRID}
        z_tuned = next((z for z in Z_GRID if rates[z] <= FA_TARGET), Z_GRID[-1])
        matched = {
            "block": block,
            "lambda": "tuned (PPI++)" if lam is None else lam,
            "z": float(z_tuned),
            "false_alarm_rate": round(rates[z_tuned], 4),
            **_delays(e1_alarm(changed, change_labels, block, z_tuned, lam), change_at),
        }
        bonf = {
            "block": block,
            "lambda": "tuned (PPI++)" if lam is None else lam,
            "looks": looks,
            "z": round(z_bonf, 4),
            "false_alarm_rate": round(fa(e1_alarm(null, null_labels, block, z_bonf, lam)), 4),
            **_delays(e1_alarm(changed, change_labels, block, z_bonf, lam), change_at),
        }
        return matched, bonf

    e1_tuned, e1_bonf = e1(1.0)
    pp_tuned, pp_bonf = e1(None)
    none = {
        "false_alarm_rate": 0.0,
        "median_delay": float(HORIZON),
        "detected_within_horizon": 0.0,
    }
    winner = "e1" if e1_tuned["median_delay"] < fixed["median_delay"] else "fixed_threshold"
    return {
        "w_before": W_BEFORE,
        "w_after": W_AFTER,
        "floor": FLOOR,
        "horizon": HORIZON,
        "false_alarm_target": FA_TARGET,
        "streams": n_streams,
        "n_human": N_HUMAN,
        "judge": "good",
        "monitors": {
            "none": none,
            "fixed_threshold": fixed,
            "e1_matched": e1_tuned,
            "e1_bonferroni": e1_bonf,
            "e1_ppi_pp_matched": pp_tuned,
            "e1_ppi_pp_bonferroni": pp_bonf,
        },
        "faster_at_matched_false_alarms": winner,
        "met": winner == "e1",
    }
