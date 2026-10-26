"""The statistics the judge calibration (and later the evidence engine) stand on.

Everything here is a small, pure function over lists of booleans, so each one can be checked
against a hand-worked example.

Labels are **binary** (PRD M7): ``True`` = "loss" (the candidate is materially worse than the
reference), ``False`` = "win or tie". Binary because ``w``, the quantity every decision is
stated in, is the win-or-tie rate; and because a three-way label would need a "tie" that
judges and humans disagree about for reasons that don't matter to ``w``.
"""

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from scipy.stats import beta


@dataclass(frozen=True)
class Confusion:
    """Rows are the human (the reference truth), columns the judge. "Positive" = loss."""

    tp: int  # both say loss
    fn: int  # human says loss, judge misses it
    fp: int  # judge says loss, human says win or tie
    tn: int  # both say win or tie

    @property
    def n(self) -> int:
        return self.tp + self.fn + self.fp + self.tn


def confusion(human: Sequence[bool], judge: Sequence[bool]) -> Confusion:
    if len(human) != len(judge):
        raise ValueError(f"need paired labels, got {len(human)} and {len(judge)}")
    tp = sum(h and j for h, j in zip(human, judge, strict=True))
    fn = sum(h and not j for h, j in zip(human, judge, strict=True))
    fp = sum(not h and j for h, j in zip(human, judge, strict=True))
    tn = sum(not h and not j for h, j in zip(human, judge, strict=True))
    return Confusion(tp, fn, fp, tn)


def accuracy(c: Confusion) -> float:
    return (c.tp + c.tn) / c.n if c.n else math.nan


def loss_recall(c: Confusion) -> float:
    """Of the real losses, how many did the judge catch? The number that matters most: a
    judge that misses losses makes a bad route look safe (PRD M7)."""
    positives = c.tp + c.fn
    return c.tp / positives if positives else math.nan


def win_recall(c: Confusion) -> float:
    negatives = c.tn + c.fp
    return c.tn / negatives if negatives else math.nan


def cohen_kappa(c: Confusion) -> float:
    """Agreement beyond chance: (observed - expected) / (1 - expected).

    The "kappa paradox" (PRD M7): when losses are rare, expected agreement is already high, so
    kappa can be low even with high raw accuracy. Always report accuracy and per-class recall
    next to it. 1.0 = perfect, 0 = no better than chance, negative = worse than chance.
    """
    n = c.n
    if n == 0:
        return math.nan
    observed = (c.tp + c.tn) / n
    human_loss, judge_loss = (c.tp + c.fn) / n, (c.tp + c.fp) / n
    expected = human_loss * judge_loss + (1 - human_loss) * (1 - judge_loss)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else math.nan  # both always say the same single class
    return (observed - expected) / (1 - expected)


def bootstrap_ci(
    human: Sequence[bool],
    judge: Sequence[bool],
    statistic: Callable[[Confusion], float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for ``statistic``: resample the *pairs* with replacement
    (keeping each human label with its judge label), recompute, take the middle 1-alpha.

    Seeded: the same data gives the same interval, so a calibration report is reproducible.
    Resamples where the statistic is undefined (e.g. no losses drawn) are skipped.
    """
    rng = random.Random(seed)
    n = len(human)
    values = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        v = statistic(confusion([human[i] for i in idx], [judge[i] for i in idx]))
        if not math.isnan(v):
            values.append(v)
    if not values:
        return math.nan, math.nan
    values.sort()
    lo = values[int(alpha / 2 * len(values))]
    hi = values[min(len(values) - 1, int((1 - alpha / 2) * len(values)))]
    return lo, hi


def clopper_pearson_lower(successes: int, n: int, confidence: float = 0.95) -> float:
    """One-sided lower confidence bound for a proportion (exact, Clopper-Pearson).

    With zero failures in n trials the bound is ``(1 - confidence) ** (1 / n)``: the "rule of
    three" behind the PRD's cache sample-size note (about 300 clean hits to show >= 99%).
    """
    if n == 0:
        return math.nan
    if successes == 0:
        return 0.0
    return float(beta.ppf(1 - confidence, successes, n - successes + 1))
