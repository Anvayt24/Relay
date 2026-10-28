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

import numpy as np
from scipy.stats import beta, norm


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


# ---------------------------------------------------------------- prediction-powered inference
MIN_PPI_LABELED = 50  # PRD Appendix A guard: no rectifier from fewer than 50 human labels


@dataclass(frozen=True)
class PPIResult:
    estimate: float
    ci_low: float
    ci_high: float
    lam: float  # the power-tuning weight actually used
    se: float
    n_labeled: int
    n_unlabeled: int
    human_only_estimate: float
    human_only_ci: tuple[float, float]
    ess_gain: float  # how many times more human labels the PPI interval is "worth"

    @property
    def width(self) -> float:
        return self.ci_high - self.ci_low

    @property
    def human_only_width(self) -> float:
        return self.human_only_ci[1] - self.human_only_ci[0]

    def as_dict(self) -> dict[str, float | int | list[float]]:
        return {
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "lam": self.lam,
            "se": self.se,
            "n_labeled": self.n_labeled,
            "n_unlabeled": self.n_unlabeled,
            "human_only_estimate": self.human_only_estimate,
            "human_only_ci": list(self.human_only_ci),
            "ess_gain": self.ess_gain,
        }


def ppi_mean_ci(
    judge_unlabeled: "np.ndarray",
    human_labeled: "np.ndarray",
    judge_labeled: "np.ndarray",
    *,
    alpha: float = 0.05,
) -> PPIResult:
    """Prediction-powered estimate of a proportion, with a valid confidence interval (PPI++).

    The problem: you want the win-or-tie rate ``w``. Humans give the truth but are scarce
    (n labels); a judge is cheap and covers many items (N) but is biased. Using either alone is
    bad: humans alone give a wide interval, the judge alone gives a confident wrong number.

    The estimate corrects the judge with the humans:

        theta = lambda * mean(f over the N judge-only items)
              + mean(y - lambda * f over the n human-labeled items)

    The second term is the **rectifier**: how wrong the judge is on items where the truth is
    known. If the judge's bias is the same on the N unlabeled items, it cancels, so the
    estimate is unbiased even though the judge is not.

    ``lambda`` (power tuning) decides how much to trust the judge:
    ``clip(Cov(y,f) / ((1 + n/N) Var(f)), 0, 1)``, estimated on the labeled set. 1 is classic
    PPI; 0 ignores the judge and reduces to the human-only mean. Tuned this way, a poor judge
    can't make the interval worse than human-only (plain PPI can).

    Variance: ``lambda^2 Var(f)/N + Var(y - lambda f)/n``; the interval is
    ``theta +/- z * sqrt(variance)`` (clipped to [0, 1]).

    Requirements and guards (Appendix A):
    - the labeled and judge-only sets must be **disjoint** random samples of the same
      population; otherwise the two means correlate and the variance is wrong (caller's job);
    - **n >= 50**: fewer labels can't support a rectifier;
    - **degenerate variance**: with a judge that matches the humans on every labeled item the
      rectifier's sample variance is 0 and the interval would collapse falsely. Its variance is
      therefore computed with two pseudo-observations, one disagreement in each direction
      (human 1 / judge 0, and human 0 / judge 1), which keeps the interval honestly open.
    """
    f_unl = np.asarray(judge_unlabeled, dtype=float)
    y = np.asarray(human_labeled, dtype=float)
    f_lab = np.asarray(judge_labeled, dtype=float)
    if y.shape != f_lab.shape or y.ndim != 1:
        raise ValueError("human and judge labels on the labeled set must be paired 1-d arrays")
    n, big_n = len(y), len(f_unl)
    if n < MIN_PPI_LABELED:
        raise ValueError(f"PPI needs at least {MIN_PPI_LABELED} human labels, got {n}")
    if big_n == 0:
        raise ValueError("PPI needs judge-only items")

    var_f = float(np.var(f_lab, ddof=1))
    if var_f == 0.0:
        lam = 0.0  # a constant judge carries no information about which items are good
    else:
        cov = float(np.cov(y, f_lab, ddof=1)[0, 1])
        lam = float(np.clip(cov / ((1 + n / big_n) * var_f), 0.0, 1.0))

    residual = y - lam * f_lab
    theta = lam * float(f_unl.mean()) + float(residual.mean())

    # Pseudo-observations for the rectifier's variance only: (y=1, f=0) and (y=0, f=1).
    padded = np.concatenate([residual, [1.0, -lam]])
    var_rect = float(np.var(padded, ddof=1)) / n
    var_judge = lam**2 * (float(np.var(f_unl, ddof=1)) if big_n > 1 else 0.0) / big_n
    se = math.sqrt(var_judge + var_rect)

    z = float(norm.ppf(1 - alpha / 2))
    lo, hi = max(0.0, theta - z * se), min(1.0, theta + z * se)

    # Human-only baseline, with the same pseudo-count so the comparison is like for like.
    y_padded = np.concatenate([y, [1.0, 0.0]])
    se_h = math.sqrt(float(np.var(y_padded, ddof=1)) / n)
    h_est = float(y.mean())
    h_ci = (max(0.0, h_est - z * se_h), min(1.0, h_est + z * se_h))
    gain = (se_h**2) / (se**2) if se > 0 else math.nan

    return PPIResult(
        estimate=min(1.0, max(0.0, theta)),
        ci_low=lo,
        ci_high=hi,
        lam=lam,
        se=se,
        n_labeled=n,
        n_unlabeled=big_n,
        human_only_estimate=h_est,
        human_only_ci=h_ci,
        ess_gain=gain,
    )
