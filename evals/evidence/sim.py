"""Vectorized draws from the simulator's quality-and-judge model (PRD §11.1).

The evidence experiments need hundreds of thousands of judged pairs per configuration, so they
draw them as numpy arrays instead of calling ``relay.simulator`` one request at a time. The
model is the same one:

- each pair's true outcome is win-or-tie with probability ``w`` (the simulator's ``WinRate``),
  a loss otherwise;
- the judge is ``MockJudge``: it says "loss" with probability ``sensitivity`` on a true loss
  and ``false_loss_rate`` on a true win-or-tie;
- the human label is the truth (the simulator's ground truth stands in for a careful human).

Everything is coded 1 = win or tie, so means are estimates of ``w``. A test checks these draws
against ``relay.simulator``'s own ``MockJudge`` rates.
"""

from dataclasses import dataclass

import numpy as np

from relay.simulator.judge import JudgeParams

# The three judges of relay/evidence/validate.py, plus PRD §10.9 item 3's drifting one.
JUDGES: dict[str, JudgeParams] = {
    "good": JudgeParams(sensitivity=0.90, false_loss_rate=0.03),
    "mediocre": JudgeParams(sensitivity=0.75, false_loss_rate=0.10),
    "loss_blind": JudgeParams(sensitivity=0.45, false_loss_rate=0.02),
}


@dataclass(frozen=True)
class CellProbs:
    """P(human, judge) for one pair, 1 = win or tie: (11, 10, 01, 00)."""

    p11: float
    p10: float
    p01: float
    p00: float

    @property
    def judge_ok(self) -> float:
        return self.p11 + self.p01

    def as_array(self) -> np.ndarray:
        return np.array([self.p11, self.p10, self.p01, self.p00])


def cell_probs(w: float, judge: JudgeParams) -> CellProbs:
    return CellProbs(
        p11=w * (1 - judge.false_loss_rate),  # truly fine, judge agrees
        p10=w * judge.false_loss_rate,  # truly fine, judge cries loss
        p01=(1 - w) * (1 - judge.sensitivity),  # a real loss the judge misses
        p00=(1 - w) * judge.sensitivity,  # a real loss the judge catches
    )


def labeled_counts(
    rng: np.random.Generator, n: int, w: float, judge: JudgeParams, size: int
) -> np.ndarray:
    """``size`` labeled samples of ``n`` pairs each, as (size, 4) cell counts."""
    return rng.multinomial(n, cell_probs(w, judge).as_array(), size=size)


def judge_only_ok(
    rng: np.random.Generator, n: int, w: float, judge: JudgeParams, size: int
) -> np.ndarray:
    """How many of ``n`` judge-only pairs the judge called win-or-tie, ``size`` times."""
    return rng.binomial(n, cell_probs(w, judge).judge_ok, size=size)


def judge_stream(rng: np.random.Generator, w: np.ndarray, judge: JudgeParams) -> np.ndarray:
    """The judge's verdict (True = win or tie) on a stream of pairs whose true win-or-tie
    probability is ``w`` (any shape: one entry per pair)."""
    truly_ok = rng.random(w.shape) < w
    says_loss = rng.random(w.shape) < np.where(truly_ok, judge.false_loss_rate, judge.sensitivity)
    return ~says_loss
