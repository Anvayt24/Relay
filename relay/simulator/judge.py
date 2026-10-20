"""A mock judge with known, tunable errors (PRD §11.1, §10.9).

The real judge (Phase 9) is an LLM that answers "is the candidate *materially worse* than the
reference?" (the §10.1 rubric). It makes mistakes, and the evidence engine has to correct for
them (prediction-powered inference, Appendix A). To test that correction, we need a judge
whose mistakes we control exactly. This is that judge: it looks up the true outcome from the
ground truth and then lies with set probabilities.

The verdict is binary, like the rubric: ``"loss"`` or ``"win_or_tie"``. Its errors are two
numbers, as for any binary classifier:

- ``sensitivity`` = P(says loss | true loss). Below 1, the judge *misses* real losses, the
  dangerous direction, because a degraded route looks safe.
- ``false_loss_rate`` = P(says loss | true win or tie). Above 0, the judge raises false
  alarms, the expensive direction, because a good route looks unsafe.

Different values for the two give an *asymmetric bias*. One "accuracy" number could not
express "this judge misses losses but rarely raises false alarms".

**Drift.** ``drift`` changes the parameters from a given request index on. §10.9 experiment 3
uses it: the judge's bias drifts after a degradation, the human-label rectifier goes stale,
and the experiment shows what that does to the gate.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from relay.simulator.quality import outcome
from relay.simulator.seeding import uniform_for

Verdict = Literal["loss", "win_or_tie"]


@dataclass(frozen=True)
class JudgeParams:
    sensitivity: float = 1.0
    false_loss_rate: float = 0.0

    def __post_init__(self) -> None:
        for name in ("sensitivity", "false_loss_rate"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")


class MockJudge:
    def __init__(
        self,
        seed: int,
        params: JudgeParams | None = None,
        drift: Sequence[tuple[int, JudgeParams]] = (),
    ) -> None:
        self.seed = seed
        self.params = params or JudgeParams()
        self.drift = sorted(drift, key=lambda d: d[0])

    def params_at(self, index: int) -> JudgeParams:
        params = self.params
        for at_index, drifted in self.drift:
            if index < at_index:
                break
            params = drifted
        return params

    def verdict(
        self, *, cand_correct: bool, ref_correct: bool, index: int, candidate: str = ""
    ) -> Verdict:
        # `candidate` keys the draw, so judging two candidates against the same reference on
        # the same request gives independent judging errors.
        params = self.params_at(index)
        is_loss = outcome(cand_correct, ref_correct) == "loss"
        p_says_loss = params.sensitivity if is_loss else params.false_loss_rate
        u = uniform_for(self.seed, index, candidate, "judge")
        return "loss" if u < p_says_loss else "win_or_tie"
