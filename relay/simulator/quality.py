"""Ground-truth quality (PRD §11.1): every simulated response is secretly right or wrong.

Each response carries a hidden attribute ``c ~ Bernoulli(q[model, cohort, t])``. Relay never
sees it. The evidence experiments (Phases 9-12) do: they know the *true* win-or-tie rate, so
they can measure whether Relay's judge, intervals and gates actually find the truth. With
real providers that is impossible, because nobody knows the true answer.

**From c to win / tie / loss.** For a (candidate, reference) pair answering the same request:

    c_cand  c_ref   outcome
      1       0     win
      0       1     loss
      1       1     tie   (both acceptable)
      0       0     tie   (both bad: the candidate is not *materially worse*)

**Experiments are stated in w, so the simulator solves for q.** The PRD's decisions are about
``w`` = P(candidate is not materially worse) = 1 - P(loss). With independent draws,

    P(loss) = P(c_ref = 1) · P(c_cand = 0) = q_ref · (1 - q_cand)
    ⇒ q_cand = 1 - (1 - w) / q_ref

That is ``q_for_w``. It is only feasible when ``q_ref ≥ 1 - w``: if the reference is right
only 5% of the time, a candidate cannot lose 7% of the time, because losing needs a right
reference. Independence between models is an assumption. Real models share "hard prompts",
so their errors correlate; modelling per-prompt difficulty is out of scope.

**Time is the request index, not the wall clock.** A shift "w drops to 0.85 at request 5000"
then lands on exactly the same requests in every run, which is what makes a
detection-delay experiment (PRD M5) reproducible.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from relay.simulator.seeding import uniform_for

Outcome = Literal["win", "tie", "loss"]


@dataclass(frozen=True)
class WinRate:
    """Quality stated as a target win-or-tie rate ``w`` against a reference model."""

    reference: str
    w: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.w <= 1.0:
            raise ValueError(f"w must be in [0, 1], got {self.w}")


# A quality level is either q itself or a w to solve for.
Level = float | WinRate


@dataclass(frozen=True)
class QualitySpec:
    """``q`` for one model, possibly changing at given request indexes (silent degradation)."""

    base: Level = 1.0
    # (at_index, new level), applied from that index on. Sorted at construction.
    shifts: Sequence[tuple[int, Level]] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for level in (self.base, *(level for _, level in self.shifts)):
            if isinstance(level, float | int) and not 0.0 <= level <= 1.0:
                raise ValueError(f"q must be in [0, 1], got {level}")
        # frozen: object.__setattr__ is the sanctioned way to normalize in __post_init__.
        object.__setattr__(self, "shifts", tuple(sorted(self.shifts, key=lambda s: s[0])))

    def level_at(self, index: int) -> Level:
        level = self.base
        for at_index, shifted in self.shifts:
            if index < at_index:
                break
            level = shifted
        return level

    def breakpoints(self) -> list[int]:
        return [at_index for at_index, _ in self.shifts]


def q_for_w(w: float, q_ref: float) -> float:
    if q_ref < 1.0 - w:
        raise ValueError(
            f"infeasible: w={w} needs a reference right at least {1 - w:.3f} of the time, "
            f"but q_ref={q_ref}"
        )
    if q_ref == 0.0:  # then w == 1 and P(loss) = 0 whatever the candidate does
        return 1.0
    return 1.0 - (1.0 - w) / q_ref


def draw_correct(seed: int, model: str, cohort: str, index: int, q: float) -> bool:
    # One uniform per (request, model, cohort), compared against q. Because the uniform does
    # not depend on q, lowering q at a shift flips exactly the requests with u in
    # [q_new, q_old) from right to wrong and leaves every other request unchanged. Before and
    # after a degradation are therefore coupled request by request (common random numbers).
    u = uniform_for(seed, index, model, cohort, "quality")
    return u < q


def outcome(cand_correct: bool, ref_correct: bool) -> Outcome:
    if cand_correct and not ref_correct:
        return "win"
    if ref_correct and not cand_correct:
        return "loss"
    return "tie"
