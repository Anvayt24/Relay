"""Pre-registered looks (PRD §10.4; ADR 0019).

Re-checking a fixed-sample interval every time a judged pair arrives inflates the error rate:
given enough peeks, a random walk crosses any fixed line, so a merely-adequate route would
eventually be declared "safe" by chance. E1's answer (without E2's confidence sequences) is
to decide only at a few looks fixed in advance:

- a look happens each time the judged-sample count reaches ``every``, ``2*every``, ...;
- there are at most ``max_looks`` of them;
- each look tests at ``alpha / max_looks`` (Bonferroni), so the chance that *any* look errs
  is at most ``alpha``.

At look *j* the decision uses exactly the first ``j * every`` judged pairs in time order: the
data set was fixed before anyone looked at it, so the test is a genuine pre-registered one,
and replaying a decision gives the same answer.
"""

from dataclasses import dataclass

from relay.evidence.stats import bonferroni_alpha


@dataclass(frozen=True)
class LookSchedule:
    every: int = 500
    max_looks: int = 4
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.every < 1 or self.max_looks < 1:
            raise ValueError("every and max_looks must be at least 1")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")

    @property
    def alpha_per_look(self) -> float:
        return bonferroni_alpha(self.alpha, self.max_looks)

    def look_for(self, n_judged: int) -> int:
        """The latest look reached with ``n_judged`` samples (0 = none yet)."""
        return min(n_judged // self.every, self.max_looks)

    def sample_size(self, look: int) -> int:
        """How many judged pairs the decision at ``look`` uses."""
        return look * self.every

    def is_final(self, look: int) -> bool:
        """After the last look an undecided card stays undecided: running out of looks is
        never a reason to promote."""
        return look >= self.max_looks
