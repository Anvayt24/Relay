"""Scenario configuration: how each simulated model behaves, as frozen dataclasses.

A ``Scenario`` is the whole input of an experiment. Together with a seed, it determines every
logical-time outcome. Everything is validated at construction, so a typo'd probability or an
infeasible quality target fails before any traffic is sent, not halfway through a chaos run.

Two clocks drive a scenario, on purpose:
- **Request index (logical time)** drives quality shifts. These are reproducible exactly.
- **Wall-clock seconds since the first request** drive ``Incident`` windows and capacity
  refill. Chaos is about time ("the provider is down from t=30 s to t=60 s"), and the PRD
  accepts that this part is not deterministic under concurrency. That is why chaos results
  are reported as mean ± spread over ≥ 5 runs (§11.1).
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from relay.schemas.errors import ErrorClass
from relay.simulator.quality import Level, QualitySpec, WinRate, draw_correct, q_for_w

# z-score of the 95th percentile of a standard normal.
Z95 = 1.6448536269514722


def lognormal_params(median: float, p95: float) -> tuple[float, float]:
    """(mu, sigma) of the lognormal with this median and p95.

    Why lognormal: latencies and output lengths are positive and right-skewed. Most requests
    are near the median, and a long tail is far slower. That tail drives p99 and timeouts, so
    a symmetric (normal) model would hide exactly the behavior chaos tests are about.

    Why median and p95 as inputs: those are the numbers dashboards show and people quote.
    For a lognormal, ``median = e^mu`` and ``p95 = e^(mu + 1.645·sigma)``.
    """
    if median <= 0 or p95 < median:
        raise ValueError(f"need 0 < median <= p95, got median={median}, p95={p95}")
    mu = math.log(median)
    return mu, (math.log(p95) - mu) / Z95


@dataclass(frozen=True)
class LatencyModel:
    # Time to first token. 0/0 means instant.
    ttft_median_s: float = 0.0
    ttft_p95_s: float = 0.0
    # Generation speed after the first token. inf means instant; per request it varies
    # uniformly within ±jitter (a fraction).
    tokens_per_s: float = math.inf
    tokens_per_s_jitter: float = 0.0

    def __post_init__(self) -> None:
        if self.ttft_median_s != 0 or self.ttft_p95_s != 0:
            lognormal_params(self.ttft_median_s, self.ttft_p95_s)  # validates
        if self.tokens_per_s <= 0:
            raise ValueError("tokens_per_s must be > 0")
        if not 0.0 <= self.tokens_per_s_jitter < 1.0:
            raise ValueError("tokens_per_s_jitter must be in [0, 1)")


@dataclass(frozen=True)
class OutputLength:
    """Reply length in tokens, lognormal. ``max_tokens`` caps it (finish_reason "length")."""

    median_tokens: float = 20
    p95_tokens: float = 60

    def __post_init__(self) -> None:
        lognormal_params(self.median_tokens, self.p95_tokens)  # validates


@dataclass(frozen=True)
class ErrorMix:
    """Per-request probability of each injected failure. At most one fires per request.

    These are the failures a provider produces *by itself*. Failures caused by load (429s
    from exhausted capacity) come from ``Capacity`` instead, so they respond to how hard the
    simulator is pushed, as a real provider's do.
    """

    rate_limited: float = 0.0
    quota_exhausted: float = 0.0
    overloaded: float = 0.0
    server_error: float = 0.0
    timeout: float = 0.0
    stream_broken: float = 0.0  # cut the connection partway through the reply
    retry_after_s: float = 1.0  # sent with rate_limited
    # How long a "timeout" hangs before failing. It models the adapter's own read timeout:
    # nothing arrives, then the call gives up.
    hang_s: float = 30.0

    def __post_init__(self) -> None:
        for error_class, p in self.weights():
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"{error_class} probability must be in [0, 1], got {p}")
        total = sum(p for _, p in self.weights())
        # A small tolerance, because e.g. 0.7 + 0.2 + 0.1 is 1.0000000000000002 in floats.
        if total > 1.0 + 1e-9:
            raise ValueError(f"error probabilities sum to {total} > 1")

    def weights(self) -> list[tuple[ErrorClass, float]]:
        # A fixed order, so the same uniform draw always maps to the same class.
        return [
            (ErrorClass.RATE_LIMITED, self.rate_limited),
            (ErrorClass.QUOTA_EXHAUSTED, self.quota_exhausted),
            (ErrorClass.OVERLOADED, self.overloaded),
            (ErrorClass.SERVER_ERROR, self.server_error),
            (ErrorClass.TIMEOUT, self.timeout),
            (ErrorClass.STREAM_BROKEN, self.stream_broken),
        ]

    def pick(self, u: float) -> ErrorClass | None:
        """Map one uniform draw in [0, 1) to a class, or None for "no error".

        The interval [0, 1) is cut into consecutive slices, one per class, each as wide as
        its probability. Whichever slice u falls in fires, and past the last slice nothing
        fires. One draw, so two errors can never fire together.
        """
        edge = 0.0
        for error_class, p in self.weights():
            edge += p
            if u < edge:
                return error_class
        return None


NO_ERRORS = ErrorMix()


@dataclass(frozen=True)
class Capacity:
    """Rate limits per simulated API key, refilled continuously (a token bucket).

    ``scope`` mirrors real providers, and it matters for Relay's pause logic in Phase 6:
    - ``"account"``: one budget per key, shared by every model (most providers).
    - ``"model"``: a separate budget per (key, model), as Groq does.
    Pausing the wrong scope either keeps hitting the limit or idles healthy models.
    """

    rpm: int
    tpm: int
    scope: Literal["account", "model"] = "account"
    # Charged against TPM when a request sets no max_tokens. Providers reserve the output
    # cap up front, because they cannot know the real length until generation ends.
    default_output_tokens: int = 1024

    def __post_init__(self) -> None:
        if self.rpm <= 0 or self.tpm <= 0 or self.default_output_tokens <= 0:
            raise ValueError("rpm, tpm and default_output_tokens must be > 0")


@dataclass(frozen=True)
class Incident:
    """A scheduled wall-clock window that overrides behavior: the building block of every
    chaos scenario (PRD §11.2, E-C). Times are seconds since the simulator's first request.

    - Hard outage:     ``errors=ErrorMix(server_error=1.0)``
    - 429 storm:       ``errors=ErrorMix(rate_limited=0.8)``
    - Latency spike:   ``latency_multiplier=5``
    - Mid-stream drop: ``errors=ErrorMix(stream_broken=0.5)``
    Silent degradation is a quality shift by request index (``QualitySpec.shifts``), because
    a model that gets worse while returning 200s is not an incident anyone sees.
    """

    start_s: float
    end_s: float
    models: Sequence[str] = ()  # empty = every model
    errors: ErrorMix | None = None  # replaces the model's own ErrorMix while active
    latency_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.start_s < self.end_s:
            raise ValueError("need 0 <= start_s < end_s")
        if self.latency_multiplier <= 0:
            raise ValueError("latency_multiplier must be > 0")

    def applies(self, model: str, elapsed_s: float) -> bool:
        in_window = self.start_s <= elapsed_s < self.end_s
        return in_window and (not self.models or model in self.models)


@dataclass(frozen=True)
class ModelProfile:
    latency: LatencyModel = field(default_factory=LatencyModel)
    length: OutputLength = field(default_factory=OutputLength)
    errors: ErrorMix = NO_ERRORS
    quality: QualitySpec = field(default_factory=QualitySpec)
    # Per-cohort overrides of `quality`: this is the "cohort" in q[model, cohort, t].
    cohort_quality: Mapping[str, QualitySpec] = field(default_factory=dict)

    def quality_for(self, cohort: str) -> QualitySpec:
        return self.cohort_quality.get(cohort, self.quality)


@dataclass(frozen=True)
class Scenario:
    # Keyed by the *upstream* model id (what an adapter sends), e.g. "mock-strong-v1".
    # A request for any other model gets model_not_found, as from a real provider.
    models: Mapping[str, ModelProfile]
    capacity: Capacity | None = None
    incidents: Sequence[Incident] = ()

    def __post_init__(self) -> None:
        self._validate_quality()

    def q(self, model: str, cohort: str, index: int) -> float:
        """q[model, cohort, t], resolving a ``WinRate`` against its reference at the same
        cohort and time."""
        level = self.models[model].quality_for(cohort).level_at(index)
        return self._resolve(level, cohort, index, seen=(model,))

    def correct(self, seed: int, model: str, cohort: str, index: int) -> bool:
        return draw_correct(seed, model, cohort, index, self.q(model, cohort, index))

    def _resolve(self, level: Level, cohort: str, index: int, seen: tuple[str, ...]) -> float:
        if not isinstance(level, WinRate):
            return float(level)
        if level.reference not in self.models:
            raise ValueError(f"WinRate reference {level.reference!r} is not a scenario model")
        if level.reference in seen:
            raise ValueError(f"WinRate references form a cycle: {(*seen, level.reference)}")
        ref_level = self.models[level.reference].quality_for(cohort).level_at(index)
        q_ref = self._resolve(ref_level, cohort, index, seen=(*seen, level.reference))
        return q_for_w(level.w, q_ref)

    def _validate_quality(self) -> None:
        # q is piecewise constant in the index, changing only at shift points. Checking every
        # model and cohort at index 0 and at every shift point therefore covers all time:
        # an infeasible w anywhere in the scenario fails here, at construction.
        cohorts = {"default"}
        indexes = {0}
        for profile in self.models.values():
            cohorts.update(profile.cohort_quality)
            for spec in (profile.quality, *profile.cohort_quality.values()):
                indexes.update(spec.breakpoints())
        for model in self.models:
            for cohort in cohorts:
                for index in indexes:
                    self.q(model, cohort, index)


# ---------------------------------------------------------------- presets
# Model ids match the provider_model of the mock entries in relay/config/models.py, so Relay's
# in-process mock provider can run any preset. A test keeps the two in sync.
STRONG = "mock-strong-v1"
CHEAP = "mock-cheap-v1"

PRESETS: dict[str, Scenario] = {
    # Instant and flawless: for runs where only the plumbing matters.
    "ideal": Scenario(models={STRONG: ModelProfile(), CHEAP: ModelProfile()}),
    # Plausible numbers for a demo or a smoke test. They are illustrative, not measured from
    # any real provider (PRD §13: label simulator results as such).
    "realistic": Scenario(
        models={
            STRONG: ModelProfile(
                latency=LatencyModel(
                    ttft_median_s=0.6, ttft_p95_s=1.5, tokens_per_s=50, tokens_per_s_jitter=0.2
                ),
                length=OutputLength(median_tokens=60, p95_tokens=200),
                errors=ErrorMix(
                    rate_limited=0.005,
                    overloaded=0.003,
                    server_error=0.002,
                    timeout=0.001,
                    stream_broken=0.002,
                ),
                quality=QualitySpec(base=0.9),
            ),
            CHEAP: ModelProfile(
                latency=LatencyModel(
                    ttft_median_s=0.3, ttft_p95_s=0.8, tokens_per_s=120, tokens_per_s_jitter=0.2
                ),
                length=OutputLength(median_tokens=50, p95_tokens=150),
                errors=ErrorMix(
                    rate_limited=0.005,
                    overloaded=0.003,
                    server_error=0.002,
                    timeout=0.001,
                    stream_broken=0.002,
                ),
                # The cheap model is "not materially worse" 93% of the time.
                quality=QualitySpec(base=WinRate(reference=STRONG, w=0.93)),
            ),
        },
        capacity=Capacity(rpm=600, tpm=400_000),
    ),
}
