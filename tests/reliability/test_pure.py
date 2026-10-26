"""The pure building blocks of the retry owner: policy table, backoff, retry budget, deadline."""

import random

from hypothesis import given, settings
from hypothesis import strategies as st

from relay.reliability.backoff import full_jitter
from relay.reliability.deadline import Deadline
from relay.reliability.policy import (
    BREAKER_FAILURES,
    BREAKER_IGNORES,
    CONTEXT_FALLBACK,
    FALLBACK_ON,
    RETRY_ONCE,
    can_fall_back,
)
from relay.reliability.retry_budget import RetryBudget
from relay.schemas.errors import ErrorClass

from .conftest import FakeClock  # type: ignore[import-not-found]


# ---------------------------------------------------------------- policy (Appendix B)
def test_terminal_classes_never_retry_or_fall_back() -> None:
    for terminal in (ErrorClass.BAD_REQUEST, ErrorClass.CONTENT_FILTERED):
        assert terminal not in RETRY_ONCE
        assert not can_fall_back(terminal)


def test_rate_limits_are_left_to_the_pause_state() -> None:
    # Never retried on the same limit, never counted by the breaker (FR-5), but they do
    # fall back to another provider.
    for limit in (ErrorClass.RATE_LIMITED, ErrorClass.QUOTA_EXHAUSTED):
        assert limit not in RETRY_ONCE
        assert limit not in BREAKER_FAILURES
        assert limit in BREAKER_IGNORES
        assert can_fall_back(limit)


def test_policy_sets_are_consistent() -> None:
    assert RETRY_ONCE <= FALLBACK_ON  # a retry that fails again falls back
    assert not BREAKER_FAILURES & BREAKER_IGNORES
    assert not FALLBACK_ON & CONTEXT_FALLBACK
    # A broken stream counts toward the breaker, but it can never be a fallback trigger:
    # the client already has part of an answer (FR-5).
    assert ErrorClass.STREAM_BROKEN in BREAKER_FAILURES
    assert not can_fall_back(ErrorClass.STREAM_BROKEN)


# ---------------------------------------------------------------- backoff
@given(attempt=st.integers(0, 20), seed=st.integers(0, 2**32))
def test_full_jitter_stays_within_the_exponential_cap(attempt: int, seed: int) -> None:
    delay = full_jitter(attempt, base_s=0.2, cap_s=2.0, rng=random.Random(seed))
    assert 0.0 <= delay <= min(2.0, 0.2 * 2**attempt)


def test_full_jitter_spreads_retries_over_the_whole_window() -> None:
    # The point of *full* jitter: retries that failed together don't come back together.
    rng = random.Random(1)
    delays = [full_jitter(0, base_s=1.0, cap_s=10.0, rng=rng) for _ in range(2000)]
    assert min(delays) < 0.05
    assert max(delays) > 0.95
    assert 0.45 < sum(delays) / len(delays) < 0.55


# ---------------------------------------------------------------- retry budget
def test_retry_budget_allows_the_minimum_even_with_no_traffic() -> None:
    clock = FakeClock()
    budget = RetryBudget(ratio=0.1, minimum=3, window_s=10, clock=clock)
    assert [budget.try_spend("p") for _ in range(4)] == [True, True, True, False]
    clock.advance(10.1)  # the window slid past them
    assert budget.try_spend("p")


def test_retry_budget_is_per_provider() -> None:
    budget = RetryBudget(ratio=0.0, minimum=1, window_s=10, clock=FakeClock())
    assert budget.try_spend("a")
    assert not budget.try_spend("a")
    assert budget.try_spend("b")


@settings(max_examples=200, deadline=None)
@given(
    events=st.lists(st.tuples(st.sampled_from(["attempt", "retry"]), st.floats(0, 3)), max_size=200)
)
def test_retries_never_exceed_ratio_plus_minimum_in_any_window(
    events: list[tuple[str, float]],
) -> None:
    """FR-4: retries <= 10% of requests (+ a small floor) per provider per window, whatever
    the arrival pattern."""
    clock = FakeClock()
    budget = RetryBudget(ratio=0.1, minimum=3, window_s=10, clock=clock)
    attempts: list[float] = []
    granted: list[float] = []
    for kind, gap in events:
        clock.advance(gap)
        if kind == "attempt":
            budget.record_attempt("p")
            attempts.append(clock.now)
        elif budget.try_spend("p"):
            granted.append(clock.now)
            window_start = clock.now - 10
            in_window = [t for t in granted if t > window_start]
            attempts_in_window = [t for t in attempts if t > window_start]
            assert len(in_window) <= 3 + 0.1 * len(attempts_in_window)


# ---------------------------------------------------------------- deadline
def test_deadline_clamps_and_refuses_attempts_that_cannot_finish() -> None:
    d = Deadline(at=100.0)
    assert d.remaining(99.0) == 1.0
    assert d.remaining(101.0) == 0.0
    assert d.clamp(99.0, 15.0) == 1.0
    assert d.clamp(80.0, 15.0) == 15.0
    assert d.allows(99.0, 0.5)
    assert not d.allows(99.8, 0.5)
    assert not d.allows(100.0, 0.0)  # "no time needed" still needs *some* time left
