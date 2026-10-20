from typing import Any

import pytest

from relay.simulator.capacity import CapacityTracker
from relay.simulator.scenario import Capacity


def test_rpm_exhaustion_gives_an_exact_retry_after(clock: Any) -> None:
    tracker = CapacityTracker(Capacity(rpm=60, tpm=1_000_000), clock)  # refills 1 request/s
    for _ in range(60):
        assert tracker.admit("k", "m", 1, 1).admitted
    refused = tracker.admit("k", "m", 1, 1)
    assert not refused.admitted
    assert refused.retry_after_s == pytest.approx(1.0)
    assert refused.remaining_requests == 0
    assert refused.scope == "account"

    clock.advance(0.5)
    assert tracker.admit("k", "m", 1, 1).retry_after_s == pytest.approx(0.5)
    clock.advance(0.5)
    assert tracker.admit("k", "m", 1, 1).admitted


def test_tpm_charges_prompt_plus_max_tokens(clock: Any) -> None:
    tracker = CapacityTracker(Capacity(rpm=1000, tpm=600), clock)  # refills 10 tokens/s
    state = tracker.admit("k", "m", prompt_tokens=100, max_tokens=400)
    assert state.admitted
    assert state.remaining_tokens == 100
    # A tiny question with a big output cap is refused: the cap is reserved up front.
    refused = tracker.admit("k", "m", prompt_tokens=1, max_tokens=499)
    assert not refused.admitted
    assert refused.retry_after_s == pytest.approx(40.0)  # 400 missing / 10 per s


def test_default_output_tokens_are_charged_without_max_tokens(clock: Any) -> None:
    tracker = CapacityTracker(Capacity(rpm=1000, tpm=2000, default_output_tokens=1500), clock)
    assert tracker.admit("k", "m", prompt_tokens=10, max_tokens=None).remaining_tokens == 490


def test_a_refused_request_charges_nothing(clock: Any) -> None:
    tracker = CapacityTracker(Capacity(rpm=10, tpm=100), clock)
    assert not tracker.admit("k", "m", prompt_tokens=200, max_tokens=0).admitted  # TPM refuses
    assert tracker.admit("k", "m", 1, 1).remaining_requests == 9  # RPM was not charged


def test_keys_are_isolated(clock: Any) -> None:
    tracker = CapacityTracker(Capacity(rpm=1, tpm=1000), clock)
    assert tracker.admit("alice", "m", 1, 1).admitted
    assert not tracker.admit("alice", "m", 1, 1).admitted
    assert tracker.admit("bob", "m", 1, 1).admitted


def test_account_scope_is_shared_across_models(clock: Any) -> None:
    tracker = CapacityTracker(Capacity(rpm=1, tpm=1000, scope="account"), clock)
    assert tracker.admit("k", "m1", 1, 1).admitted
    assert not tracker.admit("k", "m2", 1, 1).admitted


def test_model_scope_gives_each_model_its_own_budget(clock: Any) -> None:
    tracker = CapacityTracker(Capacity(rpm=1, tpm=1000, scope="model"), clock)
    assert tracker.admit("k", "m1", 1, 1).admitted
    assert tracker.admit("k", "m2", 1, 1).admitted
    refused = tracker.admit("k", "m1", 1, 1)
    assert not refused.admitted
    assert refused.scope == "model"


def test_reset_reports_time_until_full(clock: Any) -> None:
    tracker = CapacityTracker(Capacity(rpm=60, tpm=6000), clock)
    state = tracker.admit("k", "m", prompt_tokens=50, max_tokens=50)
    assert state.reset_requests_s == pytest.approx(1.0)  # 1 request at 1/s
    assert state.reset_tokens_s == pytest.approx(1.0)  # 100 tokens at 100/s
