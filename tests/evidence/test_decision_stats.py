"""The statistics the evidence engine decides with (ADRs 0019, 0020): PPI from counts, the
bootstrap interval, one-sided p-values against a floor, Holm, Benjamini-Hochberg and the
"needs N more" power estimate."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from relay.evidence import stats


def _counts(y: np.ndarray, f: np.ndarray) -> tuple[int, int, int, int]:
    return int(np.sum(y & f)), int(np.sum(y & ~f)), int(np.sum(~y & f)), int(np.sum(~y & ~f))


@settings(max_examples=60, deadline=None)
@given(
    n=st.integers(50, 400),
    big_n=st.integers(2, 4000),
    w=st.floats(0.5, 0.99),
    agree=st.floats(0.5, 1.0),
    seed=st.integers(0, 10_000),
)
def test_counts_form_equals_the_array_form(
    n: int, big_n: int, w: float, agree: float, seed: int
) -> None:
    # The vectorized form is only useful if it is the *same* estimator.
    rng = np.random.default_rng(seed)
    y = rng.random(n) < w
    f = np.where(rng.random(n) < agree, y, ~y)
    fu = rng.random(big_n) < w
    ref = stats.ppi_mean_ci(fu, y, f)
    theta, se, lam = stats.ppi_from_counts(*_counts(y, f), int(fu.sum()), big_n)
    assert float(lam) == pytest.approx(ref.lam, abs=1e-12)
    assert float(se) == pytest.approx(ref.se, abs=1e-12)
    # ppi_mean_ci clips the estimate to [0, 1]; the counts form returns the raw theta.
    assert min(1.0, max(0.0, float(theta))) == pytest.approx(ref.estimate, abs=1e-12)


def test_counts_form_broadcasts_over_arrays() -> None:
    theta, se, lam = stats.ppi_from_counts(
        np.array([80, 85]), np.array([5, 5]), np.array([5, 5]), np.array([10, 5]), 900, 1000
    )
    assert theta.shape == se.shape == lam.shape == (2,)


def test_bootstrap_is_seeded_and_brackets_the_estimate() -> None:
    rng = np.random.default_rng(3)
    y = rng.random(120) < 0.9
    f = np.where(rng.random(120) < 0.9, y, ~y)
    fu = rng.random(2000) < 0.9
    lo1, hi1, reps = stats.ppi_bootstrap_ci(fu, y, f, seed=7, n_boot=500)
    lo2, hi2, _ = stats.ppi_bootstrap_ci(fu, y, f, seed=7, n_boot=500)
    assert (lo1, hi1) == (lo2, hi2)
    est = stats.ppi_mean_ci(fu, y, f).estimate
    assert lo1 <= est <= hi1
    assert len(reps) == 500


def test_bootstrap_width_is_close_to_wald_when_n_is_large() -> None:
    # Both estimate the same sampling distribution; with plenty of data they should agree.
    rng = np.random.default_rng(4)
    y = rng.random(1000) < 0.8
    f = np.where(rng.random(1000) < 0.85, y, ~y)
    fu = rng.random(5000) < 0.8
    wald = stats.ppi_mean_ci(fu, y, f)
    lo, hi, _ = stats.ppi_bootstrap_ci(fu, y, f, n_boot=4000)
    assert hi - lo == pytest.approx(wald.width, rel=0.15)


def test_p_values_against_a_floor() -> None:
    # 1.645 standard errors above the floor: p_safe = 0.05 exactly, p_unsafe = 0.95.
    assert stats.p_safe(0.9 + 1.6448536 * 0.01, 0.01, 0.9) == pytest.approx(0.05, abs=1e-6)
    assert stats.p_unsafe(0.9 + 1.6448536 * 0.01, 0.01, 0.9) == pytest.approx(0.95, abs=1e-6)
    # A zero standard error is decided by the side alone (no division by zero).
    assert stats.p_safe(0.95, 0.0, 0.9) == 0.0
    assert stats.p_unsafe(0.95, 0.0, 0.9) == 1.0


def test_bonferroni_splits_alpha_over_looks() -> None:
    assert stats.bonferroni_alpha(0.05, 4) == pytest.approx(0.0125)
    with pytest.raises(ValueError):
        stats.bonferroni_alpha(0.05, 0)


def test_holm_textbook_example() -> None:
    # Sorted: .005 <= .05/4, .01 <= .05/3, then .03 > .05/2 stops it.
    assert stats.holm([0.01, 0.04, 0.03, 0.005], 0.05) == [True, False, False, True]


def test_holm_is_never_less_powerful_than_bonferroni() -> None:
    rng = np.random.default_rng(5)
    for _ in range(200):
        p = list(rng.random(6) * 0.05)
        bonf = [x <= 0.05 / 6 for x in p]
        assert all(h or not b for h, b in zip(stats.holm(p, 0.05), bonf, strict=True))


def test_benjamini_hochberg_textbook_example() -> None:
    # k*q/m = .0125, .025, .0375, .05: every sorted p is under its line, so all four go.
    assert stats.benjamini_hochberg([0.01, 0.04, 0.03, 0.005], 0.05) == [True] * 4
    # Only the step-up cutoff matters: .03 fails its own line (.025) but a later one passes.
    assert stats.benjamini_hochberg([0.001, 0.03, 0.035, 0.9], 0.05) == [True, True, True, False]


def test_benjamini_hochberg_rejects_at_least_what_holm_rejects() -> None:
    rng = np.random.default_rng(6)
    for _ in range(200):
        p = list(rng.random(8) * 0.1)
        holm, bh = stats.holm(p, 0.05), stats.benjamini_hochberg(p, 0.05)
        assert all(b or not h for h, b in zip(holm, bh, strict=True))


def test_empty_families() -> None:
    assert stats.holm([], 0.05) == []
    assert stats.benjamini_hochberg([], 0.05) == []


def test_samples_needed() -> None:
    # Already decisive: nothing more needed.
    assert stats.samples_needed(0.95, 0.01, 500, 0.90, 0.05) == 0
    # Halving the gap needs about four times the data.
    wide = stats.samples_needed(0.92, 0.02, 500, 0.90, 0.05)
    narrow = stats.samples_needed(0.91, 0.02, 500, 0.90, 0.05)
    assert wide is not None and narrow is not None
    assert narrow > 3 * wide
    assert stats.samples_needed(0.90, 0.02, 500, 0.90, 0.05) is None


@settings(max_examples=60, deadline=None)
@given(
    est=st.floats(0.51, 0.99),
    se=st.floats(0.001, 0.1),
    n=st.integers(50, 5000),
)
def test_samples_needed_is_monotone_in_the_gap(est: float, se: float, n: int) -> None:
    floor = 0.5
    near = stats.samples_needed(est, se, n, floor, 0.05)
    far = stats.samples_needed(min(0.999, est + 0.01), se, n, floor, 0.05)
    assert near is not None and far is not None
    assert far <= near


def test_a_fixed_lambda_of_one_is_classic_ppi() -> None:
    # theta = mean(judge on unlabeled) + mean(human - judge on labeled): it follows the judge
    # fully, which the M5 monitor needs when the population moves away from the labels.
    theta, _, lam = stats.ppi_from_counts(80, 5, 10, 5, 850, 1000, lam=1.0)
    assert float(lam) == 1.0
    assert float(theta) == pytest.approx(0.85 + (85 / 100 - 90 / 100))
