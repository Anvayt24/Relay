import pytest

from relay.simulator.quality import QualitySpec, WinRate, draw_correct, outcome, q_for_w
from relay.simulator.scenario import ModelProfile, Scenario

N = 20_000


@pytest.mark.parametrize(("w", "q_ref"), [(0.93, 0.9), (0.85, 0.9), (0.99, 0.5), (1.0, 0.7)])
def test_q_for_w_round_trips(w: float, q_ref: float) -> None:
    q_cand = q_for_w(w, q_ref)
    assert 0.0 <= q_cand <= 1.0
    # w = 1 - P(loss) = 1 - q_ref * (1 - q_cand)
    assert 1 - q_ref * (1 - q_cand) == pytest.approx(w)


def test_q_for_w_rejects_infeasible_targets() -> None:
    with pytest.raises(ValueError, match="infeasible"):
        q_for_w(0.80, 0.1)


def test_shifts_apply_from_their_index_on() -> None:
    spec = QualitySpec(base=0.9, shifts=[(200, 0.5), (100, 0.7)])  # unsorted on purpose
    assert spec.level_at(0) == 0.9
    assert spec.level_at(99) == 0.9
    assert spec.level_at(100) == 0.7
    assert spec.level_at(199) == 0.7
    assert spec.level_at(10_000) == 0.5


def test_quality_spec_validates_q() -> None:
    with pytest.raises(ValueError):
        QualitySpec(base=1.5)
    with pytest.raises(ValueError):
        WinRate("x", 1.2)


def test_outcome_table() -> None:
    assert outcome(cand_correct=True, ref_correct=False) == "win"
    assert outcome(cand_correct=False, ref_correct=True) == "loss"
    assert outcome(cand_correct=True, ref_correct=True) == "tie"
    assert outcome(cand_correct=False, ref_correct=False) == "tie"


def test_empirical_win_rate_matches_the_target() -> None:
    scenario = Scenario(
        models={
            "strong": ModelProfile(quality=QualitySpec(base=0.9)),
            "cheap": ModelProfile(quality=QualitySpec(base=WinRate("strong", 0.93))),
        }
    )
    losses = sum(
        outcome(
            scenario.correct(1, "cheap", "default", i), scenario.correct(1, "strong", "default", i)
        )
        == "loss"
        for i in range(N)
    )
    # Fixed seed: deterministic, never flaky. The standard error is ~0.0018, so 0.01 is ~5 SE.
    assert 1 - losses / N == pytest.approx(0.93, abs=0.01)


def test_degradation_flips_only_requests_in_the_gap() -> None:
    # Common random numbers: with the same uniform per request, lowering q from 0.9 to 0.7 can
    # only turn right answers wrong, never the other way round.
    for i in range(2_000):
        before = draw_correct(3, "m", "default", i, 0.9)
        after = draw_correct(3, "m", "default", i, 0.7)
        assert not (after and not before)


def test_correctness_is_a_pure_function_of_its_keys() -> None:
    assert [draw_correct(5, "m", "c", i, 0.5) for i in range(50)] == [
        draw_correct(5, "m", "c", i, 0.5) for i in range(50)
    ]
