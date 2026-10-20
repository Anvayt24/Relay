import pytest

from relay.simulator.judge import JudgeParams, MockJudge

N = 20_000


def test_a_perfect_judge_reports_the_truth() -> None:
    judge = MockJudge(seed=0)
    for cand in (True, False):
        for ref in (True, False):
            expected = "loss" if (ref and not cand) else "win_or_tie"
            assert judge.verdict(cand_correct=cand, ref_correct=ref, index=0) == expected


def test_sensitivity_and_false_loss_rate_are_respected() -> None:
    judge = MockJudge(seed=1, params=JudgeParams(sensitivity=0.7, false_loss_rate=0.05))
    on_losses = [judge.verdict(cand_correct=False, ref_correct=True, index=i) for i in range(N)]
    on_ties = [judge.verdict(cand_correct=True, ref_correct=True, index=i) for i in range(N)]
    assert on_losses.count("loss") / N == pytest.approx(0.7, abs=0.015)
    assert on_ties.count("loss") / N == pytest.approx(0.05, abs=0.006)


def test_drift_changes_the_parameters_from_its_index() -> None:
    blind = JudgeParams(sensitivity=0.0)
    judge = MockJudge(seed=2, drift=[(100, blind)])
    assert judge.params_at(99) == JudgeParams()
    assert judge.params_at(100) == blind
    assert judge.verdict(cand_correct=False, ref_correct=True, index=99) == "loss"
    assert judge.verdict(cand_correct=False, ref_correct=True, index=100) == "win_or_tie"


def test_params_validate() -> None:
    with pytest.raises(ValueError):
        JudgeParams(sensitivity=1.1)


def test_judging_errors_are_independent_per_candidate() -> None:
    judge = MockJudge(seed=3, params=JudgeParams(sensitivity=0.5))
    a = [
        judge.verdict(cand_correct=False, ref_correct=True, index=i, candidate="a")
        for i in range(200)
    ]
    b = [
        judge.verdict(cand_correct=False, ref_correct=True, index=i, candidate="b")
        for i in range(200)
    ]
    assert a != b
