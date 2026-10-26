import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from relay.evidence import stats
from relay.evidence.stats import Confusion


def test_confusion_counts_and_derived_rates() -> None:
    human = [True, True, True, False, False, False, False, False]
    judge = [True, True, False, True, False, False, False, False]
    c = stats.confusion(human, judge)
    assert (c.tp, c.fn, c.fp, c.tn) == (2, 1, 1, 4)
    assert stats.accuracy(c) == 6 / 8
    assert stats.loss_recall(c) == 2 / 3  # the judge caught 2 of 3 real losses
    assert stats.win_recall(c) == 4 / 5


def test_kappa_known_values() -> None:
    assert stats.cohen_kappa(Confusion(tp=5, fn=0, fp=0, tn=5)) == 1.0  # perfect
    # Hand-worked: observed 0.7; both say loss half the time -> expected 0.5 -> kappa 0.4.
    assert stats.cohen_kappa(Confusion(tp=4, fn=1, fp=2, tn=3)) == pytest.approx(0.4)
    assert stats.cohen_kappa(Confusion(tp=0, fn=5, fp=5, tn=0)) == -1.0  # always opposite


def test_kappa_paradox_high_accuracy_no_skill() -> None:
    # 2% losses; the judge misses both and false-alarms once: 97% accurate, but no skill.
    c = Confusion(tp=0, fn=2, fp=1, tn=97)
    assert stats.accuracy(c) == 0.97
    assert stats.cohen_kappa(c) < 0.01


def test_degenerate_inputs_give_nan_not_a_made_up_number() -> None:
    assert math.isnan(stats.cohen_kappa(Confusion(0, 0, 0, 0)))
    assert math.isnan(stats.loss_recall(Confusion(0, 0, 3, 7)))  # no real losses to catch
    with pytest.raises(ValueError, match="paired"):
        stats.confusion([True], [True, False])


@given(st.lists(st.tuples(st.booleans(), st.booleans()), min_size=1, max_size=200))
def test_kappa_is_bounded(pairs: list[tuple[bool, bool]]) -> None:
    c = stats.confusion([h for h, _ in pairs], [j for _, j in pairs])
    k = stats.cohen_kappa(c)
    assert math.isnan(k) or -1.0 - 1e-9 <= k <= 1.0 + 1e-9


def test_bootstrap_is_seeded_and_brackets_the_estimate() -> None:
    human = [True] * 30 + [False] * 70
    judge = [True] * 24 + [False] * 6 + [False] * 66 + [True] * 4
    a = stats.bootstrap_ci(human, judge, stats.loss_recall, seed=3)
    b = stats.bootstrap_ci(human, judge, stats.loss_recall, seed=3)
    assert a == b  # same data, same interval
    point = stats.loss_recall(stats.confusion(human, judge))
    assert a[0] < point < a[1]
    assert point == pytest.approx(0.8)


def test_clopper_pearson_rule_of_three() -> None:
    # Zero failures in n: the bound is 0.05 ** (1/n). About 300 clean hits to show >= 99%.
    assert stats.clopper_pearson_lower(300, 300) == pytest.approx(0.05 ** (1 / 300), abs=1e-9)
    assert stats.clopper_pearson_lower(300, 300) > 0.99
    assert stats.clopper_pearson_lower(299, 300) < stats.clopper_pearson_lower(300, 300)
    assert stats.clopper_pearson_lower(0, 10) == 0.0
    assert math.isnan(stats.clopper_pearson_lower(0, 0))
