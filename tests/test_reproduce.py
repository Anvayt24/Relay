"""scripts/reproduce.py decides whether regenerated numbers "changed". Its comparison must
ignore noise (the last digit of a float, line endings) and catch anything real."""

from pathlib import Path

from scripts.reproduce import ROOT, STEPS, close, same


def test_the_last_digit_of_a_float_is_noise_but_a_real_change_is_not() -> None:
    assert close({"a": 0.9545235625892844}, {"a": 0.9545235625892845})
    assert close([1.0, 2.0], [1.0, 2.0 + 1e-12])
    assert not close({"a": 0.95}, {"a": 0.951})
    assert not close({"a": 1}, {"a": 2})
    assert not close({"a": 1}, {"b": 1})  # a renamed or missing key is a change
    assert not close([1.0], [1.0, 2.0])


def test_json_is_compared_by_value_and_markdown_as_text() -> None:
    assert same("x.json", '{"a": 0.1000000000001}', '{"a": 0.1}')
    assert not same("x.json", '{"a": 0.1}', '{"a": 0.2}')
    assert same("x.md", "one\ntwo\n", "one\ntwo\n")
    assert not same("x.md", "33.5% saved", "33.6% saved")
    assert not same("x.md", None, "now it exists")


def test_every_file_a_step_writes_is_one_the_repository_keeps() -> None:
    for what, _, outputs in STEPS:
        for rel in outputs:
            assert (Path(ROOT) / rel).exists(), f"{what} writes {rel}, which is missing"
