"""The adversarial calibration cases (evals/judge/adversarial.jsonl), checked offline.

Whether a real judge *resists* them is measured in the live pilot. What can be proven without
a model is the structural defense: every attacker-controlled string lands inside its own
delimited block, and no attacker string can contain the delimiter's secret nonce.
"""

import json
from pathlib import Path

import pytest

from relay.evidence.rubric import build_messages

CASES = [
    json.loads(line)
    for line in (Path(__file__).parents[2] / "evals" / "judge" / "adversarial.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
    if line.strip()
]


def test_the_case_file_is_well_formed() -> None:
    assert len(CASES) >= 8
    assert {c["truth"] for c in CASES} == {"loss", "tie"}  # attacks on both outcomes
    assert len({c["id"] for c in CASES}) == len(CASES)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_attacker_text_stays_inside_its_delimited_block(case: dict[str, str]) -> None:
    user = build_messages(case["request"], case["candidate"], case["reference"])[1]["content"]
    nonce = user.split("<<<REQUEST-", 1)[1].split(">>>", 1)[0]
    for text in (case["request"], case["candidate"], case["reference"]):
        assert nonce not in text  # the attacker cannot know it, so cannot close the block
    # The three blocks appear exactly once each, in order, with the texts verbatim inside.
    positions = [
        user.index(f"<<<{label}-{nonce}>>>") for label in ("REQUEST", "RESPONSE-A", "RESPONSE-B")
    ]
    assert positions == sorted(positions)
    for label, text in (
        ("REQUEST", case["request"]),
        ("RESPONSE-A", case["candidate"]),
        ("RESPONSE-B", case["reference"]),
    ):
        block = user.split(f"<<<{label}-{nonce}>>>\n", 1)[1].split(
            f"\n<<<END-{label}-{nonce}>>>", 1
        )[0]
        assert block == text
