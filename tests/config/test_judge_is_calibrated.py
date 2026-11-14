"""A route may only use a judge that passed the M7 gate against human labels.

The result lives in evals/judge/public/calibration.json, which `make reproduce` regenerates.
One judge = one model plus one rubric version: Sonnet on rubric v2 passing says nothing about
Sonnet on v1 or about another model.
"""

import json
from pathlib import Path

from relay.config.loader import load_config

CALIBRATION = (
    Path(__file__).resolve().parents[2] / "evals" / "judge" / "public" / "calibration.json"
)


def test_every_shipped_judge_passed_the_gate_at_its_rubric_version() -> None:
    result = json.loads(CALIBRATION.read_text(encoding="utf-8"))["judges"]
    config = load_config()
    # Any route that declares a judge counts, even before a tenant turns evidence_sampling
    # on: the shipped config is meant to be switched on with one line, so it must be ready.
    shipped = {
        (r.evidence.judge, r.evidence.judge_prompt_version)
        for r in config.routes.values()
        if r.evidence is not None
    }
    for judge, version in shipped:
        entry = result.get(judge)
        assert entry is not None, f"{judge} was never calibrated"
        assert version in entry["rubric"], f"{judge} was calibrated on {entry['rubric']}"
        assert entry["all"]["status"] == "calibrated", f"{judge} did not pass the gate"


def test_the_live_route_ships_the_calibrated_sonnet_judge() -> None:
    live = load_config().routes["live"].evidence
    assert live is not None
    assert (live.judge, live.judge_prompt_version, live.judge_temperature) == (
        "anthropic/sonnet",
        "v2",
        None,
    )
