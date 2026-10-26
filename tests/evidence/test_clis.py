"""The command-line entry points, run end to end with fake adapters (no real model calls)."""

import json
import sys
from pathlib import Path

import pytest

from relay.evidence import calibrate, pairs, validate
from relay.evidence.pairs import EvalPair, read_jsonl, write_jsonl
from relay.providers.registry import ProviderKeys

from ..conftest import TEST_CONFIG_PATH  # type: ignore[import-not-found]
from .test_rubric_and_judge import FakeAdapter


def test_build_writes_judged_pairs_and_counts_its_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "\n".join(json.dumps({"prompt": p}) for p in ("what is 2+2", "name a color")),
        encoding="utf-8",
    )
    fake = FakeAdapter(
        "four", "4", '{"verdict": "neither"}',  # pair 0: candidate, reference, judge
        "red", "blue", '{"verdict": "neither"}',  # pair 1
    )  # fmt: skip
    monkeypatch.setattr(pairs, "build_adapters", lambda *a, **k: {"mock": fake})
    out = tmp_path / "pairs.jsonl"
    argv = [
        "build", "--prompts", str(prompts), "--candidate", "mock/cheap",
        "--reference", "mock/strong", "--judge", "mock/strong", "--limit", "2",
        "--out", str(out), "--config", TEST_CONFIG_PATH, "--max-calls", "6",
    ]  # fmt: skip
    pairs.main(argv)

    built = read_jsonl(out)
    assert [p.candidate_response for p in built] == ["four", "red"]
    assert [p.judge_label for p in built] == ["tie", "tie"]
    assert len(fake.requests) == 6  # exactly the 3 calls per prompt that were planned
    text = capsys.readouterr().out
    assert "6 model calls planned" in text and "sha256" in text


def test_build_refuses_a_model_whose_provider_has_no_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts = tmp_path / "p.jsonl"
    prompts.write_text(json.dumps({"prompt": "hi"}), encoding="utf-8")
    monkeypatch.setattr(pairs, "build_adapters", lambda *a, **k: {})
    argv = [
        "build", "--prompts", str(prompts), "--candidate", "mock/cheap",
        "--reference", "mock/strong", "--judge", "mock/strong",
        "--out", str(tmp_path / "o.jsonl"), "--config", TEST_CONFIG_PATH,
    ]  # fmt: skip
    with pytest.raises(SystemExit, match="no API key"):
        pairs.main(argv)


def test_calibrate_cli_writes_a_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ps = [
        EvalPair(
            id=f"p{i}", prompt_hash="h", prompt="q", cohort="c", candidate_model="a",
            reference_model="b", candidate_response="x", reference_response="y",
            judge_label="loss" if i % 3 == 0 else "tie", judge_model="j",
            judge_prompt_version="v1", position_swapped=False, text_retained=True,
            ts="2026-01-01T00:00:00+00:00", source="pilot",
        )
        for i in range(60)
    ]  # fmt: skip
    pairs_path, labels_path, out = tmp_path / "p.jsonl", tmp_path / "l.jsonl", tmp_path / "c.json"
    write_jsonl(ps, pairs_path)
    labels_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "pair_id": p.id,
                    "label": p.judge_label,  # a perfect agreement with the judge
                    "annotator": "me",
                    "is_relabel": False,
                }
            )
            for p in ps
        ),
        encoding="utf-8",
    )
    argv = [
        "calibrate",
        "--pairs",
        str(pairs_path),
        "--labels",
        str(labels_path),
        "--out",
        str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    calibrate.main()
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["status"] == "calibrated" and report["n"] == 60
    assert len(report["pairs_sha256"]) == 64

    monkeypatch.setattr(sys, "argv", ["calibrate", "--pairs", str(pairs_path), "--self-pairs"])
    calibrate.main()
    assert json.loads(capsys.readouterr().out.split("}\n", 1)[1])["n"] == 60


def test_validate_cli_prints_and_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "sim.json"
    monkeypatch.setattr(sys, "argv", ["validate", "--n", "300", "--out", str(out)])
    validate.main()
    results = json.loads(out.read_text(encoding="utf-8"))["results"]
    assert [r["judge"] for r in results] == ["good", "mediocre", "loss_blind"]
    assert "loss_blind" in capsys.readouterr().out


def test_provider_keys_object_is_unused_here() -> None:
    # The CLI builds real adapters from ProviderKeys(); tests replace build_adapters, so no
    # key is ever read. Guard against a test accidentally constructing one with keys.
    assert ProviderKeys(_env_file=None).get("NOT_SET_ANYWHERE") is None
