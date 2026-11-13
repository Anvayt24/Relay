"""The README's quickstart is the path a new user follows, so CI runs it as written
(scripts/readme_quickstart.py). These tests keep that script honest about what it extracts."""

from pathlib import Path

import pytest

from scripts.readme_quickstart import extract_quickstart

ROOT = Path(__file__).resolve().parents[1]


def test_the_real_readme_has_one_quickstart_block_with_the_whole_path() -> None:
    block = extract_quickstart((ROOT / "README.md").read_text(encoding="utf-8"))
    assert "docker compose up" in block  # it starts the stack ...
    assert "rk_" in block  # ... reads the first key from the logs ...
    assert "/v1/chat/completions" in block  # ... and sends a request with it


def test_only_the_marked_block_is_taken() -> None:
    text = "```sh\nrm -rf /\n```\n\n```sh quickstart\necho hi\n```\n\n```sh\necho no\n```\n"
    assert extract_quickstart(text) == "echo hi\n"


def test_no_marked_block_is_an_error_not_a_silent_pass() -> None:
    with pytest.raises(ValueError, match="quickstart"):
        extract_quickstart("```sh\necho hi\n```\n")


def test_two_marked_blocks_are_ambiguous() -> None:
    block = "```sh quickstart\necho a\n```\n"
    with pytest.raises(ValueError, match="more than one"):
        extract_quickstart(block + block)
