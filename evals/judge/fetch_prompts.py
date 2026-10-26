"""Fetch a small, stratified sample of public prompts for judge calibration.

    python -m evals.judge.fetch_prompts --n 30 --out evals/judge/prompts.jsonl

Source: **databricks-dolly-15k**, license **CC BY-SA 3.0** (verified on its dataset card,
2026-10-23): free to use, including commercially, with attribution and share-alike. Only the
`instruction` field of records with no extra `context` is used, so each prompt stands alone.
Credit: Databricks, "databricks-dolly-15k".

Why a public set: calibration prompts go to third-party model providers (the candidate,
reference and judge). Real tenant prompts must never be used for that without opt-in (PRD
§10.2), and Gemini is never used at all (PRD §13).

Stratified by the dataset's `category` so the sample isn't all trivia, and seeded so the same
command gives the same prompts. The output includes the source file's sha256.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import httpx

URL = (
    "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/"
    "databricks-dolly-15k.jsonl"
)
CATEGORIES = ("open_qa", "general_qa", "brainstorming", "classification", "creative_writing")
MAX_CHARS = 280  # short prompts: cheap to answer and to judge


def sample(records: list[dict[str, str]], n: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    usable = [
        r
        for r in records
        if r["category"] in CATEGORIES
        and not r["context"].strip()
        and len(r["instruction"]) <= MAX_CHARS
    ]
    by_cat: dict[str, list[dict[str, str]]] = {c: [] for c in CATEGORIES}
    for r in usable:
        by_cat[r["category"]].append(r)
    for group in by_cat.values():
        rng.shuffle(group)
    picked: list[dict[str, str]] = []
    while len(picked) < n and any(by_cat.values()):  # round-robin across categories
        for cat in CATEGORIES:
            if by_cat[cat] and len(picked) < n:
                picked.append(by_cat[cat].pop())
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m evals.judge.fetch_prompts", description=__doc__
    )
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="evals/judge/prompts.jsonl")
    args = parser.parse_args()

    raw = httpx.get(URL, timeout=120, follow_redirects=True)
    raw.raise_for_status()
    records = [json.loads(line) for line in raw.text.splitlines() if line.strip()]
    chosen = sample(records, args.n, args.seed)
    out = Path(args.out)
    out.write_text(
        "".join(
            json.dumps({"prompt": r["instruction"], "category": r["category"]}) + "\n"
            for r in chosen
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(chosen)} prompts to {out}")
    digest = hashlib.sha256(raw.content).hexdigest()
    print(f"source sha256 {digest} (databricks-dolly-15k, CC BY-SA 3.0)")


if __name__ == "__main__":
    main()
