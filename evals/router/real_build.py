"""Build the *real* router dataset from judged pairs (the counted real run; ADR 0018 §7).

    python -m evals.router.real_build --pairs evals/router/real/pairs.jsonl \\
        [--labels my_labels.jsonl]

Input is the JSONL written by ``python -m relay.evidence.pairs build`` (candidate = the cheap
model, reference = the strong one, a judge's verdict per prompt). Output is the same dataset
layout the simulated world uses (``data.py``), so ``train.py`` and ``evaluate.py`` work on it
unchanged with ``--world real``.

What differs from the simulation, and is stated everywhere the numbers appear:

- **There is no ground truth.** ``y_true`` is null, except where a **human** labeled the pair
  (``--labels``, made with ``python -m relay.evidence.label``): those become the gold subset PPI
  needs, but only when there are at least 50 of them in a held-out split.
- **The judge has not passed the M7 gate** (no human labels at calibration scale yet), so every
  result is "uncalibrated" and may not drive a decision.
- The set is small (about a hundred prompts), so intervals are wide.
- Judge abstentions are dropped (never guessed).

Clusters (cosine > 0.95) and the 60/20/20 split by whole cluster are built exactly as for the
simulation (PRD FR-7: remove near-duplicates, split by prompt).
"""

import argparse
import asyncio
import importlib.metadata
from pathlib import Path
from typing import Any

import numpy as np

from evals.router.build import GOLD_PER_SPLIT, NEAR_DUPLICATE_COSINE, assign_splits, cluster
from evals.router.data import file_sha256, write_dataset
from relay.cache.embed import MODEL_NAME, FastEmbedder
from relay.evidence.calibrate import read_labels
from relay.evidence.pairs import EvalPair, read_jsonl

REAL = "real"


def rows_from_pairs(
    pairs: list[EvalPair], human: dict[str, str]
) -> tuple[list[dict[str, Any]], int]:
    """Dataset rows (without cluster/split yet) and how many pairs were dropped (abstentions)."""
    rows, dropped = [], 0
    for p in pairs:
        if p.judge_label is None:
            dropped += 1  # an abstention is not a label
            continue
        y_true = None
        if p.id in human:
            y_true = int(human[p.id] != "loss")
        rows.append(
            {
                "id": p.prompt_hash,
                "text": p.prompt,
                "family": p.cohort,  # the cohort key stands in for the simulation's family
                "instance": p.prompt_hash,
                "y_true": y_true,
                "y_judge": int(p.judge_label != "loss"),
                "w": None,
            }
        )
    return rows, dropped


def gold_by_split(rows: list[dict[str, Any]], splits: list[str]) -> list[bool]:
    """Human-labeled pairs in a held-out split are the gold set (all of them: there are few)."""
    return [
        r["y_true"] is not None and s in ("val", "test") for r, s in zip(rows, splits, strict=True)
    ]


async def build(pairs_path: Path, labels_path: Path | None) -> None:
    pairs = read_jsonl(pairs_path)
    human: dict[str, str] = {}
    if labels_path is not None:
        for row in read_labels(labels_path):
            if not row.get("is_relabel"):
                human.setdefault(row["pair_id"], row["label"])
    rows, dropped = rows_from_pairs(pairs, human)
    embedder = FastEmbedder()
    await embedder.start()
    vectors = np.asarray(await embedder.embed_many([r["text"] for r in rows]), dtype=np.float32)
    await embedder.close()

    clusters = cluster([r["instance"] for r in rows], vectors.astype(np.float64))
    splits = assign_splits(clusters, [r["id"] for r in rows])
    gold = gold_by_split(rows, splits)
    for r, c, s, g in zip(rows, clusters, splits, gold, strict=True):
        r.update(cluster=c, split=s, is_gold=g)

    first = pairs[0] if pairs else None
    write_dataset(
        REAL,
        rows,
        vectors,
        {
            "world": "REAL run; judge UNCALIBRATED (no human labels at M7 scale); not reproducible",
            "embedding_model": MODEL_NAME,
            "fastembed_version": importlib.metadata.version("fastembed"),
            "near_duplicate_cosine": NEAR_DUPLICATE_COSINE,
            "gold_per_split_cap": GOLD_PER_SPLIT,
            "source_pairs_sha256": file_sha256(pairs_path),
            "pairs_total": len(pairs),
            "abstentions_dropped": dropped,
            "human_labels": len(human),
            "candidate_model": first.candidate_model if first else None,
            "reference_model": first.reference_model if first else None,
            "judge_model": first.judge_model if first else None,
            "judge_prompt_version": first.judge_prompt_version if first else None,
            "label": "y = 1 if the judge says the cheap answer is win-or-tie vs the strong "
            "answer. y_true is null unless a human labeled the pair (then it is the gold "
            "subset). The judge is uncalibrated: nothing here may drive a decision.",
        },
    )
    counts = {s: splits.count(s) for s in ("train", "val", "test")}
    print(
        f"{len(rows)} rows ({dropped} abstentions dropped), splits {counts}, "
        f"{sum(gold)} human-labeled gold"
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m evals.router.real_build", description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--labels", default=None, help="human labels JSONL (optional)")
    args = parser.parse_args()
    asyncio.run(build(Path(args.pairs), Path(args.labels) if args.labels else None))


if __name__ == "__main__":
    main()
