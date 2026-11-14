"""The frozen router datasets and how they are read.

A dataset is a folder (``sim/`` or ``real/``) holding:

- ``prompts.jsonl``     one row per prompt: id, text, family, instance, cluster, split, is_gold,
                        y_true (the truth, or null when unknown), y_judge, w (hidden, sim only)
- ``embeddings.npy``    float32, one L2-normalized bge-small vector per row, in the same order
- ``manifest.json``     sha256 of each file, seed, embedding model, what the labels mean

Everything the evaluation and the CI gate need is frozen in here, so they need no embedding
model and no network, and are deterministic.

``y_judge`` and ``y_true`` are **1 = win-or-tie, 0 = loss** (the router predicts the
probability of 1). ``is_gold`` marks the small subset of each held-out split whose truth is
known (in the simulation, the truth plays the human; in real data, human labels): the
labeled set PPI needs, kept disjoint from the judge-only prompts.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from relay.routing.features import scalar_features

HERE = Path(__file__).parent
SPLITS = ("train", "val", "test")


def write_text(path: Path, text: str) -> None:
    # newline="\n": Path.write_text on Windows emits CRLF, git stores LF, and a sha256 taken on
    # Windows would then differ on Linux CI (the cache eval hit exactly this).
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Dataset:
    name: str
    rows: list[dict[str, Any]]
    embeddings: np.ndarray  # (n, dim)
    scalars: np.ndarray  # (n, N_SCALARS), raw
    manifest: dict[str, Any]

    def mask(self, split: str) -> np.ndarray:
        return np.array([r["split"] == split for r in self.rows])

    def col(self, key: str, *, default: float = float("nan")) -> np.ndarray:
        return np.array(
            [default if r[key] is None else float(r[key]) for r in self.rows], dtype=float
        )

    @property
    def y_judge(self) -> np.ndarray:
        return self.col("y_judge")

    @property
    def y_true(self) -> np.ndarray:
        return self.col("y_true")  # nan where unknown

    @property
    def gold(self) -> np.ndarray:
        return np.array([bool(r["is_gold"]) for r in self.rows])

    @property
    def families(self) -> np.ndarray:
        return np.array([r["family"] for r in self.rows])

    @property
    def has_truth(self) -> bool:
        return bool(np.isfinite(self.y_true).any())


def load(name: str) -> Dataset:
    folder = HERE / name
    rows = [
        json.loads(line)
        for line in (folder / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    embeddings = np.load(folder / "embeddings.npy").astype(np.float64)
    scalars = np.array([scalar_features(r["text"]) for r in rows], dtype=float)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    return Dataset(name, rows, embeddings, scalars, manifest)


def write_dataset(
    name: str,
    rows: list[dict[str, Any]],
    embeddings: np.ndarray,
    manifest_extra: dict[str, Any],
) -> None:
    folder = HERE / name
    folder.mkdir(parents=True, exist_ok=True)
    write_text(
        folder / "prompts.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    np.save(folder / "embeddings.npy", np.asarray(embeddings, dtype=np.float32))
    manifest = {
        **manifest_extra,
        "rows": len(rows),
        "sha256": {f: file_sha256(folder / f) for f in ("prompts.jsonl", "embeddings.npy")},
    }
    write_text(folder / "manifest.json", json.dumps(manifest, indent=2))
