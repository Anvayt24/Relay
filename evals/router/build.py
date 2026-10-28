"""Build the frozen simulated-world router dataset.

    python -m evals.router.build            # needs the embedding model (a one-time download)

Steps (PRD FR-7 "Data splits"):
1. generate ~3,000 prompts from the world (``world.py``) and give each its truth and its noisy
   judge label;
2. embed every prompt with bge-small;
3. **remove near-duplicates by clustering them**: two prompts are in one cluster if they are
   paraphrases of the same question (same ``instance``) or their embeddings have cosine
   similarity > 0.95 (union-find); a cluster is never split;
4. assign **whole clusters** to train / validation / test (60 / 20 / 20) by hash, so a paraphrase
   can't sit on both sides of a split and flatter the router;
5. mark a "gold" subset of each held-out split (the labeled set PPI needs; here the truth plays
   the human), disjoint from the judge-only prompts.

Rebuild only to change the data; then re-run ``make train-router`` and ``make eval-router`` and
commit the artifact and ``metrics.json`` (CLAUDE.md: a router change re-runs the eval).
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
from typing import Any

import numpy as np

from evals.router import world
from evals.router.data import write_dataset
from relay.cache.embed import MODEL_NAME, FastEmbedder

SIM = "sim"
NEAR_DUPLICATE_COSINE = 0.95
GOLD_PER_SPLIT = 150
SPLIT_EDGES = (0.60, 0.80)  # train < 0.60 <= val < 0.80 <= test


def _unit(key: str, salt: str = "") -> float:
    return hashlib.sha256((salt + key).encode("utf-8")).digest()[0] / 256.0


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def cluster(
    instances: list[str], embeddings: np.ndarray, threshold: float = NEAR_DUPLICATE_COSINE
) -> list[int]:
    """Cluster id per prompt: same instance, or cosine similarity above ``threshold``."""
    n = len(instances)
    uf = UnionFind(n)
    first: dict[str, int] = {}
    for i, inst in enumerate(instances):
        if inst in first:
            uf.union(first[inst], i)
        else:
            first[inst] = i
    sims = embeddings @ embeddings.T
    ii, jj = np.where(np.triu(sims > threshold, k=1))
    for a, b in zip(ii.tolist(), jj.tolist(), strict=True):
        uf.union(a, b)
    return [uf.find(i) for i in range(n)]


def assign_splits(clusters: list[int], keys: list[str]) -> list[str]:
    """Whole clusters go to one split, decided by a hash of the cluster's smallest member key."""
    smallest: dict[int, str] = {}
    for c, k in zip(clusters, keys, strict=True):
        if c not in smallest or k < smallest[c]:
            smallest[c] = k
    out = []
    for c in clusters:
        u = _unit(smallest[c], "split")
        out.append("train" if u < SPLIT_EDGES[0] else "val" if u < SPLIT_EDGES[1] else "test")
    return out


def pick_gold(splits: list[str], keys: list[str], n_per_split: int) -> list[bool]:
    """The first ``n_per_split`` prompts of each held-out split by a hash order: a random sample
    that does not depend on generation order."""
    gold = [False] * len(keys)
    for split in ("val", "test"):
        members = sorted(
            (i for i, s in enumerate(splits) if s == split), key=lambda i: _unit(keys[i], "gold")
        )
        for i in members[:n_per_split]:
            gold[i] = True
    return gold


async def build(seed: int) -> None:
    items = world.generate(seed)
    keys = [world.text_id(it.text) for it in items]
    embedder = FastEmbedder()
    await embedder.start()
    vectors = np.asarray(await embedder.embed_many([it.text for it in items]), dtype=np.float32)
    await embedder.close()

    clusters = cluster([it.instance for it in items], vectors.astype(np.float64))
    splits = assign_splits(clusters, keys)
    gold = pick_gold(splits, keys, GOLD_PER_SPLIT)

    rows: list[dict[str, Any]] = []
    for i, it in enumerate(items):
        y_true, y_judge, w = world.labels(seed, it, i)
        rows.append(
            {
                "id": keys[i],
                "text": it.text,
                "family": it.family,
                "instance": it.instance,
                "cluster": clusters[i],
                "split": splits[i],
                "is_gold": gold[i],
                "y_true": int(y_true),
                "y_judge": int(y_judge),
                "w": round(w, 6),
            }
        )
    write_dataset(
        SIM,
        rows,
        vectors,
        {
            "world": "simulated, difficulty fixed in docs/adr/0018 before any router was trained",
            "seed": seed,
            "embedding_model": MODEL_NAME,
            "fastembed_version": importlib.metadata.version("fastembed"),
            "near_duplicate_cosine": NEAR_DUPLICATE_COSINE,
            "gold_per_split": GOLD_PER_SPLIT,
            "label": "y = 1 if the cheap answer is win-or-tie vs strong. y_true is ground truth; "
            "y_judge is the noisy MockJudge label the router trains on. "
            "is_gold: truth known (plays the human) for PPI; disjoint from judge-only prompts.",
        },
    )
    counts = {s: splits.count(s) for s in ("train", "val", "test")}
    print(f"{len(rows)} prompts, {len(set(clusters))} clusters, splits {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m evals.router.build", description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(build(args.seed))


if __name__ == "__main__":
    main()
