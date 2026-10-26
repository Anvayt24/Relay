"""Build the frozen cache-evaluation data: labeled prompt pairs, and their embeddings.

    python -m evals.cache.build            # needs the embedding model (a one-time download)

Writes, next to this file:
- ``pairs.jsonl``      the labeled pairs (cached prompt, new prompt, label, family, split)
- ``embeddings.npy``   one L2-normalized float32 vector per unique prompt, in ``prompts.json`` order
- ``prompts.json``     the unique prompts, indexing ``embeddings.npy``
- ``manifest.json``    sha256 of each file, the seed, the model and library versions

The evaluation (`evaluate.py`) reads only these frozen files: it needs no model and no network,
so it is deterministic and CI can run it. Rebuild only to change the data, then re-run
`make eval-cache` and commit the new ``metrics.json`` (CLAUDE.md: any change to the cache
re-runs the eval).

**Leakage.** Instances (one fact, e.g. "15% of 200") are split into the tune and test pools
*first*, by hash, and pairs are drawn only within a pool. So no prompt, and no fact, appears on
both sides of the split: the threshold chosen on tune can't have seen the test prompts.
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import random
from pathlib import Path

import numpy as np

from evals.cache.families import FAMILIES, Family
from relay.cache.embed import MODEL_NAME, FastEmbedder

HERE = Path(__file__).parent


def split_of(family: str, instance: dict[str, str]) -> str:
    """Stable 50/50 split of instances, independent of generation order."""
    key = family + "|" + json.dumps(instance, sort_keys=True)
    return "tune" if hashlib.sha256(key.encode()).digest()[0] % 2 == 0 else "test"


def render(family: Family, template: int, instance: dict[str, str]) -> str:
    return family.templates[template].format(**instance)


def _shares_a_value(a: dict[str, str], b: dict[str, str]) -> bool:
    return any(a[k] == b[k] for k in a) and a != b


def generate(seed: int = 0) -> list[dict[str, object]]:
    rng = random.Random(seed)
    pairs: list[dict[str, object]] = []
    pools: dict[tuple[str, str], list[dict[str, str]]] = {}
    for fam in FAMILIES:
        for inst in fam.instances:
            pools.setdefault((fam.name, split_of(fam.name, inst)), []).append(inst)

    for fam in FAMILIES:
        for split in ("tune", "test"):
            pool = pools[(fam.name, split)]
            for inst in pool:
                # Positives: the same fact in two different wordings.
                for a, b in rng.sample([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], 3):
                    pairs.append(_pair(fam, split, "positive", render(fam, a, inst),
                                       render(fam, b, inst), 1))  # fmt: skip
                # Hard negatives: one fact changed, preferably sharing the rest of the prompt.
                near = [o for o in pool if _shares_a_value(inst, o)] or [
                    o for o in pool if o != inst
                ]
                for other in rng.sample(near, min(3, len(near))):
                    t = rng.randrange(len(fam.templates))
                    u = t if rng.random() < 0.5 else rng.randrange(len(fam.templates))
                    pairs.append(_pair(fam, split, "near_miss", render(fam, t, inst),
                                       render(fam, u, other), 0))  # fmt: skip
                # Easy negative: an unrelated prompt from another family, same split.
                other_fam = rng.choice([f for f in FAMILIES if f is not fam])
                other_inst = rng.choice(pools[(other_fam.name, split)])
                pairs.append(_pair(fam, split, "unrelated", render(fam, 0, inst),
                                   render(other_fam, rng.randrange(4), other_inst), 0))  # fmt: skip
    return pairs


def _pair(fam: Family, split: str, kind: str, a: str, b: str, label: int) -> dict[str, object]:
    return {"family": fam.name, "split": split, "kind": kind, "a": a, "b": b, "label": label}


def write_text(name: str, text: str) -> None:
    # newline="\n": Path.write_text on Windows would emit CRLF, git stores LF, and the sha256 in
    # the manifest would then differ between a Windows checkout and Linux CI (the manifest test
    # caught exactly this).
    with open(HERE / name, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


async def build(seed: int) -> None:
    pairs = generate(seed)
    prompts = sorted({p for pair in pairs for p in (pair["a"], pair["b"])})  # type: ignore[type-var]
    embedder = FastEmbedder()
    await embedder.start()
    vectors = await embedder.embed_many(prompts)
    await embedder.close()

    write_text("pairs.jsonl", "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs))
    write_text("prompts.json", json.dumps(prompts, ensure_ascii=False))
    np.save(HERE / "embeddings.npy", np.asarray(vectors, dtype=np.float32))
    manifest = {
        "seed": seed,
        "embedding_model": MODEL_NAME,
        "fastembed_version": importlib.metadata.version("fastembed"),
        "pairs": len(pairs),
        "unique_prompts": len(prompts),
        "label": "positive iff the cached answer is an acceptable answer to the new prompt "
        "(constructed: same fact, different wording); not human-judged",
        "sha256": {
            f: hashlib.sha256((HERE / f).read_bytes()).hexdigest()
            for f in ("pairs.jsonl", "prompts.json", "embeddings.npy")
        },
    }
    write_text("manifest.json", json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "sha256"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m evals.cache.build", description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(build(args.seed))


if __name__ == "__main__":
    main()
