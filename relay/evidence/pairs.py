"""Evaluation pairs: a candidate answer and a reference answer to the same prompt, judged.

``EvalPair`` is one row of PRD §9.4's ``eval_pairs`` table, wherever it came from. It is the
only type for that row (found in review: the shadow stage's ``ShadowPair`` and this file's
pair described the same table with different field names, and had two insert paths):

- **live pairs** (``source="shadow"``): written by the shadow stage through the evidence store;
- **offline sets** (``pilot``, ``public``, ``sim``): built by ``evals/judge/build_pairs.py``
  and kept as JSONL files, the frozen input to calibration. A file is hashed, so a calibration
  report names exactly the data it came from.

The labeling tool reads the same JSONL, exported from the store or written by the builder.
"""

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalPair:
    pair_id: str
    prompt_hash: str
    cohort: str
    # cheap_vs_strong | fallback:a->b | band:lo-hi | canary:v; "<candidate>_vs_<reference>"
    # for an offline set.
    comparison: str
    candidate_model: str
    reference_model: str
    judge_label: str | None  # win | tie | loss | None (abstained)
    judge_model: str
    judge_prompt_version: str
    position_swapped: bool  # the judge saw the candidate as response B
    text_retained: bool
    ts: datetime
    source: str  # shadow | pilot | public | sim
    request_id: str | None = None  # live traffic only
    route: str | None = None  # live traffic only
    policy_version: str | None = None
    # The texts, for human labeling. Live pairs keep them only for `full`-logging tenants.
    prompt_text: str | None = None
    candidate_text: str | None = None
    reference_text: str | None = None
    cost_nano_usd: int = 0
    # When the three texts must be deleted (the tenant's retention_days); None: no deadline.
    text_expires_at: datetime | None = None

    @property
    def judge_loss(self) -> bool | None:
        return None if self.judge_label is None else self.judge_label == "loss"


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def offline_comparison(candidate: str, reference: str) -> str:
    return f"{candidate}_vs_{reference}"


# JSONL files written before the two pair types were merged use these names. Read, never
# written, so the committed files (and the hashes reports cite) stay valid.
_OLD_NAMES = {
    "id": "pair_id",
    "prompt": "prompt_text",
    "candidate_response": "candidate_text",
    "reference_response": "reference_text",
}


def to_json(p: EvalPair) -> dict[str, Any]:
    out = asdict(p)
    out["ts"] = p.ts.isoformat()
    out["text_expires_at"] = p.text_expires_at.isoformat() if p.text_expires_at else None
    return out


def from_json(data: Mapping[str, Any]) -> EvalPair:
    row = {_OLD_NAMES.get(k, k): v for k, v in data.items()}
    row.setdefault("comparison", offline_comparison(row["candidate_model"], row["reference_model"]))
    row["ts"] = datetime.fromisoformat(row["ts"])
    if row.get("text_expires_at"):
        row["text_expires_at"] = datetime.fromisoformat(row["text_expires_at"])
    return EvalPair(**row)


def write_jsonl(pairs: Iterable[EvalPair], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for p in pairs:
            f.write(json.dumps(to_json(p), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[EvalPair]:
    with open(path, encoding="utf-8") as f:
        return [from_json(json.loads(line)) for line in f if line.strip()]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
