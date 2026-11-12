"""Public human-labeled pairs for calibrating the judge (PRD §10.2: "human labels come from
`full`-logging tenants or public datasets"; §10.3: adversarial cases in the calibration set).

    python -m evals.judge.public.data      # downloads, converts, writes the frozen files

Two sources, both with a human (or author-verified) verdict on every pair:

- **MT-Bench human judgments** (lmsys, CC-BY-4.0): expert votes on pairs of answers from six
  2023 models. A vote is a *preference* (A better, B better, tie), not quite Relay's question
  ("is one *materially* worse?"), so some disagreement is the question's, not the judge's.
- **LLMBar** (princeton-nlp, MIT): one output follows the instruction and the other does not,
  checked by the authors. The adversarial subsets are built to fool LLM judges.

How a public pair becomes a Relay pair: the **candidate** is the side that plays "cheap":
- MT-Bench: always the weaker model of the two (by the fixed ranking in ``STRENGTH``), so the
  label reads as in cheap-vs-strong: "loss" = the stronger model's answer was preferred;
- LLMBar: a seeded coin picks the side, so losses and wins both occur.

Only first-turn MT-Bench votes are used: Relay's rubric judges one request and one answer.
A pair voted on by several annotators keeps its first vote as *the* label; the others are kept
apart, to measure how often two humans agree (the ceiling for the judge).

The output files are frozen and hashed, like the other eval data: a re-run of this script
must reproduce them byte for byte (fixed seeds, fixed timestamps).
"""

import hashlib
import json
import random
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from relay.evidence.cohorts import cohort_of
from relay.evidence.pairs import EvalPair, offline_comparison, prompt_hash, write_jsonl

HERE = Path(__file__).parent

# Strongest first (MT-Bench's own leaderboard order for these six).
STRENGTH = ("gpt-4", "claude-v1", "gpt-3.5-turbo", "vicuna-13b-v1.2", "alpaca-13b", "llama-13b")
# Fixed, so the frozen files never change on a re-fetch. Not when anyone labeled anything.
TS = datetime(2026, 11, 7, tzinfo=UTC)

N_MTBENCH = 300
LLMBAR_SAMPLE = {
    "Natural": 60,
    "Adversarial/Neighbor": 15,
    "Adversarial/GPTInst": 15,
    "Adversarial/GPTOut": 15,
    "Adversarial/Manual": 15,
}
SEED = 0

MTBENCH_ROWS = "https://datasets-server.huggingface.co/rows"
MTBENCH_DATASET = "lmsys/mt_bench_human_judgments"
LLMBAR_RAW = "https://raw.githubusercontent.com/princeton-nlp/LLMBar/main/Dataset/LLMBar"


@dataclass(frozen=True)
class PublicSet:
    pairs: list[EvalPair]
    labels: list[dict[str, Any]]  # one per pair: the human label calibrate reads
    extra_votes: list[dict[str, Any]]  # further annotators' votes, for human-human agreement


def _label_row(pair_id: str, label: str, annotator: str) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "label": label,
        "annotator": annotator,
        "is_relabel": False,
        "ts": TS.isoformat(),
    }


def _rank(model: str) -> int:
    if model not in STRENGTH:
        raise ValueError(f"unknown model {model!r}: add it to STRENGTH before using its votes")
    return STRENGTH.index(model)


def _mtbench_label(vote: Mapping[str, Any], candidate: str) -> str:
    winner = str(vote["winner"])
    if winner.startswith("tie"):
        return "tie"
    if winner not in ("model_a", "model_b"):
        raise ValueError(f"unexpected winner {winner!r}")
    return "win" if vote[winner] == candidate else "loss"


def _answer(vote: Mapping[str, Any], model: str) -> str:
    side = "conversation_a" if vote["model_a"] == model else "conversation_b"
    return str(vote[side][1]["content"])


def mtbench_set(rows: Iterable[Mapping[str, Any]], *, n: int, seed: int) -> PublicSet:
    # Group the first-turn votes by pair (question, candidate, reference), in data order.
    groups: dict[tuple[int, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["turn"] != 1:
            continue
        a, b = str(row["model_a"]), str(row["model_b"])
        candidate, reference = (a, b) if _rank(a) > _rank(b) else (b, a)
        groups.setdefault((int(row["question_id"]), candidate, reference), []).append(row)

    # Spread the sample over questions: shuffle each question's pairs (seeded), then take one
    # from each question in turn, so no question is over-represented.
    rng = random.Random(seed)
    by_question: dict[int, list[tuple[int, str, str]]] = {}
    for key in sorted(groups):
        by_question.setdefault(key[0], []).append(key)
    queues = [by_question[q] for q in sorted(by_question)]
    for queue in queues:
        rng.shuffle(queue)
    chosen: list[tuple[int, str, str]] = []
    while len(chosen) < n and any(queues):
        for queue in queues:
            if queue and len(chosen) < n:
                chosen.append(queue.pop())

    pairs: list[EvalPair] = []
    labels: list[dict[str, Any]] = []
    extra: list[dict[str, Any]] = []
    for qid, candidate, reference in chosen:
        votes = groups[(qid, candidate, reference)]
        first = votes[0]
        prompt = str(first["conversation_a"][0]["content"])
        pair_id = f"mtbench-{qid}-{candidate}-vs-{reference}"
        pairs.append(
            EvalPair(
                pair_id=pair_id,
                prompt_hash=prompt_hash(prompt),
                cohort=cohort_of(prompt).key,
                comparison=offline_comparison(candidate, reference),
                candidate_model=candidate,
                reference_model=reference,
                judge_label=None,
                judge_model="",  # not judged yet: run.py fills these in
                judge_prompt_version="",
                position_swapped=False,
                text_retained=True,
                ts=TS,
                source="mtbench",
                prompt_text=prompt,
                candidate_text=_answer(first, candidate),
                reference_text=_answer(first, reference),
            )
        )
        for i, v in enumerate(votes):
            row = _label_row(pair_id, _mtbench_label(v, candidate), f"mtbench:{v['judge']}")
            (labels if i == 0 else extra).append(row)
    return PublicSet(pairs, labels, extra)


def llmbar_set(
    subsets: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    per_subset: Mapping[str, int],
    seed: int,
) -> tuple[list[EvalPair], list[dict[str, Any]]]:
    rng = random.Random(seed)
    pairs: list[EvalPair] = []
    labels: list[dict[str, Any]] = []
    for name in sorted(subsets):
        items = subsets[name]
        slug = name.lower().replace("/", "-")  # "Adversarial/Neighbor" -> "adversarial-neighbor"
        source = "llmbar-adversarial" if name.startswith("Adversarial") else "llmbar-natural"
        for index in sorted(rng.sample(range(len(items)), per_subset[name])):
            item = items[index]
            side = 1 if rng.random() < 0.5 else 2  # which output plays the candidate
            prompt = str(item["input"])
            pair_id = f"llmbar-{slug}-{index:03d}"
            candidate, reference = f"llmbar/output-{side}", f"llmbar/output-{3 - side}"
            pairs.append(
                EvalPair(
                    pair_id=pair_id,
                    prompt_hash=prompt_hash(prompt),
                    cohort=cohort_of(prompt).key,
                    comparison=offline_comparison(candidate, reference),
                    candidate_model=candidate,
                    reference_model=reference,
                    judge_label=None,
                    judge_model="",
                    judge_prompt_version="",
                    position_swapped=False,
                    text_retained=True,
                    ts=TS,
                    source=source,
                    prompt_text=prompt,
                    candidate_text=str(item[f"output_{side}"]),
                    reference_text=str(item[f"output_{3 - side}"]),
                )
            )
            # LLMBar's label names the objectively better output; there are no ties.
            label = "win" if int(item["label"]) == side else "loss"
            labels.append(_label_row(pair_id, label, "llmbar:authors"))
    return pairs, labels


def judging_order(pairs: Sequence[EvalPair], *, seed: int) -> list[EvalPair]:
    """Sources interleaved at random (seeded). If a judge's quota ends partway and a fallback
    judge takes over, each judge's share is then still a fair sample of both datasets."""
    order = list(pairs)
    random.Random(seed).shuffle(order)
    return order


# ---------------------------------------------------------------- download (network)
def fetch_mtbench(client: httpx.Client) -> list[dict[str, Any]]:
    """Every row of the `human` split, through the Hugging Face rows API (JSON, so no Parquet
    reader is needed), 100 rows a page."""
    rows: list[dict[str, Any]] = []
    total = None
    while total is None or len(rows) < total:
        params: dict[str, str | int] = {
            "dataset": MTBENCH_DATASET,
            "config": "default",
            "split": "human",
            "offset": len(rows),
            "length": 100,
        }
        body = _get_json(client, MTBENCH_ROWS, params)
        total = int(body["num_rows_total"])
        batch = [r["row"] for r in body["rows"]]
        if not batch:
            raise RuntimeError(f"MT-Bench rows API returned no rows at offset {len(rows)}")
        rows.extend(batch)
    return rows


def fetch_llmbar(client: httpx.Client) -> dict[str, list[dict[str, Any]]]:
    return {name: _get_json(client, f"{LLMBAR_RAW}/{name}/dataset.json") for name in LLMBAR_SAMPLE}


def _get_json(
    client: httpx.Client,
    url: str,
    params: Mapping[str, str | int] | None = None,
    *,
    attempts: int = 6,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """GET with patience: a 429 or 5xx from a public host is waited out (its Retry-After, else
    exponential backoff) instead of crashing the fetch partway (the rows API rate-limits)."""
    for attempt in range(attempts):
        response = client.get(url, params=params)
        if response.status_code != 429 and response.status_code < 500:
            response.raise_for_status()
            return response.json()
        if attempt == attempts - 1:
            response.raise_for_status()
        hinted = response.headers.get("retry-after", "")
        sleep(float(hinted) if hinted.isdigit() else min(60.0, 2.0 * 2**attempt))
    raise AssertionError("unreachable")


def _write_rows(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        mt_rows = fetch_mtbench(client)
        llmbar = fetch_llmbar(client)
    mt = mtbench_set(mt_rows, n=N_MTBENCH, seed=SEED)
    lb_pairs, lb_labels = llmbar_set(llmbar, per_subset=LLMBAR_SAMPLE, seed=SEED)
    pairs = judging_order([*mt.pairs, *lb_pairs], seed=SEED)
    write_jsonl(pairs, HERE / "pairs.jsonl")
    _write_rows([*mt.labels, *lb_labels], HERE / "labels.jsonl")
    _write_rows(mt.extra_votes, HERE / "extra_votes.jsonl")
    print(
        f"{len(mt_rows)} MT-Bench votes -> {len(mt.pairs)} pairs ({len(mt.extra_votes)} extra "
        f"votes); {sum(len(v) for v in llmbar.values())} LLMBar items -> {len(lb_pairs)} pairs"
    )
    for name in ("pairs.jsonl", "labels.jsonl", "extra_votes.jsonl"):
        digest = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        print(f"sha256 {digest}  {name}")


if __name__ == "__main__":
    main()
