"""Evaluation pairs: a candidate answer and a reference answer to the same prompt, judged.

``EvalPair`` is the row of PRD §9.4's ``eval_pairs`` table. A JSONL file of them is the frozen
input to calibration: hashed, so a calibration report names exactly the data it came from.

Two commands:

    python -m relay.evidence.pairs build --prompts P.jsonl --candidate groq/gpt-oss-20b \\
        --reference groq/gpt-oss-120b --judge groq/gpt-oss-120b --limit 30 --out pairs.jsonl
    python -m relay.evidence.pairs import --pairs pairs.jsonl     # into Postgres

``build`` calls **real models** when the models are real, so it is a deliberate, counted
operation, never part of a test: it makes at most ``--max-calls`` calls (default 100) and
refuses to start if the plan needs more. It never prints a key.
"""

import argparse
import asyncio
import hashlib
import json
import random
from collections.abc import AsyncGenerator, Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from relay.config.loader import load_config
from relay.evidence.cohorts import cohort_of
from relay.evidence.judge import Judge, JudgeResult, LLMJudge
from relay.evidence.rubric import JUDGE_PROMPT_VERSION
from relay.providers.base import ProviderAdapter
from relay.providers.registry import ProviderKeys, build_adapters
from relay.schemas.chat import ChatChunk, ChatRequest, ChatResponse
from relay.schemas.errors import ErrorClass, ProviderError
from relay.settings import Settings


@dataclass(frozen=True)
class EvalPair:
    id: str
    prompt_hash: str
    prompt: str  # kept in the JSONL for human labeling; see text_retained
    cohort: str
    candidate_model: str
    reference_model: str
    candidate_response: str
    reference_response: str
    judge_label: str | None  # win | tie | loss | None (abstained)
    judge_model: str
    judge_prompt_version: str
    position_swapped: bool
    text_retained: bool
    ts: str
    source: str  # "pilot" | "sim" | "public": where the prompts came from

    @property
    def judge_loss(self) -> bool | None:
        return None if self.judge_label is None else self.judge_label == "loss"


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def write_jsonl(pairs: Iterable[EvalPair], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for p in pairs:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[EvalPair]:
    with open(path, encoding="utf-8") as f:
        return [EvalPair(**json.loads(line)) for line in f if line.strip()]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QuotaReached(Exception):
    """The provider says "no more today" (or asks for a wait too long to sit through). The
    run stops cleanly, keeps what it has, and can be resumed later (``--resume``)."""


# A dropped connection or a 5xx on a long batch run is retried a few times (a one-off network
# blip ended a 150-prompt run once); a persistent failure still raises.
TRANSIENT = frozenset({ErrorClass.SERVER_ERROR, ErrorClass.TIMEOUT, ErrorClass.OVERLOADED})
MAX_WAIT_S = 120.0  # a Retry-After longer than this is treated as a quota, not a hiccup


class PatientAdapter:
    """Wraps an adapter for the pilot: on a 429 it waits for the provider's own Retry-After
    and tries again, a few times, instead of crashing a run that has already spent calls.

    Free tiers have small per-minute token caps (Groq's gpt-oss-120b: 8,000 tokens/min), so
    a 30-pair run *will* hit them. Only rate limits are retried; any other error propagates.
    Relay's gateway never does this on a request path (it has the pause state and fallbacks);
    this is an offline batch tool where waiting is the right answer.
    """

    def __init__(
        self, inner: ProviderAdapter, *, max_waits: int = 8, max_transient: int = 3
    ) -> None:
        self._inner = inner
        self._max_waits = max_waits
        self._max_transient = max_transient
        self.name = inner.name
        self.rate_limit_waits = 0
        self.quota_hit = False

    async def complete(self, req: ChatRequest) -> ChatResponse:
        for attempt in range(self._max_waits + 1):
            try:
                return await self._inner.complete(req)
            except ProviderError as e:
                # The provider's own Retry-After when it sends one. Without one (OpenRouter's
                # shared free pool just says "retry shortly"), back off exponentially, capped:
                # 2, 4, 8, 16, 30, 30, ... seconds, so a congested pool gets time to clear.
                hinted = e.signal.retry_after_s if e.signal else None
                wait = hinted if hinted else min(30.0, 2.0 * 2**attempt)
                if e.error_class is ErrorClass.QUOTA_EXHAUSTED or wait > MAX_WAIT_S:
                    self.quota_hit = True
                    raise QuotaReached(f"{self.name}: {e.message}") from e
                transient = e.error_class in TRANSIENT
                if (e.error_class is not ErrorClass.RATE_LIMITED and not transient) or attempt == (
                    self._max_transient if transient else self._max_waits
                ):
                    raise
                self.rate_limit_waits += 1
                await asyncio.sleep(wait + 0.5)
        raise AssertionError("unreachable")

    def stream(self, req: ChatRequest) -> AsyncGenerator[ChatChunk, None]:
        return self._inner.stream(req)

    async def aclose(self) -> None:
        await self._inner.aclose()


async def make_pair(
    *,
    index: int,
    prompt: str,
    candidate: ProviderAdapter,
    candidate_upstream: str,
    candidate_name: str,
    reference: ProviderAdapter,
    reference_upstream: str,
    reference_name: str,
    judge: Judge,
    judge_model: str,
    max_tokens: int,
    source: str,
) -> EvalPair:
    def req(model: str) -> ChatRequest:
        return ChatRequest.model_validate(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            }
        )

    cand = (await candidate.complete(req(candidate_upstream))).choices[0].message.content or ""
    ref = (await reference.complete(req(reference_upstream))).choices[0].message.content or ""
    try:
        result = await judge.judge(request=prompt, candidate=cand, reference=ref)
    except QuotaReached:
        # The judge's daily quota ran out *after* both answers were paid for: keep the pair
        # with no verdict (an abstention). `pairs repair` judges it later for one call.
        result = JudgeResult(None, False, judge_model, JUDGE_PROMPT_VERSION, "quota_reached")
    return EvalPair(
        id=f"{source}-{index:04d}",
        prompt_hash=prompt_hash(prompt),
        prompt=prompt,
        cohort=cohort_of(prompt).key,
        candidate_model=candidate_name,
        reference_model=reference_name,
        candidate_response=cand,
        reference_response=ref,
        judge_label=result.label,
        judge_model=judge_model,
        judge_prompt_version=result.prompt_version,
        position_swapped=result.position_swapped,
        text_retained=True,
        ts=datetime.now(UTC).isoformat(),
        source=source,
    )


def read_prompts(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["prompt"] for line in lines if line.strip()]


def load_existing(path: Path) -> list[EvalPair]:
    """The pairs already written by an earlier (stopped) run, or none if there is no file."""
    return read_jsonl(path) if path.exists() else []


def planned_calls(n_prompts: int, *, both_orders: bool) -> int:
    """candidate + reference + judge (twice with both_orders) per prompt."""
    return n_prompts * (3 + (1 if both_orders else 0))


def needs_repair(p: EvalPair) -> tuple[bool, bool, bool]:
    """(regenerate candidate, regenerate reference, re-judge): an empty answer means the
    model spent its whole token budget on hidden reasoning; an abstention means the judge did."""
    cand_empty = not p.candidate_response.strip()
    ref_empty = not p.reference_response.strip()
    return cand_empty, ref_empty, cand_empty or ref_empty or p.judge_label is None


def repair_calls(pairs: Sequence[EvalPair]) -> int:
    total = 0
    for p in pairs:
        cand, ref, rejudge = needs_repair(p)
        total += cand + ref + rejudge
    return total


async def repair_pair(
    p: EvalPair,
    *,
    candidate: ProviderAdapter,
    candidate_upstream: str,
    reference: ProviderAdapter,
    reference_upstream: str,
    judge: Judge,
    max_tokens: int,
) -> EvalPair:
    """Regenerate empty answers (with a bigger cap) and judge again where needed."""
    cand_empty, ref_empty, rejudge = needs_repair(p)
    if not rejudge:
        return p

    def req(model: str) -> ChatRequest:
        return ChatRequest.model_validate(
            {
                "model": model,
                "messages": [{"role": "user", "content": p.prompt}],
                "max_tokens": max_tokens,
            }
        )

    cand, ref = p.candidate_response, p.reference_response
    if cand_empty:
        cand = (await candidate.complete(req(candidate_upstream))).choices[0].message.content or ""
    if ref_empty:
        ref = (await reference.complete(req(reference_upstream))).choices[0].message.content or ""
    result = await judge.judge(request=p.prompt, candidate=cand, reference=ref)
    return replace(
        p,
        candidate_response=cand,
        reference_response=ref,
        judge_label=result.label,
        judge_prompt_version=result.prompt_version,
        position_swapped=result.position_swapped,
    )


async def repair(args: argparse.Namespace) -> None:
    pairs = read_jsonl(Path(args.pairs))
    needed = repair_calls(pairs)
    if needed > args.max_calls:
        raise SystemExit(f"repair needs {needed} model calls, above --max-calls {args.max_calls}")
    config = load_config(args.config)
    adapters = build_adapters(ProviderKeys(), Settings(), config.providers)

    def resolve(name: str) -> tuple[ProviderAdapter, str]:
        spec = config.models[name]
        if spec.provider not in adapters:
            raise SystemExit(f"no API key configured for provider {spec.provider!r} ({name})")
        return PatientAdapter(adapters[spec.provider]), spec.provider_model

    cand_adapter, cand_up = resolve(args.candidate)
    ref_adapter, ref_up = resolve(args.reference)
    judge_adapter, judge_up = resolve(args.judge)
    judge = LLMJudge(
        judge_adapter,
        model=judge_up,
        name=args.judge,
        rng=random.Random(args.seed),
        max_tokens=args.judge_max_tokens,
    )
    n_pairs = sum(needs_repair(p)[2] for p in pairs)
    print(f"{n_pairs} pairs to repair, {needed} calls (cap {args.max_calls})")
    fixed: list[EvalPair] = []
    try:
        for i, p in enumerate(pairs):
            fixed.append(
                await repair_pair(
                    p,
                    candidate=cand_adapter,
                    candidate_upstream=cand_up,
                    reference=ref_adapter,
                    reference_upstream=ref_up,
                    judge=judge,
                    max_tokens=args.max_tokens,
                )
            )
            write_jsonl(fixed + pairs[i + 1 :], Path(args.pairs))
    finally:
        for adapter in {id(a): a for a in adapters.values()}.values():
            await adapter.aclose()
    left = sum(p.judge_label is None for p in fixed)
    empty = sum(not x.candidate_response.strip() or not x.reference_response.strip() for x in fixed)
    print(f"repaired; {left} judge abstentions and {empty} empty answers remain")


async def build(args: argparse.Namespace) -> None:
    all_prompts = read_prompts(Path(args.prompts))[: args.limit]
    out = Path(args.out)
    # --resume: keep the pairs already in the output file and only do the rest. A free-tier
    # daily quota stops a run partway; this is how it finishes the next day without paying
    # again for what is already done.
    pairs: list[EvalPair] = load_existing(out) if args.resume else []
    done = {p.prompt_hash for p in pairs}
    todo = [(i, p) for i, p in enumerate(all_prompts) if prompt_hash(p) not in done]
    needed = planned_calls(len(todo), both_orders=args.both_orders)
    if needed > args.max_calls:
        raise SystemExit(
            f"this plan needs {needed} model calls, above --max-calls {args.max_calls}; "
            "raise it deliberately or lower --limit"
        )
    config = load_config(args.config)
    adapters = build_adapters(ProviderKeys(), Settings(), config.providers)

    def resolve(name: str) -> tuple[ProviderAdapter, str]:
        spec = config.models[name]
        if spec.provider not in adapters:
            raise SystemExit(f"no API key configured for provider {spec.provider!r} ({name})")
        return PatientAdapter(adapters[spec.provider]), spec.provider_model

    cand_adapter, cand_up = resolve(args.candidate)
    ref_adapter, ref_up = resolve(args.reference)
    judge_adapter, judge_up = resolve(args.judge)
    judge = LLMJudge(
        judge_adapter,
        model=judge_up,
        name=args.judge,
        rng=random.Random(args.seed),
        both_orders=args.both_orders,
        max_tokens=args.judge_max_tokens,
    )
    print(
        f"{len(todo)} prompts to do ({len(pairs)} already done), "
        f"{needed} model calls planned (cap {args.max_calls})"
    )
    patient = [
        a for a in (cand_adapter, ref_adapter, judge_adapter) if isinstance(a, PatientAdapter)
    ]
    stopped = ""
    try:
        for i, prompt in todo:
            pairs.append(
                await make_pair(
                    index=i,
                    prompt=prompt,
                    candidate=cand_adapter,
                    candidate_upstream=cand_up,
                    candidate_name=args.candidate,
                    reference=ref_adapter,
                    reference_upstream=ref_up,
                    reference_name=args.reference,
                    judge=judge,
                    judge_model=args.judge,
                    max_tokens=args.max_tokens,
                    source=args.source,
                )
            )
            write_jsonl(pairs, out)  # saved after every pair: a crash loses one
            print(f"  {len(pairs)}/{len(all_prompts)}", flush=True)
            if any(a.quota_hit for a in patient):
                stopped = "a provider's quota ran out"
                break
    except QuotaReached as e:
        stopped = str(e)
    finally:
        for adapter in {id(a): a for a in adapters.values()}.values():
            await adapter.aclose()
    write_jsonl(pairs, out)
    abstained = sum(p.judge_label is None for p in pairs)
    print(f"wrote {len(pairs)} pairs to {args.out} ({abstained} judge abstentions)")
    if stopped:
        print(f"STOPPED EARLY ({stopped}). Resume later with --resume, then run `pairs repair`.")
    print(f"sha256 {file_sha256(Path(args.out))}")


async def import_to_postgres(pairs: Sequence[EvalPair], dsn: str) -> int:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        await conn.executemany(
            "INSERT INTO eval_pairs (pair_id, prompt_hash, cohort, comparison, candidate_model,"
            " reference_model, judge_label, judge_model, judge_prompt_version,"
            " position_swapped, text_retained, ts, source)"
            " VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13) ON CONFLICT (pair_id) DO NOTHING",
            [
                (
                    p.id,
                    p.prompt_hash,
                    p.cohort,
                    f"{p.candidate_model}_vs_{p.reference_model}",
                    p.candidate_model,
                    p.reference_model,
                    p.judge_label,
                    p.judge_model,
                    p.judge_prompt_version,
                    p.position_swapped,
                    p.text_retained,
                    datetime.fromisoformat(p.ts),
                    p.source,
                )
                for p in pairs
            ],
        )
    finally:
        await conn.close()
    return len(pairs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m relay.evidence.pairs", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="generate and judge pairs with real models (counted calls)")
    b.add_argument("--prompts", required=True, help="JSONL, one {'prompt': ...} per line")
    b.add_argument("--candidate", required=True, help="Relay model name, e.g. groq/gpt-oss-20b")
    b.add_argument("--reference", required=True)
    b.add_argument("--judge", required=True)
    b.add_argument("--limit", type=int, default=30)
    b.add_argument("--max-tokens", type=int, default=256)
    b.add_argument(
        "--judge-max-tokens",
        type=int,
        default=16,
        help="16 suits a non-reasoning judge; reasoning models (gpt-oss) need ~200 so their "
        "hidden reasoning leaves room for the verdict",
    )
    b.add_argument("--max-calls", type=int, default=100)
    b.add_argument("--both-orders", action="store_true")
    b.add_argument("--source", default="pilot")
    b.add_argument(
        "--resume", action="store_true", help="keep pairs already in --out; do only the rest"
    )
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--config", default=None)
    b.add_argument("--out", required=True)
    r = sub.add_parser("repair", help="regenerate empty answers and re-judge abstentions")
    r.add_argument("--pairs", required=True, help="a pairs JSONL, rewritten in place")
    r.add_argument("--candidate", required=True)
    r.add_argument("--reference", required=True)
    r.add_argument("--judge", required=True)
    r.add_argument("--max-tokens", type=int, default=1000)
    r.add_argument("--judge-max-tokens", type=int, default=800)
    r.add_argument("--max-calls", type=int, default=40)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--config", default=None)
    i = sub.add_parser("import", help="load a pairs JSONL into Postgres (RELAY_DATABASE_URL)")
    i.add_argument("--pairs", required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        asyncio.run(build(args))
    elif args.command == "repair":
        asyncio.run(repair(args))
    else:
        dsn = Settings().database_url
        if not dsn:
            raise SystemExit("RELAY_DATABASE_URL is not set")
        n = asyncio.run(import_to_postgres(read_jsonl(Path(args.pairs)), dsn))
        print(f"imported {n} pairs")


if __name__ == "__main__":
    main()
