"""Have the judge judge the public pairs: real calls, counted, paced, resumable.

    python -u -m evals.judge.public.run --judges gemini/flash-lite,groq/gpt-oss-20b

Each pair costs one judge call (the answers are already in the data). What keeps a long run
on free tiers going:

- **Pacing:** calls are spaced under the provider's per-minute limit, so 429s are the
  exception, not the rhythm.
- **429s:** the patient adapter (``evals/judge/build_pairs.py``) waits out the provider's
  Retry-After and tries again.
- **Fallback judge:** when the first judge's daily quota ends (or it keeps failing), the next
  judge in ``--judges`` takes the remaining pairs. **Each pair records its judge, and each
  judge is calibrated only on its own pairs** (report.py): calibration is a property of one
  judge, and a pooled number would describe no judge at all. The pairs are already in a
  random, mixed order (data.judging_order), so each judge's share is a fair sample.
- **Failed calls are retried, never saved:** LLMJudge turns a failed call into "no verdict".
  Saved, it would look like the judge declined to answer, and the pair would never be judged.
  A real abstention (the judge answered, but with no parseable verdict) is kept: it is data.
- **Resumable and capped:** the file is rewritten after every pair, pairs that already have a
  judge are skipped, and the run stops cleanly at ``--max-calls`` upstream requests (429s
  included).
"""

import argparse
import asyncio
import json
import random
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from evals.judge.build_pairs import PatientAdapter, QuotaReached
from relay.config.loader import load_config
from relay.config.pricing import list_cost_nano
from relay.evidence.judge import Judge, LLMJudge
from relay.evidence.pairs import EvalPair, read_jsonl, write_jsonl
from relay.providers.anthropic import AnthropicAdapter
from relay.providers.base import ProviderAdapter
from relay.providers.registry import ProviderKeys, build_adapters
from relay.schemas.chat import ChatChunk, ChatRequest, ChatResponse, Usage
from relay.schemas.errors import ProviderError
from relay.settings import Settings

HERE = Path(__file__).parent


@dataclass(frozen=True)
class JudgeSettings:
    max_tokens: int  # the reply's budget (a reasoning model spends some before the JSON)
    pace_s: float  # gap between calls, under the provider's per-minute limit
    temperature: float | None = 0.0  # None: send none (Claude Sonnet 5.5 rejects any other)


# Per judge:
# - flash-lite answers directly: 16 tokens hold a v1 verdict. Free tier ~15 requests/minute.
# - gpt-oss-20b reasons before answering (the Phase 9 pilot needed ~200 tokens).
# - Gemini Flash thinks by default, and its thoughts count against max_tokens, so it gets
#   room; on the free tier its tokens cost nothing.
# - Claude Sonnet 5.5 runs with thinking off (NoThinkingAnthropic): 600 tokens hold a reason
#   and a verdict (300 cut a long reason off before its verdict in the probe; only what is
#   used is billed). It is the one paid judge: --max-usd caps it.
JUDGE_SETTINGS: dict[str, JudgeSettings] = {
    "gemini/flash-lite": JudgeSettings(16, 4.5),
    "groq/gpt-oss-20b": JudgeSettings(300, 2.5),
    "gemini/flash": JudgeSettings(2048, 6.5),
    "anthropic/sonnet": JudgeSettings(600, 1.5, temperature=None),
}
DEFAULT_SETTINGS = JudgeSettings(16, 4.0)


class NoThinkingAnthropic(AnthropicAdapter):
    """The Anthropic adapter with thinking switched off, for judge calls only.

    Claude Sonnet 5.5 thinks by default (billed as output tokens), and refuses
    ``{"type": "disabled"}``; ``between_tools`` is how thinking is turned off on it. A verdict
    with a one-line reason needs no hidden reasoning, so this keeps a paid judge cheap. Lives
    here, not in relay/: the gateway forwards requests as clients send them."""

    def _body(self, req: ChatRequest) -> dict[str, Any]:
        body = super()._body(req)
        body["thinking"] = {"type": "between_tools"}
        return body


class CountingAdapter:
    """Counts every upstream request (429 retries included) and remembers whether the last
    one failed, which LLMJudge alone can't tell (it reports a failed call as an abstention)."""

    def __init__(self, inner: ProviderAdapter) -> None:
        self._inner = inner
        self.name = inner.name
        self.calls = 0
        self.failures = 0
        self.last_ok = True

    async def complete(self, req: ChatRequest) -> ChatResponse:
        self.calls += 1
        try:
            response = await self._inner.complete(req)
        except ProviderError:
            self.failures += 1
            self.last_ok = False
            raise
        self.last_ok = True
        return response

    def stream(self, req: ChatRequest) -> AsyncGenerator[ChatChunk, None]:
        return self._inner.stream(req)

    async def aclose(self) -> None:
        await self._inner.aclose()


@dataclass
class JudgeSlot:
    name: str  # the Relay model name, recorded on every pair it judges
    judge: Judge
    counter: CountingAdapter
    pace_s: float = 0.0
    # USD for one call from its reported usage (list price); None for a free judge.
    price: Callable[[Usage], float] | None = None


@dataclass
class RunSummary:
    judged: dict[str, int] = field(default_factory=dict)
    failed_calls: dict[str, int] = field(default_factory=dict)
    # (from judge, to judge, pairs the first had judged when it was left)
    switches: list[tuple[str, str, int]] = field(default_factory=list)
    upstream_calls: int = 0
    spent_usd: float = 0.0  # list price of the calls made, from the usage they reported
    stopped: str | None = None  # why the run ended early; None = every pair judged


async def judge_all(
    pairs: list[EvalPair],
    slots: Sequence[JudgeSlot],
    *,
    max_calls: int,
    save: Callable[[list[EvalPair]], None],
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    failures_before_switch: int = 3,
    progress: Callable[[str], None] = lambda _: None,
    max_usd: float | None = None,
    limit: int | None = None,
) -> RunSummary:
    """Judge every pair that has no judge yet, in order. ``pairs`` is updated in place."""
    summary = RunSummary()
    current = 0  # index of the judge in use
    consecutive_failures = 0
    last_call: float | None = None

    def upstream() -> int:
        return sum(s.counter.calls for s in slots)

    def leave(reason: str) -> None:
        nonlocal current, consecutive_failures
        old = slots[current]
        if current + 1 < len(slots):
            summary.switches.append(
                (old.name, slots[current + 1].name, summary.judged.get(old.name, 0))
            )
            progress(f"switching judge {old.name} -> {slots[current + 1].name}: {reason}")
        current += 1
        consecutive_failures = 0

    judged_now = 0
    for i, pair in enumerate(pairs):
        if pair.judge_model:
            continue  # judged in an earlier run
        if limit is not None and judged_now >= limit:
            summary.stopped = f"limit of {limit} pairs reached (a probe)"
            summary.upstream_calls = upstream()
            return summary
        while True:
            if current >= len(slots):
                summary.stopped = "no judge left (each one's quota ran out or it kept failing)"
                summary.upstream_calls = upstream()
                return summary
            if upstream() >= max_calls:
                summary.stopped = f"call cap of {max_calls} reached"
                summary.upstream_calls = upstream()
                return summary
            if max_usd is not None and summary.spent_usd >= max_usd:
                summary.stopped = f"spend cap of ${max_usd:.2f} reached"
                summary.upstream_calls = upstream()
                return summary
            slot = slots[current]
            if last_call is not None and slot.pace_s > 0:
                wait = slot.pace_s - (clock() - last_call)
                if wait > 0:
                    await sleep(wait)
            last_call = clock()
            try:
                result = await slot.judge.judge(
                    request=pair.prompt_text or "",
                    candidate=pair.candidate_text or "",
                    reference=pair.reference_text or "",
                )
            except QuotaReached as e:
                leave(f"quota: {e}")
                continue
            if slot.price is not None and result.usage is not None:
                summary.spent_usd += slot.price(result.usage)
            if result.label is None and not slot.counter.last_ok:
                # The call failed (LLMJudge reports that as no verdict): try again, and move
                # to the next judge if this one keeps failing.
                summary.failed_calls[slot.name] = summary.failed_calls.get(slot.name, 0) + 1
                consecutive_failures += 1
                if consecutive_failures >= failures_before_switch:
                    leave(f"{consecutive_failures} failed calls in a row")
                continue
            consecutive_failures = 0
            pairs[i] = replace(
                pair,
                judge_label=result.label,
                judge_model=slot.name,
                judge_prompt_version=result.prompt_version,
                position_swapped=result.position_swapped,
                judge_reason=result.reason,
            )
            summary.judged[slot.name] = summary.judged.get(slot.name, 0) + 1
            judged_now += 1
            save(pairs)
            done = sum(bool(p.judge_model) for p in pairs)
            progress(f"{done}/{len(pairs)} judged ({slot.name}: {result.label})")
            break
    summary.upstream_calls = upstream()
    return summary


def _slot(name: str, args: argparse.Namespace, config: Any, adapters: dict[str, Any]) -> JudgeSlot:
    spec = config.models[name]
    if spec.provider not in adapters:
        raise SystemExit(f"no API key configured for provider {spec.provider!r} ({name})")
    settings = JUDGE_SETTINGS.get(name, DEFAULT_SETTINGS)
    adapter: ProviderAdapter = adapters[spec.provider]
    if isinstance(adapter, AnthropicAdapter):
        provider = config.providers[spec.provider]
        adapter = NoThinkingAnthropic(
            spec.provider,
            base_url=provider.base_url,
            api_key=ProviderKeys().get(provider.api_key_env) or "",
            limit_scope=provider.limit_scope,
        )
        adapters[f"{spec.provider}:judge"] = adapter  # closed with the others
    counted = CountingAdapter(adapter)
    judge = LLMJudge(
        PatientAdapter(counted),
        model=spec.provider_model,
        name=name,
        rng=random.Random(args.seed),
        max_tokens=settings.max_tokens,
        prompt_version=args.prompt_version,
        temperature=settings.temperature,
    )

    def price(usage: Usage) -> float:
        return list_cost_nano(spec, usage) / 1e9  # nano-USD to USD, at list price

    return JudgeSlot(name=name, judge=judge, counter=counted, pace_s=settings.pace_s, price=price)


async def _main(args: argparse.Namespace, pairs: list[EvalPair], out: Path) -> None:
    config = load_config(args.config)
    adapters: dict[str, Any] = dict(build_adapters(ProviderKeys(), Settings(), config.providers))
    slots = [_slot(name, args, config, adapters) for name in args.judges.split(",")]
    todo = sum(not p.judge_model for p in pairs)
    print(
        f"{todo} of {len(pairs)} pairs to judge -> {out}; judges {args.judges}; rubric "
        f"{args.prompt_version}; caps {args.max_calls} calls, ${args.max_usd} list price"
    )
    try:
        summary = await judge_all(
            pairs,
            slots,
            max_calls=args.max_calls,
            max_usd=args.max_usd,
            limit=args.limit,
            save=lambda ps: write_jsonl(ps, out),
            progress=lambda line: print(line, flush=True),
        )
    finally:
        for adapter in {id(a): a for a in adapters.values()}.values():
            await adapter.aclose()
    print(json.dumps(summary.__dict__, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m evals.judge.public.run", description=__doc__)
    parser.add_argument("--pairs", default=str(HERE / "pairs.jsonl"), help="the frozen pairs")
    parser.add_argument("--out", required=True, help="this judge's run file (resumed if present)")
    parser.add_argument(
        "--judges",
        required=True,
        help="Relay model names, in order of preference: the first is the judge; later ones "
        "take over only when it can't go on",
    )
    parser.add_argument("--prompt-version", default="v1", choices=["v1", "v2"])
    parser.add_argument("--max-calls", type=int, default=600)
    parser.add_argument("--max-usd", type=float, default=None, help="stop at this list price")
    parser.add_argument("--limit", type=int, default=None, help="judge at most N pairs (a probe)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    out = Path(args.out)
    # A run file per judge: started from the frozen, unjudged pairs, then judged in place.
    pairs = read_jsonl(out if out.exists() else Path(args.pairs))
    out.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(_main(args, pairs, out))


if __name__ == "__main__":
    main()
