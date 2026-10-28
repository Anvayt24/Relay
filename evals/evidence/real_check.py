"""Real-world sanity check of the online evidence path (PRD §10.9 item 4). NON-REPRODUCIBLE.

    python -m evals.evidence.real_check --cheap <gemini id> --strong <gemini id> --n 30

What it exercises, end to end, with a real provider (Gemini) and nothing simulated:

    gateway (relay/cheap answers) → shadow stage (the strong model answers the same request)
    → bulk judge (the strong model) → eval_pairs → evidence worker → route card

It runs the in-process gateway with in-memory stores, so it needs no database; the Postgres
path is covered by the integration tests. Every real call is counted and the run refuses to
go past ``--max-calls``: past the budget a call fails like a provider quota, and the run
stops cleanly. Requests are paced (``--pace-s``) to stay under the free tier's per-minute
limits. There is **no extra retry layer**: only the gateway's own single retry, because on
2026-10-28 an overloaded Gemini plus a batch-retry wrapper spent ~8 calls per failed request.
After ``--max-failures`` failed requests in a row (the provider is overloaded) the run stops.
Whatever was gathered is always written, even when the run stops early.

What it is **not**: a quality measurement. The judge is the same family as both candidates
and has no human labels, so the card is ``uncalibrated`` by design and its estimate is
judge-only. Prompts are public (Dolly, CC BY-SA 3.0): the Gemini free tier may use prompts.

Outputs (in ``--out``): ``pairs.jsonl`` (the judged pairs, with text), ``cards.json`` and
``real_check.md``.
"""

import argparse
import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from relay.api.app import create_app
from relay.config.loader import load_config
from relay.config.schema import RelayConfig
from relay.evidence import worker
from relay.evidence.cards import ALL, render_markdown
from relay.evidence.store import MemoryEvidenceStore
from relay.providers.base import ProviderAdapter
from relay.providers.registry import ProviderKeys, build_adapters
from relay.schemas.chat import ChatRequest, ChatResponse
from relay.schemas.errors import ErrorClass, ProviderError
from relay.settings import Settings
from relay.telemetry.log_writer import MemorySink

HERE = Path(__file__).parent
PROMPTS = Path("evals/router/real/prompts_raw.jsonl")  # public Dolly prompts (Phase 11)
CALLS_PER_REQUEST = 3  # the answer, the shadow answer, the judge


class Counting:
    """Counts real calls and refuses to exceed the budget (never a surprise bill)."""

    def __init__(self, inner: ProviderAdapter, max_calls: int) -> None:
        self._inner = inner
        self.name = inner.name
        self.max_calls = max_calls
        self.calls = 0

    @property
    def exhausted(self) -> bool:
        return self.calls >= self.max_calls

    async def complete(self, req: ChatRequest) -> ChatResponse:
        if self.exhausted:
            # A provider error, not an exception: the gateway turns it into an ordinary
            # failed request, and the run loop stops.
            raise ProviderError(ErrorClass.QUOTA_EXHAUSTED, f"call budget of {self.max_calls}")
        self.calls += 1
        return await self._inner.complete(req)

    def stream(self, req: ChatRequest) -> Any:
        raise RuntimeError("the real check sends non-streaming requests only")

    async def aclose(self) -> None:
        await self._inner.aclose()


def config_for(args: argparse.Namespace) -> RelayConfig:
    base = load_config()
    gemini = base.providers["gemini"]
    return RelayConfig.model_validate(
        {
            "price_table_version": base.price_table_version,
            "providers": {"gemini": gemini.model_dump()},
            "models": {
                "gemini/cheap": {
                    "provider": "gemini",
                    "provider_model": args.cheap,
                    "tier": "cheap",
                    "price_in": 0,
                    "price_out": 0,
                    "billing": "free",
                },
                "gemini/strong": {
                    "provider": "gemini",
                    "provider_model": args.strong,
                    "tier": "strong",
                    "price_in": 0,
                    "price_out": 0,
                    "billing": "free",
                },
            },
            "routes": {
                "check": {
                    "aliases": {"relay/cheap": "gemini/cheap", "relay/strong": "gemini/strong"},
                    "evidence": {
                        "judge": "gemini/strong",
                        "judge_max_tokens": args.judge_max_tokens,
                        "sample_rate": 1.0,
                        "sample_rate_decided": 1.0,
                        "look_every": 10,
                        "max_looks": 3,
                        "min_cohort_n": 5,
                    },
                }
            },
            "tenants": {
                "check": {
                    "route": "check",
                    "limits": {"rpm": 1000, "tpm": 10**7, "concurrency": 4},
                    "budget_usd_daily": 1,
                    "budget_mode": "soft",
                    "allowed_providers": ["gemini"],
                    "evidence_sampling": "on",
                    "logging": "full",
                }
            },
        }
    )


async def run(args: argparse.Namespace, prompts: list[str], out: Path) -> int:
    config = config_for(args)
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, env="dev", auth_disabled=True, auth_disabled_tenant="check"
    )
    real = build_adapters(ProviderKeys(), settings, config.providers)
    if "gemini" not in real:
        print("GEMINI_API_KEY is not set")
        return 2
    counted = Counting(real["gemini"], args.max_calls)
    store, sink = MemoryEvidenceStore(), MemorySink()
    app = create_app(
        settings, providers={"gemini": counted}, config=config, log_sink=sink, evidence_store=store
    )
    failed = in_a_row = sent = 0
    stopped = ""
    started = time.monotonic()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://check", timeout=300
        ) as c:
            for i, prompt in enumerate(prompts):
                if counted.calls + CALLS_PER_REQUEST > args.max_calls:
                    stopped = f"call budget ({args.max_calls}) reached"
                    break
                body = {"model": "relay/cheap", "max_tokens": args.max_tokens,
                        "messages": [{"role": "user", "content": prompt}]}  # fmt: skip
                r = await c.post("/v1/chat/completions", json=body)
                sent += 1
                ok = r.status_code == 200
                failed += not ok
                in_a_row = 0 if ok else in_a_row + 1
                print(f"[{i + 1}/{len(prompts)}] {r.status_code} calls={counted.calls}")
                await app.state.shadow_stage.drain(timeout_s=600)  # one at a time: paced
                if in_a_row >= args.max_failures:
                    stopped = f"{in_a_row} failed requests in a row (provider overloaded?)"
                    break
                await asyncio.sleep(args.pace_s)
    cards = await worker.run_once(store, config)
    _write(out / "pairs.jsonl", "".join(
        json.dumps({**asdict(p), "ts": p.ts.isoformat()}, ensure_ascii=False) + "\n"
        for p in store.pairs
    ))  # fmt: skip
    _write(out / "cards.json", json.dumps([c.as_dict() for c in cards], indent=2) + "\n")
    labels = [p.judge_label for p in store.pairs]
    card = next((c for c in cards if c.cohort == ALL), None)
    report = [
        "# Evidence engine: real sanity check (NON-REPRODUCIBLE)",
        "",
        "`python -m evals.evidence.real_check` on the real Gemini API. It shows the online path "
        "works end to end (gateway -> shadow -> judge -> pairs -> worker -> card); it is **not** a "
        "quality measurement: the judge is the same family as both models and has no human "
        "labels, so the card is `uncalibrated` by design and its estimate is judge-only.",
        "",
        f"- models: cheap `{args.cheap}`, strong (and judge) `{args.strong}`",
        f"- requests: {sent} of {len(prompts)} public Dolly prompts sent; failed: {failed}"
        + (f"; **stopped early: {stopped}**" if stopped else ""),
        f"- real calls: {counted.calls} (budget {args.max_calls}); "
        f"wall time {time.monotonic() - started:.0f} s",
        f"- shadow pairs stored: {len(store.pairs)}; verdicts: "
        f"win {labels.count('win')}, tie {labels.count('tie')}, loss {labels.count('loss')}, "
        f"abstained {labels.count(None)}",
        f"- card state: **{card.state if card else 'none'}**"
        + (f" ({card.reason})" if card and card.reason else ""),
        "",
        render_markdown(cards),
        "",
    ]
    _write(out / "real_check.md", "\n".join(report))
    print("\n".join(report))
    return 0


def _write(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m evals.evidence.real_check", description=__doc__)
    p.add_argument("--cheap", required=True, help="Gemini model id for relay/cheap")
    p.add_argument("--strong", required=True, help="Gemini model id for relay/strong and judge")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--max-calls", type=int, default=100)
    p.add_argument("--max-tokens", type=int, default=1000)
    p.add_argument("--judge-max-tokens", type=int, default=800)
    p.add_argument("--pace-s", type=float, default=12.0)
    p.add_argument("--max-failures", type=int, default=3, help="stop after this many in a row")
    p.add_argument("--out", default=str(HERE / "real"))
    args = p.parse_args()
    if args.n * CALLS_PER_REQUEST > args.max_calls:
        raise SystemExit(f"{args.n} requests need {args.n * CALLS_PER_REQUEST} calls > --max-calls")
    prompts = [json.loads(line)["prompt"] for line in PROMPTS.read_text("utf-8").splitlines()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raise SystemExit(asyncio.run(run(args, prompts[: args.n], out)))


if __name__ == "__main__":
    main()
