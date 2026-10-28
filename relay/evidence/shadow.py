"""Shadow sampling: judged pairs from live traffic, gathered after the user has their answer
(PRD §10.2; ADR 0022).

    Log → [Shadow] → Admission → Cache → Route → Reservation → Provider

``process`` does nothing. ``finalize`` runs when the request is over (for a stream, when the
stream has ended) and may start a **background task** that:

1. picks the comparisons this request can provide evidence for:
   - ``cheap_vs_strong``: the cheap model answered → ask the strong model too;
   - ``fallback:<src>-><dst>``: a healthy primary answered → ask each gated fallback target
     (evidence gathered *before* an outage needs it);
   - ``band:<lo>-<hi>``: the router sent a request strong with a score just under the
     threshold → ask the cheap model (the only evidence for lowering the threshold);
2. keeps each with probability ``sample_rate`` (lower once its card is decided);
3. checks privacy (the extra model and the judge must both be providers the tenant allows)
   and the route's daily shadow budget;
4. calls the extra model through the dispatcher at batch priority, so it counts against
   provider capacity, breakers and pauses, and is skipped (never queued) when they say no;
5. judges candidate vs reference and stores the pair (text only for `full`-logging tenants).

**Nothing here can affect the user's response.** It has been sent before ``finalize`` runs;
every failure in the task is counted and swallowed; pending tasks are capped and the
overflow is dropped (the cache write-back's pattern).
"""

import asyncio
import contextlib
import logging
import random
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from prometheus_client import Counter

from relay.admission.store import CapacityStore, StoreUnavailable
from relay.admission.tokens import output_cap, prompt_tokens, text_tokens
from relay.config.models import ModelSpec
from relay.config.pricing import NANO_PER_USD, list_cost_nano
from relay.config.schema import EvidenceSpec, RelayConfig, RouteSpec
from relay.evidence.board import CardBoard, fallback_comparison
from relay.evidence.cohorts import cohort_of
from relay.evidence.judge import Judge
from relay.evidence.pairs import prompt_hash
from relay.evidence.rubric import SYSTEM
from relay.evidence.store import EvidenceStore, ShadowPair
from relay.pipeline.context import RequestContext
from relay.pipeline.stage import Continue, Outcome, StageResult
from relay.routing.base import RouteDecision
from relay.schemas.chat import ChatRequest, ChatResponse, Usage
from relay.schemas.errors import RelayError

logger = logging.getLogger(__name__)

SHADOW = Counter(
    "relay_shadow_total",
    "Shadow comparisons by outcome (judged, abstained, or why it was skipped).",
    ["route", "kind", "outcome"],
)

MAX_PENDING = 50
JUDGE_MAX_TOKENS = 16

# A judge, by Relay model name and output cap; None when that model has no adapter (no key).
type JudgeFactory = Callable[[str, int], Judge | None]
# The dispatcher's entry point (``ReliableDispatcher.run``), as a callable so tests can stand
# in for it. For a non-streaming request it returns a ChatResponse.
type ShadowCall = Callable[[RequestContext], Awaitable[object]]


@dataclass(frozen=True)
class Job:
    comparison: str
    candidate: str  # Relay model name
    reference: str
    call: str  # which of the two is the extra (shadow) call; the other already answered


def comparison_kind(comparison: str) -> str:
    return comparison.split(":", 1)[0]


def band_comparison(lo: float, hi: float) -> str:
    return f"band:{lo:g}-{hi:g}"


def jobs_for(ctx: RequestContext, route: RouteSpec, spec: EvidenceSpec) -> list[Job]:
    """The comparisons this finished request can give evidence for (before sampling)."""
    served = ctx.served
    if served is None or served.hop != 0:
        # Only a healthy primary's answer is a fair reference or candidate: an answer from a
        # fallback hop came from an outage, not from the policy being measured.
        return []
    cheap, strong = route.aliases.get("relay/cheap"), route.aliases.get("relay/strong")
    jobs = []
    if cheap and strong and served.model == cheap and cheap != strong:
        # In a canary's arm the cheap answer was chosen by the candidate bundle: its evidence
        # is the canary's card (Phase 13), and it decides the canary's fate.
        version = ctx.applied.version if ctx.applied is not None else None
        canary = ctx.canary_arm == "canary" and version is not None
        comparison = f"canary:{version}" if canary else "cheap_vs_strong"
        jobs.append(Job(comparison, candidate=cheap, reference=strong, call=strong))
    if spec.sample_fallbacks:
        for edge in route.fallback_graph.get(served.model, []):
            if edge.quality_gate == "required":
                jobs.append(
                    Job(
                        fallback_comparison(served.model, edge.to),
                        candidate=edge.to,
                        reference=served.model,
                        call=edge.to,
                    )
                )
    if (
        spec.band_lo is not None
        and route.router is not None
        and cheap
        and served.model == strong
        and ctx.router_reason == "router"
        and ctx.router_score is not None
        and spec.band_lo <= ctx.router_score < route.router.threshold
    ):
        jobs.append(
            Job(
                band_comparison(spec.band_lo, route.router.threshold),
                candidate=cheap,
                reference=served.model,
                call=cheap,
            )
        )
    return jobs


def request_text(req: ChatRequest) -> str:
    """What the judge is shown as "the request": the conversation, one line per message."""
    if len(req.messages) == 1:
        return req.messages[0].content
    return "\n".join(f"{m.role}: {m.content}" for m in req.messages)


class ShadowStage:
    name = "shadow"

    def __init__(
        self,
        config: RelayConfig,
        models: Mapping[str, ModelSpec],
        *,
        call: ShadowCall | None = None,
        judges: JudgeFactory,
        store: EvidenceStore,
        budget: CapacityStore,
        board: CardBoard | None = None,
        rng: random.Random | None = None,
        max_pending: int = MAX_PENDING,
    ) -> None:
        self._config = config
        self._models = models
        self._call = call
        self._judges = judges
        self._store = store
        self._budget = budget
        self._board = board
        self._rng = rng or random.Random()
        self._max_pending = max_pending
        self._pending: set[asyncio.Task[None]] = set()

    def reseed(self, seed: int) -> None:
        """Make sampling reproducible (the M8 cost experiment)."""
        self._rng = random.Random(seed)

    def bind(self, call: ShadowCall) -> None:
        """Set the shadow-call entry point (the dispatcher is built after this stage)."""
        self._call = call

    async def process(self, ctx: RequestContext) -> StageResult:
        return Continue()

    async def finalize(self, ctx: RequestContext, outcome: Outcome) -> None:
        if outcome.status != "success" or ctx.tenant.evidence_sampling != "on":
            return
        if ctx.cache is not None and ctx.cache.hit:
            return  # nothing was generated for this request
        config = ctx.applied.config if ctx.applied is not None else self._config
        route = config.routes.get(ctx.route)
        if route is None or route.evidence is None:
            return
        answer = "".join(ctx.reply_parts)
        if not answer:
            return
        spec = route.evidence
        for job in jobs_for(ctx, route, spec):
            kind = comparison_kind(job.comparison)
            rate = (
                spec.canary_sample_rate
                if kind == "canary"
                else self._rate(ctx.route, job.comparison, spec)
            )
            if self._rng.random() >= rate:
                continue
            if not self._allowed(ctx, job.call, spec.judge):
                SHADOW.labels(ctx.route, kind, "skipped_privacy").inc()
                continue
            if len(self._pending) >= self._max_pending:
                SHADOW.labels(ctx.route, kind, "dropped_pending").inc()
                continue
            task = asyncio.create_task(self._run(ctx, job, answer, spec))
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)

    def _rate(self, route: str, comparison: str, spec: EvidenceSpec) -> float:
        state = self._board.card_state(route, comparison) if self._board else None
        return spec.sample_rate_decided if state in ("safe", "unsafe") else spec.sample_rate

    def _allowed(self, ctx: RequestContext, *models: str) -> bool:
        # allowed_providers is a privacy rule (§10.2 item 6): the prompt may only reach these.
        for name in models:
            spec = self._models.get(name)
            if spec is None or not ctx.tenant.allows(name, spec.provider):
                return False
        return True

    async def drain(self, timeout_s: float = 5.0) -> None:
        if self._pending:
            await asyncio.wait(list(self._pending), timeout=timeout_s)

    # ------------------------------------------------------------------ the background task
    async def _run(self, ctx: RequestContext, job: Job, answer: str, spec: EvidenceSpec) -> None:
        kind = comparison_kind(job.comparison)
        try:
            outcome = await self._shadow(ctx, job, answer, spec)
        except Exception:  # never let a background task die loudly or leak
            logger.exception("shadow comparison failed")
            outcome = "error"
        SHADOW.labels(ctx.route, kind, outcome).inc()

    async def _shadow(self, ctx: RequestContext, job: Job, answer: str, spec: EvidenceSpec) -> str:
        judge = self._judges(spec.judge, spec.judge_max_tokens)
        if judge is None or self._call is None:
            return "skipped_no_judge"
        prompt = request_text(ctx.request)
        call_model = self._models[job.call]
        judge_model = self._models[spec.judge]

        # Budget: hold the worst case of both calls, settle to what they really cost.
        cap = output_cap(ctx.request, call_model)
        estimate = list_cost_nano(
            call_model,
            Usage(prompt_tokens=prompt_tokens(ctx.request), completion_tokens=cap, total_tokens=0),
        ) + _judge_cost(judge_model, text_tokens(prompt + answer) + cap)
        day = datetime.now(UTC).strftime("%Y%m%d")
        budget_key, res_id = f"shadow:{ctx.route}:{day}", uuid.uuid4().hex
        limit = int(spec.shadow_budget_usd_per_day * NANO_PER_USD)
        try:
            held = await self._budget.reserve(
                budget_key, res_id, amount=estimate, limit=limit, ttl_s=300
            )
        except StoreUnavailable:
            return "skipped_budget_unavailable"  # fail closed: evidence is optional, money isn't
        if not held:
            return "skipped_budget"

        spent = 0
        try:
            shadow_ctx = RequestContext(
                request_id=f"{ctx.request_id}:shadow",
                tenant=ctx.tenant,
                request=ctx.request.model_copy(update={"model": job.call, "stream": False}),
                route=ctx.route,
                priority="batch",
                cohort=ctx.cohort,
                policy_version=ctx.policy_version,
                deadline=time.monotonic() + ctx.tenant.deadlines.batch_s,
                stream=False,
                decision=RouteDecision(
                    tier=call_model.tier,
                    model=job.call,
                    score=None,
                    reason="shadow",  # one candidate, never a fallback (candidates.py)
                    policy_version=ctx.policy_version,
                ),
            )
            try:
                response = await self._call(shadow_ctx)
            except RelayError:
                return "skipped_unavailable"
            if not isinstance(response, ChatResponse):
                return "skipped_unavailable"
            if shadow_ctx.cost is not None:
                spent += shadow_ctx.cost.list_nano
            shadow_text = response.choices[0].message.content or ""
            candidate, reference = (
                (shadow_text, answer) if job.call == job.candidate else (answer, shadow_text)
            )
            result = await judge.judge(request=prompt, candidate=candidate, reference=reference)
            spent += _judge_cost(judge_model, text_tokens(prompt + candidate + reference))
        finally:
            # If the store is down the reservation expires by itself (its ttl).
            with contextlib.suppress(StoreUnavailable):
                await self._budget.settle(budget_key, res_id, actual=spent)

        keep_text = ctx.tenant.logging == "full"
        row = ShadowPair(
            pair_id=f"shadow-{uuid.uuid4().hex[:16]}",
            request_id=ctx.request_id,
            route=ctx.route,
            prompt_hash=prompt_hash(prompt),
            cohort=cohort_of(prompt).key,
            comparison=job.comparison,
            candidate_model=job.candidate,
            reference_model=job.reference,
            judge_label=result.label,
            judge_model=result.judge_model,
            judge_prompt_version=result.prompt_version,
            position_swapped=result.position_swapped,
            text_retained=keep_text,
            ts=datetime.now(UTC),
            policy_version=ctx.policy_version,
            prompt_text=prompt if keep_text else None,
            candidate_text=candidate if keep_text else None,
            reference_text=reference if keep_text else None,
            cost_nano_usd=spent,
        )
        try:
            await self._store.write_pairs([row])
        except Exception:
            logger.warning("shadow pair not stored", exc_info=True)
            return "write_failed"
        return "judged" if result.label is not None else "abstained"


def _judge_cost(model: ModelSpec, text_tokens_in: int) -> int:
    """List-price estimate of one judge call: the rubric plus the three texts in, a one-word
    verdict out. An estimate (the judge's own usage isn't reported back), used for the shadow
    budget and the M8 cost share."""
    tokens_in = text_tokens(SYSTEM) + text_tokens_in + 40  # 40: delimiters and labels
    return list_cost_nano(
        model,
        Usage(prompt_tokens=tokens_in, completion_tokens=JUDGE_MAX_TOKENS, total_tokens=0),
    )
