"""The concrete stages, in pipeline order: log → route → provider (admission stages live in
``relay/admission/stages.py``).

Each stage is a small class with its dependencies passed into ``__init__``. Nothing is looked
up globally, so a test builds a stage with exactly the router, providers or writer it wants.
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from prometheus_client import Counter

from relay.pipeline.context import RequestContext
from relay.pipeline.stage import Continue, Outcome, Reject, ShortCircuit, StageResult
from relay.routing.base import Router
from relay.routing.static import model_not_found
from relay.schemas.errors import ErrorClass, RelayError
from relay.telemetry import tracing
from relay.telemetry.log_writer import AttemptRow, LogRow, LogWriter
from relay.telemetry.metrics import record_request

if TYPE_CHECKING:
    # The dispatcher imports this package's context and streaming modules; importing it at
    # runtime here would be circular. ProviderStage only needs it as a type.
    from relay.reliability.dispatch import ReliableDispatcher

logger = logging.getLogger(__name__)

SHADOW_DECISIONS = Counter(
    "relay_policy_shadow_decisions_total",
    "Decisions made by a shadow policy's router (logged, never applied).",
    ["result"],  # ok | error
)


class LogStage:
    """Writes one request-log row per request (FR-8).

    It is the *first* stage, so its finalize runs *last*. By then every other stage has
    finalized, and the context holds the final decision, attempts, usage and outcome. Its
    process() does nothing.
    """

    name = "log"

    def __init__(self, writer: LogWriter) -> None:
        self._writer = writer

    async def process(self, ctx: RequestContext) -> StageResult:
        return Continue()

    async def finalize(self, ctx: RequestContext, outcome: Outcome) -> None:
        # enqueue() is synchronous and O(1); the database write happens later, elsewhere.
        self._writer.enqueue(row_from_context(ctx, outcome))
        # Phase 14: the request-level metrics and the trace's outcome attributes, from the
        # same finished context as the log row (ADR 0025).
        record_request(ctx, outcome, time.monotonic() - ctx.started_at)
        tracing.annotate(ctx, outcome)


def row_from_context(ctx: RequestContext, outcome: Outcome) -> LogRow:
    # Provider and model are the ones that *served*, not the router's choice: after a
    # fallback they differ, and the log must say what actually answered. A failed request
    # logs its last attempt (and the router's choice if nothing was attempted).
    served_by = ctx.served_by
    attempt = ctx.served or next(
        (a for a in reversed(ctx.attempts) if a.skipped_reason is None), None
    )
    decision = ctx.decision
    provider: str | None
    model: str | None
    if served_by is not None:
        provider, model = served_by
    else:
        provider = attempt.provider if attempt else None
        model = attempt.model if attempt else (decision.model if decision else None)
    # ctx.pricing is set per attempt, so it describes the last model called.
    tier: str | None
    if attempt is not None and ctx.pricing is not None:
        tier = ctx.pricing.model.tier
    else:
        tier = decision.tier if decision else None
    usage = ctx.usage
    return LogRow(
        request_id=ctx.request_id,
        ts=ctx.received_at,
        tenant=ctx.tenant.id,
        route=ctx.route,
        priority=ctx.priority,
        cohort=ctx.cohort,
        stream=ctx.stream,
        provider=provider,
        model=model,
        tier=tier,
        status=outcome.status,
        error_class=outcome.error_class.value if outcome.error_class else None,
        tokens_in=usage.prompt_tokens if usage else None,
        tokens_out=usage.completion_tokens if usage else None,
        latency_ms=round((time.monotonic() - ctx.started_at) * 1000),
        ttft_ms=round(ctx.ttft_s * 1000) if ctx.ttft_s is not None else None,
        queue_wait_ms=round(ctx.queue_wait_s * 1000),
        billed_cost_nano_usd=ctx.cost.billed_nano if ctx.cost else None,
        list_cost_nano_usd=ctx.cost.list_nano if ctx.cost else None,
        counterfactual_strong_nano_usd=ctx.cost.counterfactual_strong_nano if ctx.cost else None,
        price_table_version=ctx.pricing.price_table_version if ctx.pricing else None,
        fallback_hops=ctx.fallback_hops,
        degraded=ctx.degraded,
        policy_version=ctx.policy_version,
        cache_status=ctx.cache.status if ctx.cache else None,
        router_reason=ctx.router_reason,
        router_score=ctx.router_score,
        canary_arm=ctx.canary_arm,
        shadow_policy_version=ctx.shadow_policy_version,
        shadow_tier=ctx.shadow_tier,
        shadow_score=ctx.shadow_score,
        attempts=tuple(
            AttemptRow(
                request_id=ctx.request_id,
                hop=a.hop,
                retry=a.retry,
                provider=a.provider,
                model=a.model,
                error_class=a.error_class.value if a.error_class else None,
                skipped_reason=a.skipped_reason,
                latency_ms=round(a.latency_s * 1000),
                ts=ctx.received_at,
            )
            for a in ctx.attempts
        ),
    )


class RouteStage:
    """Asks the router for a model, then enforces the tenant's rules on it (FR-3).

    The router answers "which model?"; this stage answers "may *this tenant* use it?". Keeping
    the two apart means a trained router (Phase 11) never has to know about tenants, and a
    tenant rule can't be bypassed by a router change.
    """

    name = "route"

    def __init__(self) -> None:
        # Shadow decisions in flight, by request context (see _shadow_decision).
        self._shadow: dict[int, asyncio.Task[None]] = {}

    async def process(self, ctx: RequestContext) -> StageResult:
        # The router of the policy serving this request (Phase 13: the YAML one, or a bundle's).
        try:
            decision = await ctx.applied.router.decide(ctx.request, ctx)
        except RelayError as e:
            return Reject(e)
        provider = ctx.applied.config.models[decision.model].provider
        if not ctx.tenant.allows(decision.model, provider):
            # The same 404 as for a model that doesn't exist: nothing leaks about what other
            # tenants can reach.
            return Reject(model_not_found(ctx.request.model))
        ctx.decision = decision
        # The policy of record. With a policy bundle applied it is the bundle's version, set
        # when the request arrived (ADR 0023). Without one, the policy that decided: the
        # request's version was set before routing (to the static policy), so without this a
        # router's version would never reach the log row or a cache entry.
        if ctx.applied.version is None:
            ctx.policy_version = decision.policy_version
        ctx.router_reason = decision.reason
        ctx.router_score = decision.score
        shadow = ctx.policy.shadow() if ctx.policy is not None else None
        if shadow is not None:
            # Off the request's path: it runs while the provider answers and is collected in
            # finalize, before the log row is written. Awaiting it here (as before review) put
            # a whole embedding on the request whenever the live policy hadn't computed one.
            self._shadow[id(ctx)] = asyncio.create_task(
                self._shadow_decision(ctx, shadow.router, shadow.version)
            )
        return Continue()

    async def _shadow_decision(
        self, ctx: RequestContext, router: Router, version: str | None
    ) -> None:
        """Shadow mode (FR-11): the candidate's router decides too; the decision is logged and
        never applied, and it records no router metrics. A failure here is swallowed: a shadow
        bundle must never change a response."""
        try:
            decision = await router.decide(ctx.request, ctx, shadow=True)
        except Exception:
            SHADOW_DECISIONS.labels(result="error").inc()
            logger.warning("shadow policy %s could not decide", version, exc_info=True)
            return
        SHADOW_DECISIONS.labels(result="ok").inc()
        ctx.shadow_policy_version = version
        ctx.shadow_tier = decision.tier
        ctx.shadow_score = decision.score

    async def finalize(self, ctx: RequestContext, outcome: Outcome) -> None:
        task = self._shadow.pop(id(ctx), None)
        if task is not None:
            # Bounded by the shadow router's own embedding timeout; usually long finished.
            await task


class ProviderStage:
    """Gets the response from a provider, through the single retry owner.

    All the reliability decisions (retries, fallbacks, breakers, deadlines) live in
    ``relay/reliability/``, not here: one owner for retries is the whole point (FR-5). This
    stage only adapts the dispatcher to the Stage protocol, and reports a stream's outcome to
    the breaker when the stream is over.
    """

    name = "provider"

    def __init__(self, dispatcher: "ReliableDispatcher") -> None:
        self._dispatcher = dispatcher

    async def process(self, ctx: RequestContext) -> StageResult:
        try:
            return ShortCircuit(await self._dispatcher.run(ctx))
        except RelayError as e:
            return Reject(e)

    async def finalize(self, ctx: RequestContext, outcome: Outcome) -> None:
        await self._dispatcher.release_provider_tokens(ctx)
        permit = ctx.breaker_permit
        if permit is None:
            return
        ctx.breaker_permit = None
        # Only now is a stream's verdict known. A cancelled stream says nothing about the
        # provider's health, so it is neither a success nor a failure.
        if outcome.status == "success":
            self._dispatcher.breakers.record(permit, None)
        elif outcome.status == "failed_midstream":
            self._dispatcher.breakers.record(permit, ErrorClass.STREAM_BROKEN)
        else:
            self._dispatcher.breakers.release(permit)
