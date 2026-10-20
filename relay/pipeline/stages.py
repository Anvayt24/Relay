"""The concrete stages, in pipeline order: log → route → provider.

Each stage is a small class with its dependencies passed into ``__init__``. Nothing is looked
up globally, so a test builds a stage with exactly the router, providers or writer it wants.
"""

import time
from collections.abc import Mapping

from relay.admission.pause import PauseState
from relay.config.models import ModelSpec
from relay.config.pricing import Pricing
from relay.config.schema import RelayConfig
from relay.pipeline.context import ProviderAttempt, RequestContext
from relay.pipeline.stage import Continue, Outcome, Reject, ShortCircuit, StageResult
from relay.pipeline.streaming import prime
from relay.providers.base import ProviderAdapter
from relay.routing.base import Router
from relay.routing.static import model_not_found
from relay.schemas.errors import ErrorClass, ProviderError, RelayError
from relay.telemetry.log_writer import LogRow, LogWriter


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


def row_from_context(ctx: RequestContext, outcome: Outcome) -> LogRow:
    attempt = ctx.attempts[-1] if ctx.attempts else None
    decision = ctx.decision
    usage = ctx.usage
    return LogRow(
        request_id=ctx.request_id,
        ts=ctx.received_at,
        tenant=ctx.tenant.id,
        route=ctx.route,
        priority=ctx.priority,
        cohort=ctx.cohort,
        stream=ctx.stream,
        provider=attempt.provider if attempt else None,
        model=decision.model if decision else None,
        tier=decision.tier if decision else None,
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
    )


class RouteStage:
    """Asks the router for a model, then enforces the tenant's rules on it (FR-3).

    The router answers "which model?"; this stage answers "may *this tenant* use it?". Keeping
    the two apart means a trained router (Phase 11) never has to know about tenants, and a
    tenant rule can't be bypassed by a router change.
    """

    name = "route"

    def __init__(self, router: Router, config: RelayConfig) -> None:
        self._router = router
        self._config = config

    async def process(self, ctx: RequestContext) -> StageResult:
        try:
            decision = await self._router.decide(ctx.request, ctx)
        except RelayError as e:
            return Reject(e)
        provider = self._config.models[decision.model].provider
        if not ctx.tenant.allows(decision.model, provider):
            # The same 404 as for a model that doesn't exist: nothing leaks about what other
            # tenants can reach.
            return Reject(model_not_found(ctx.request.model))
        ctx.decision = decision
        return Continue()

    async def finalize(self, ctx: RequestContext, outcome: Outcome) -> None:
        return None


class ProviderStage:
    """Calls the provider the route decision points at.

    There are no retries and no fallback yet; that is Phase 7, and it lives in
    ``relay/reliability/`` rather than in this stage. A single owner for retries is the
    whole point (PRD FR-5).
    """

    name = "provider"

    def __init__(
        self,
        config: RelayConfig,
        models: Mapping[str, ModelSpec],
        providers: Mapping[str, ProviderAdapter],
        *,
        first_token_timeout_s: float,
        pause: PauseState | None = None,
    ) -> None:
        missing = {spec.provider for spec in models.values()} - set(providers)
        if missing:
            raise ValueError(f"no adapter registered for provider(s): {sorted(missing)}")
        self._config = config
        self._models = models
        self._providers = providers
        self._first_token_timeout_s = first_token_timeout_s
        self._pause = pause

    async def process(self, ctx: RequestContext) -> StageResult:
        decision = ctx.decision
        if decision is None:
            raise RuntimeError("ProviderStage ran without a route decision; RouteStage goes first")

        spec = self._models[decision.model]
        adapter = self._providers[spec.provider]
        if self._pause is not None:
            # Another replica (or this one) was just told "429" for this limit: don't call,
            # answer for the provider (FR-4). Phase 7 turns this into a fallback instead.
            wait = await self._pause.paused_for(spec.provider, spec.provider_model)
            if wait > 0:
                return Reject(
                    RelayError(
                        429,
                        ErrorClass.PROVIDER_PAUSED,
                        f"provider {spec.provider!r} is rate limiting this model; retry later",
                        retry_after_s=wait,
                    )
                )
        strong = self._config.routes[ctx.route].strong_model
        ctx.pricing = Pricing(
            model=spec,
            strong=self._config.models[strong] if strong else None,
            price_table_version=self._config.price_table_version,
        )
        # The provider gets its own model ID and always an output cap: Anthropic requires one,
        # and Phase 6 reserves budget against it, so the cap reserved must be the cap sent.
        # The client's request object is left untouched.
        upstream_request = ctx.request.model_copy(
            update={
                "model": spec.provider_model,
                "max_tokens": ctx.request.max_tokens or spec.default_max_tokens,
            }
        )

        started = time.monotonic()
        result: StageResult
        try:
            if ctx.stream:
                # Only up to the first content chunk. The rest is pulled by the response body
                # after the headers are committed (see relay/pipeline/streaming.py).
                stream = await prime(
                    adapter.stream(upstream_request),
                    timeout_s=self._first_token_timeout_s,
                    relay_model=decision.model,
                )
                result = ShortCircuit(stream)
            else:
                response = await adapter.complete(upstream_request)
                ctx.record_usage(response.usage)
                # Clients see Relay names only. Leaking the upstream ID would let clients
                # depend on it.
                result = ShortCircuit(response.model_copy(update={"model": decision.model}))
        except ProviderError as e:
            if self._pause is not None:
                await self._pause.record(spec.provider, spec.provider_model, e)
            ctx.attempts.append(
                ProviderAttempt(
                    spec.provider, decision.model, time.monotonic() - started, e.error_class
                )
            )
            return Reject(RelayError.from_provider(e))

        ctx.attempts.append(
            ProviderAttempt(spec.provider, decision.model, time.monotonic() - started, None)
        )
        return result

    async def finalize(self, ctx: RequestContext, outcome: Outcome) -> None:
        return None
