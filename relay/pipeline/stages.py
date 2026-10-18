"""The concrete stages. Phase 1 has two: route, then call the provider.

Each stage is a small class with its dependencies passed into ``__init__``. Nothing is looked
up globally, so a test builds a stage with exactly the router or providers it wants.
"""

import time
from collections.abc import Mapping

from relay.config.models import ModelSpec
from relay.pipeline.context import ProviderAttempt, RequestContext
from relay.pipeline.stage import Continue, Outcome, Reject, ShortCircuit, StageResult
from relay.providers.base import ProviderAdapter
from relay.routing.base import Router
from relay.schemas.errors import ProviderError, RelayError


class RouteStage:
    name = "route"

    def __init__(self, router: Router) -> None:
        self._router = router

    async def process(self, ctx: RequestContext) -> StageResult:
        try:
            ctx.decision = await self._router.decide(ctx.request, ctx)
        except RelayError as e:
            return Reject(e)
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
        self, models: Mapping[str, ModelSpec], providers: Mapping[str, ProviderAdapter]
    ) -> None:
        missing = {spec.provider for spec in models.values()} - set(providers)
        if missing:
            raise ValueError(f"no adapter registered for provider(s): {sorted(missing)}")
        self._models = models
        self._providers = providers

    async def process(self, ctx: RequestContext) -> StageResult:
        decision = ctx.decision
        if decision is None:
            raise RuntimeError("ProviderStage ran without a route decision; RouteStage goes first")

        spec = self._models[decision.model]
        adapter = self._providers[spec.provider]
        # The provider gets its own model ID; the client's request object is left untouched.
        upstream_request = ctx.request.model_copy(update={"model": spec.provider_model})

        started = time.monotonic()
        try:
            response = await adapter.complete(upstream_request)
        except ProviderError as e:
            ctx.attempts.append(
                ProviderAttempt(
                    spec.provider, decision.model, time.monotonic() - started, e.error_class
                )
            )
            return Reject(RelayError.from_provider(e))

        ctx.attempts.append(
            ProviderAttempt(spec.provider, decision.model, time.monotonic() - started, None)
        )
        # Clients see Relay names only. Leaking the upstream ID would let clients depend on it.
        return ShortCircuit(response.model_copy(update={"model": decision.model}))

    async def finalize(self, ctx: RequestContext, outcome: Outcome) -> None:
        return None
