"""Router v1: a trained logistic regression decides cheap vs strong for ``relay/auto``
(PRD FR-7; ADR 0017).

    alias lookup (StaticRouter) ─▶ the route's router alias?   no ─▶ that decision, unchanged
                                              │ yes
                                  embed the last user message (or reuse ctx.embedding)
                                              │
                          score = P(cheap answer is win-or-tie vs strong)
                                              │
                        score >= threshold ─▶ route's relay/cheap   else ─▶ relay/strong

**What the router never touches.** The static router runs first, so everything it decides
stays decided: unknown models (404), ``relay/cheap``, ``relay/strong``, and **explicit
provider-qualified models**, which bypass the router entirely (reason stays ``explicit``, so
ADR 0011's no-fallback rule still applies to them).

**Every failure routes to strong, and none of them is an error.** No artifact, a corrupt one,
an artifact trained on a different embedding model, an embedder that is off, still loading,
failing or slow: each ends in the strong model, with ``router_fallback:<cause>`` as the
reason and a counter. Strong is the quality-safe default (PRD degradation matrix): the cost of
a router failure is money, never answer quality.

**The router does not know about tenants.** It only chooses between the route's own
``relay/cheap`` and ``relay/strong`` aliases; ``RouteStage`` still enforces the tenant's
rules on whatever it returns (and the config loader already guarantees every tenant on a
route may call both).
"""

import asyncio
import logging
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
from prometheus_client import Counter, Histogram

from relay.cache.embed import Embedder, EmbedderBusy
from relay.config.models import ModelSpec
from relay.config.schema import RelayConfig
from relay.routing.artifact import RouterArtifact
from relay.routing.base import RouteDecision
from relay.routing.features import request_scalars
from relay.routing.static import StaticRouter
from relay.schemas.chat import ChatRequest

if TYPE_CHECKING:
    from relay.pipeline.context import RequestContext

logger = logging.getLogger(__name__)

DECISIONS = Counter(
    "relay_router_decisions_total",
    "Router decisions for the router alias.",
    ["route", "tier", "reason"],  # reason: router | router_fallback (cause is in the next metric)
)
# Not a dispatch fallback (relay_fallbacks_total: a call failed and the next model served).
# This is the router itself choosing strong because it could not score the prompt.
STRONG_DEFAULTS = Counter(
    "relay_router_strong_default_total",
    "Router decisions that went to the strong model because the prompt could not be scored, "
    "by cause.",
    ["cause"],
)
SCORE = Histogram(
    "relay_router_score",
    "Router score: estimated P(cheap answer is win-or-tie).",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
)
DECIDE_SECONDS = Histogram("relay_router_decide_seconds", "Time to make a router decision.")


class TrainedRouter:
    def __init__(
        self,
        config: RelayConfig,
        available: Mapping[str, ModelSpec],
        *,
        artifacts: Mapping[str, RouterArtifact | None],
        embedder: Embedder | None,
    ) -> None:
        self._config = config
        self._available = available
        # route name -> its artifact, or None when it is missing/invalid (route to strong).
        self._artifacts = artifacts
        self._embedder = embedder
        self._static = StaticRouter(config, available)

    async def decide(
        self, req: ChatRequest, ctx: "RequestContext", *, shadow: bool = False
    ) -> RouteDecision:
        decision = await self._static.decide(req, ctx)
        route = self._config.routes[ctx.route]
        spec = route.router
        if spec is None or decision.reason != "alias" or req.model != spec.alias:
            return decision

        started = time.perf_counter()
        artifact = self._artifacts.get(ctx.route)
        cheap_name, strong_name = route.aliases["relay/cheap"], route.aliases["relay/strong"]
        score, cause = await self._score(req, ctx, artifact, spec.embed_timeout_s, shadow)

        if score is None or artifact is None:
            cause = cause or "no_artifact"
            return self._fallback(ctx, decision, strong_name, artifact, cause, shadow)
        if cheap_name not in self._available:
            return self._fallback(ctx, decision, strong_name, artifact, "cheap_unavailable", shadow)

        tier_name = cheap_name if score >= spec.threshold else strong_name
        tier = self._config.models[tier_name].tier
        if not shadow:
            SCORE.observe(score)
            DECIDE_SECONDS.observe(time.perf_counter() - started)
            DECISIONS.labels(route=ctx.route, tier=tier, reason="router").inc()
        return RouteDecision(
            tier=tier,
            model=tier_name,
            score=score,
            reason="router",
            policy_version=artifact.version,
        )

    # ------------------------------------------------------------------ scoring
    async def _score(
        self,
        req: ChatRequest,
        ctx: "RequestContext",
        artifact: RouterArtifact | None,
        timeout_s: float,
        shadow: bool,
    ) -> tuple[float | None, str | None]:
        if artifact is None:
            return None, "no_artifact"
        embedder = self._embedder
        if embedder is None and ctx.embedding is None:
            return None, "embedder_off"
        if embedder is not None and embedder.name != artifact.embedding_model:
            # A vector from a different model has different axes; scoring it with these
            # weights would give confident nonsense.
            return None, "embedding_model_mismatch"

        text, scalars = request_scalars(req)
        embedding = ctx.embedding
        if embedding is None:
            assert embedder is not None
            if ctx.embed_failure is not None:
                return None, ctx.embed_failure  # the cache stage just tried, and it failed
            if not embedder.ready:
                return None, "embedder_not_ready"
            try:
                async with asyncio.timeout(timeout_s):
                    embedding = await embedder.embed(text)
            except TimeoutError:
                return None, "embed_timeout"
            except EmbedderBusy:
                return None, "embed_busy"
            except Exception:  # an embedding failure must never fail the request
                logger.exception("router embedding failed; routing to strong")
                return None, "embed_error"
            if not shadow:  # a shadow decision changes nothing on the context
                ctx.embedding = embedding  # reusable by whoever runs next
        if len(embedding) != artifact.dimensions:
            return None, "bad_dimension"
        return artifact.score(np.asarray(embedding, dtype=float), np.asarray(scalars)), None

    def _fallback(
        self,
        ctx: "RequestContext",
        original: RouteDecision,
        strong_name: str,
        artifact: RouterArtifact | None,
        cause: str,
        shadow: bool,
    ) -> RouteDecision:
        # If even the strong model isn't available, keep the static answer: the pipeline's own
        # checks (404 / tenant rules) are the right place to complain about that.
        name = strong_name if strong_name in self._available else original.model
        tier = self._config.models[name].tier
        if not shadow:
            STRONG_DEFAULTS.labels(cause=cause).inc()
            DECISIONS.labels(route=ctx.route, tier=tier, reason="router_fallback").inc()
        return RouteDecision(
            tier=tier,
            model=name,
            score=None,
            reason=f"router_fallback:{cause}",
            policy_version=artifact.version if artifact else "router-unavailable",
        )
