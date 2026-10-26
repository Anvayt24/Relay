"""The whole config file, and the cross-checks that make an invalid config fail at startup.

Field-level rules (types, ranges, unknown keys) come from the Pydantic models. Rules that span
sections live in ``RelayConfig._cross_checks``, one check per PRD FR-3 requirement:

- every model names a known provider (and has prices: the fields are required);
- every alias resolves to a known model;
- every fallback graph refers to known models and is **acyclic**. A cycle would let a
  failing request bounce between two models forever; the Phase 7 dispatcher relies on this;
- every tenant's route exists, and its providers and models are known;
- every alias in a tenant's route points at a model that tenant may call. Otherwise that
  alias would 404 for every request: a config bug, caught here instead of by users.

An invalid config raises ``ValueError`` (Pydantic's ``ValidationError`` is one), and the app
refuses to start (FR-3).
"""

import logging
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

from relay.config.models import STRICT, ModelSpec, ProviderSpec
from relay.config.tenants import Tenant

logger = logging.getLogger(__name__)

EdgeState = Literal["safe", "undecided", "unsafe"]


class FallbackEdge(BaseModel):
    model_config = STRICT

    to: str
    # required: usable only once evidence says the target is good enough (FR-5).
    # none: always usable, never marked degraded.
    quality_gate: Literal["required", "none"] = "required"
    # The edge's state until a route card exists (Phase 12): the stand-in for PRD §10.2's
    # offline seeding run. undecided by default, which is honest with no evidence (ADR 0010).
    seed_state: EdgeState = "undecided"


class RouteCache(BaseModel):
    """Semantic-cache parameters for a route (PRD FR-6). The threshold is *measured*, not
    guessed: it is the smallest cosine similarity that reached >= 99% precision on the tune
    split of the cache eval (`make eval-cache`, evals/cache/metrics.json)."""

    model_config = STRICT

    semantic_threshold: float = Field(default=0.95, ge=0, le=1)
    # Force a miss when the numbers/names in two prompts differ ("15% of 200" vs "300").
    lexical_guard: bool = True


class RouteSpec(BaseModel):
    model_config = STRICT

    # Alias (what clients send, e.g. "relay/cheap") -> Relay model name.
    aliases: dict[str, str]
    # Model -> where to go when it fails, in order of preference (FR-5).
    fallback_graph: dict[str, list[FallbackEdge]] = Field(default_factory=dict)
    # How many fallback models one request may *attempt* (skipped ones don't count).
    max_fallback_hops: int = Field(default=2, ge=0)
    cache: RouteCache = Field(default_factory=RouteCache)

    @property
    def strong_model(self) -> str | None:
        """The route's strong model: the "always strong" baseline for cost saved (FR-9)."""
        return self.aliases.get("relay/strong")


class ReliabilityConfig(BaseModel):
    """Defaults for the retry owner and the breakers (PRD FR-5; ADRs 0008, 0009)."""

    model_config = STRICT

    # Full-jitter backoff before the one retry: U(0, min(cap, base * 2^n)).
    backoff_base_s: float = Field(default=0.2, gt=0)
    backoff_cap_s: float = Field(default=2.0, gt=0)
    # Retry budget, per provider and per replica: retries <= ratio x attempts + min, over a
    # sliding window (FR-4: "retries limited to <= 10% of requests per provider per window").
    retry_budget_ratio: float = Field(default=0.1, ge=0, le=1)
    retry_budget_min: int = Field(default=3, ge=0)
    retry_budget_window_s: float = Field(default=10.0, gt=0)
    # Circuit breaker per (provider, model), per replica.
    breaker_window_s: int = Field(default=30, gt=0)
    breaker_failure_ratio: float = Field(default=0.5, gt=0, le=1)
    breaker_min_calls: int = Field(default=10, gt=0)
    breaker_open_s: float = Field(default=30.0, gt=0)
    breaker_half_open_probes: int = Field(default=3, gt=0)
    # Don't start an attempt (or a backoff) that has less than this left before the deadline.
    min_attempt_s: float = Field(default=0.5, ge=0)
    # After a model_not_found, skip that model on this replica for this long (and alert).
    model_unavailable_ttl_s: float = Field(default=300.0, gt=0)


class RelayConfig(BaseModel):
    model_config = STRICT

    price_table_version: str
    providers: dict[str, ProviderSpec]
    models: dict[str, ModelSpec]
    routes: dict[str, RouteSpec]
    tenants: dict[str, Tenant]
    reliability: ReliabilityConfig = Field(default_factory=ReliabilityConfig)

    @model_validator(mode="before")
    @classmethod
    def _tenant_ids_from_keys(cls, data: Any) -> Any:
        # In YAML the id is the mapping key; on the object it is a field. Copy it across so a
        # Tenant always knows its own id.
        if isinstance(data, dict) and isinstance(data.get("tenants"), dict):
            data = {
                **data,
                "tenants": {
                    tid: {**t, "id": tid} if isinstance(t, dict) else t
                    for tid, t in data["tenants"].items()
                },
            }
        return data

    @model_validator(mode="after")
    def _cross_checks(self) -> Self:
        for name, spec in self.models.items():
            if spec.provider not in self.providers:
                raise ValueError(f"model {name!r}: unknown provider {spec.provider!r}")

        for route_name, route in self.routes.items():
            for alias, target in route.aliases.items():
                if target not in self.models:
                    raise ValueError(
                        f"route {route_name!r}: alias {alias!r} -> unknown model {target!r}"
                    )
            for source, edges in route.fallback_graph.items():
                for model in (source, *(e.to for e in edges)):
                    if model not in self.models:
                        raise ValueError(
                            f"route {route_name!r}: fallback graph names unknown model {model!r}"
                        )
            _check_acyclic(route_name, route.fallback_graph)

        for tid, tenant in self.tenants.items():
            if tenant.route not in self.routes:
                raise ValueError(f"tenant {tid!r}: unknown route {tenant.route!r}")
            for provider in tenant.allowed_providers:
                if provider not in self.providers:
                    raise ValueError(f"tenant {tid!r}: unknown provider {provider!r}")
            if tenant.allowed_models != "*":
                for allowed in tenant.allowed_models:
                    if allowed not in self.models:
                        raise ValueError(f"tenant {tid!r}: unknown model {allowed!r}")
            for alias, target in self.routes[tenant.route].aliases.items():
                if not tenant.allows(target, self.models[target].provider):
                    raise ValueError(
                        f"tenant {tid!r}: route alias {alias!r} -> {target!r}, "
                        "which this tenant may not call"
                    )
            # Not an error: a shared route may name fallbacks some tenants can't use, and
            # the dispatcher skips them (allowed_providers is also a privacy rule). But a
            # fallback that can never be used is worth knowing about.
            for edges in self.routes[tenant.route].fallback_graph.values():
                for edge in edges:
                    if not tenant.allows(edge.to, self.models[edge.to].provider):
                        logger.warning(
                            "tenant %r can never use fallback %r on route %r",
                            tid,
                            edge.to,
                            tenant.route,
                        )
        return self


def _check_acyclic(route: str, graph: dict[str, list[FallbackEdge]]) -> None:
    """Depth-first search with three colors. Meeting a node that is still "in progress" (on the
    current path) means the path loops back on itself: a cycle."""
    in_progress, done = set(), set()

    def visit(node: str, path: list[str]) -> None:
        if node in done:
            return
        if node in in_progress:
            cycle = " -> ".join([*path[path.index(node) :], node])
            raise ValueError(f"route {route!r}: fallback graph has a cycle: {cycle}")
        in_progress.add(node)
        for edge in graph.get(node, []):
            visit(edge.to, [*path, node])
        in_progress.discard(node)
        done.add(node)

    for node in graph:
        visit(node, [])
