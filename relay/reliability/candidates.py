"""Which models may serve a request, in what order (PRD FR-5; ADRs 0008, 0010, 0011).

``plan_candidates`` turns a route decision into an ordered list:

1. **The primary**, the model the router chose.
2. **Fallbacks, breadth-first** over the route's graph: the primary's own edges in config
   order, then *their* edges, and so on. Breadth-first because the config author's direct
   preferences for the primary ("if this fails, try X, then Y") should be honored before
   wandering further. A model reached twice is listed once.

Then the tenant's policy filters and orders the fallbacks:
- **strict:** only ``safe`` edges and ungated (``quality_gate: none``) edges.
- **allow_degraded:** every edge, stably sorted safe → undecided → unsafe. ``unsafe`` lands
  last, so it is used "only if nothing better exists" (FR-5): only once everything before it
  has failed or been skipped.
- **explicit model** (``reason == "explicit"``): no fallbacks at all, unless the tenant opted
  in with ``fallback_on_explicit_model`` (ADR 0011).

This is a *static* plan: what the config and policy allow. Whether a candidate is usable
*right now* (paused, breaker open, out of capacity, deadline) is decided per attempt by the
dispatcher. The reservation stage sizes its hold for the most expensive model in this same
list, so the two can never disagree about which models a request might reach.
"""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

from relay.config.models import ModelSpec
from relay.config.schema import EdgeState, RelayConfig
from relay.config.tenants import Tenant
from relay.reliability.edges import EdgeStateSource
from relay.routing.base import RouteDecision

_RANK: dict[EdgeState, int] = {"safe": 0, "undecided": 1, "unsafe": 2}


@dataclass(frozen=True)
class Candidate:
    model: str
    # 0 for the primary; for a fallback, its distance from the primary in the graph.
    depth: int
    # The edge's state; None for the primary and for ungated edges.
    state: EdgeState | None = None

    @property
    def is_fallback(self) -> bool:
        return self.depth > 0

    @property
    def degraded(self) -> bool:
        """Served through an edge without evidence that it is good enough (ADR 0010)."""
        return self.state in ("undecided", "unsafe")


def plan_candidates(
    *,
    config: RelayConfig,
    models: Mapping[str, ModelSpec],
    route: str,
    tenant: Tenant,
    decision: RouteDecision,
    edge_states: EdgeStateSource,
) -> list[Candidate]:
    primary = Candidate(decision.model, depth=0)
    if decision.reason == "explicit" and not tenant.fallback_on_explicit_model:
        return [primary]

    graph = config.routes[route].fallback_graph
    fallbacks: list[Candidate] = []
    seen = {decision.model}
    frontier: deque[tuple[str, int]] = deque([(decision.model, 0)])
    while frontier:
        source, depth = frontier.popleft()
        for edge in graph.get(source, []):
            if edge.to in seen:
                continue
            seen.add(edge.to)
            frontier.append((edge.to, depth + 1))
            state = (
                None if edge.quality_gate == "none" else edge_states.state(route, source, edge.to)
            )
            fallbacks.append(Candidate(edge.to, depth=depth + 1, state=state))

    # Static usability: a model this tenant may never call (allowed_providers is also a
    # privacy rule: the prompt must not reach that provider) or one with no adapter (no key)
    # is not a candidate. Dropping it here keeps it out of the reservation's worst case too.
    def reachable(c: Candidate) -> bool:
        spec = models.get(c.model)
        return spec is not None and tenant.allows(c.model, spec.provider)

    fallbacks = [c for c in fallbacks if reachable(c)]
    if tenant.fallback_policy == "strict":
        fallbacks = [c for c in fallbacks if c.state in (None, "safe")]
    else:
        # sorted() is stable: within a state, config order (and BFS depth) is kept.
        fallbacks = sorted(fallbacks, key=lambda c: _RANK[c.state] if c.state else 0)
    return [primary, *fallbacks]
