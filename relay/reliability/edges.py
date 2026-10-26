"""Fallback edge states (PRD FR-5 edge-usability table; ADR 0010).

An edge with ``quality_gate: required`` may serve traffic only according to its *evidence*:
- ``safe``: the target is known to be good enough;
- ``unsafe``: it is known not to be;
- ``undecided``: not enough evidence either way, which is where every new edge starts.

The evidence comes from route cards, which the evidence engine produces in Phase 12. Until
then, ``SeededEdgeStates`` reads each edge's ``seed_state`` from the YAML: the stand-in for
the PRD's offline seeding run (§10.2). Both sit behind the ``EdgeStateSource`` protocol, so
Phase 12 swaps the source without touching the dispatcher.
"""

from typing import Protocol

from relay.config.schema import EdgeState, RelayConfig


class EdgeStateSource(Protocol):
    def state(self, route: str, source: str, target: str) -> EdgeState: ...


class SeededEdgeStates:
    def __init__(self, config: RelayConfig) -> None:
        self._states: dict[tuple[str, str, str], EdgeState] = {
            (route_name, source, edge.to): edge.seed_state
            for route_name, route in config.routes.items()
            for source, edges in route.fallback_graph.items()
            for edge in edges
        }

    def state(self, route: str, source: str, target: str) -> EdgeState:
        return self._states.get((route, source, target), "undecided")
