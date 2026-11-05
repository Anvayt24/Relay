"""Fallback edge states (PRD FR-5 edge-usability table; ADR 0010).

An edge with ``quality_gate: required`` may serve traffic only according to its *evidence*:
- ``safe``: the target is known to be good enough;
- ``unsafe``: it is known not to be;
- ``undecided``: not enough evidence either way, which is where every new edge starts.

Two sources sit behind the ``EdgeStateSource`` protocol, so the dispatcher never knows which
it has:
- ``SeededEdgeStates`` reads each edge's ``seed_state`` from the YAML: the stand-in for the
  PRD's offline seeding run (§10.2), used when a route has no evidence engine;
- ``CardBoard`` (``relay/evidence/board.py``) follows the route cards the evidence worker
  writes, falling back to the seed state for an edge with no card yet.
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
