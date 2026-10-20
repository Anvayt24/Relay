"""The whole config file, and the cross-checks that make an invalid config fail at startup.

Field-level rules (types, ranges, unknown keys) come from the Pydantic models. Rules that span
sections live in ``RelayConfig._cross_checks``, one check per PRD FR-3 requirement:

- every model names a known provider (and has prices: the fields are required);
- every alias resolves to a known model;
- every fallback graph refers to known models and is **acyclic**. A cycle would let a
  failing request bounce between two models forever; Phase 7 relies on this;
- every tenant's route exists, and its providers and models are known;
- every alias in a tenant's route points at a model that tenant may call. Otherwise that
  alias would 404 for every request: a config bug, caught here instead of by users.

An invalid config raises ``ValueError`` (Pydantic's ``ValidationError`` is one), and the app
refuses to start (FR-3).
"""

from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

from relay.config.models import STRICT, ModelSpec, ProviderSpec
from relay.config.tenants import Tenant


class FallbackEdge(BaseModel):
    model_config = STRICT

    to: str
    # required: usable only once evidence says the target is good enough (Phase 7 + 12).
    quality_gate: Literal["required", "none"] = "required"


class RouteSpec(BaseModel):
    model_config = STRICT

    # Alias (what clients send, e.g. "relay/cheap") -> Relay model name.
    aliases: dict[str, str]
    # Model -> where to go when it fails, in order of preference. Used from Phase 7.
    fallback_graph: dict[str, list[FallbackEdge]] = Field(default_factory=dict)

    @property
    def strong_model(self) -> str | None:
        """The route's strong model: the "always strong" baseline for cost saved (FR-9)."""
        return self.aliases.get("relay/strong")


class RelayConfig(BaseModel):
    model_config = STRICT

    price_table_version: str
    providers: dict[str, ProviderSpec]
    models: dict[str, ModelSpec]
    routes: dict[str, RouteSpec]
    tenants: dict[str, Tenant]

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
