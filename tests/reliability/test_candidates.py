"""Candidate planning: order, policy and edge states (FR-5 table; ADRs 0008, 0010, 0011)."""

import pytest

from relay.config.schema import RelayConfig
from relay.reliability.candidates import Candidate, plan_candidates
from relay.reliability.edges import SeededEdgeStates

from .conftest import FALLBACK_CONFIG, make_ctx  # type: ignore[import-not-found]


def plan(
    route: str, *, tenant: str = "strict", reason: str = "alias", **tenant_update: object
) -> list[Candidate]:
    config: RelayConfig = FALLBACK_CONFIG
    base = config.tenants[tenant]
    t = base.model_copy(update=tenant_update) if tenant_update else base
    ctx = make_ctx(route=route, tenant_override=t, reason=reason)
    assert ctx.decision is not None
    return plan_candidates(
        config=config,
        models=config.models,
        route=route,
        tenant=t,
        decision=ctx.decision,
        edge_states=SeededEdgeStates(config),
    )


def models(cands: list[Candidate]) -> list[str]:
    return [c.model for c in cands]


def test_no_graph_means_primary_only() -> None:
    assert models(plan("none")) == ["p/model"]


def test_strict_uses_only_safe_and_ungated_edges() -> None:
    assert models(plan("safe")) == ["p/model", "b/model"]
    assert models(plan("gated")) == ["p/model"]  # undecided: not usable under strict
    # mixed: unsafe b, undecided big, safe c -> only c.
    assert models(plan("mixed")) == ["p/model", "c/model"]


@pytest.mark.parametrize("route", ["gated", "mixed"])
def test_allow_degraded_orders_safe_then_undecided_then_unsafe(route: str) -> None:
    cands = plan(route, fallback_policy="allow_degraded")
    if route == "gated":
        assert [(c.model, c.state, c.degraded) for c in cands[1:]] == [
            ("b/model", "undecided", True)
        ]
    else:
        assert [(c.model, c.state) for c in cands[1:]] == [
            ("c/model", "safe"),
            ("big/model", "undecided"),
            ("b/model", "unsafe"),  # last: only if nothing better is left
        ]
        assert [c.degraded for c in cands] == [False, False, True, True]


def test_breadth_first_order_and_depth() -> None:
    cands = plan("deep")
    # The primary's own edges (b, c) before b's edge (big).
    assert [(c.model, c.depth) for c in cands] == [
        ("p/model", 0),
        ("b/model", 1),
        ("c/model", 1),
        ("big/model", 2),
    ]
    assert not any(c.degraded for c in cands)  # quality_gate: none is never degraded


def test_explicit_models_get_no_fallback_unless_opted_in() -> None:
    assert models(plan("safe", reason="explicit")) == ["p/model"]
    assert models(plan("safe", reason="explicit", fallback_on_explicit_model=True)) == [
        "p/model",
        "b/model",
    ]


def test_providers_the_tenant_may_not_use_are_never_candidates() -> None:
    # `private` may not send prompts to `backup`: its fallback is dropped, not skipped later.
    assert models(plan("safe", tenant="private")) == ["p/model"]
