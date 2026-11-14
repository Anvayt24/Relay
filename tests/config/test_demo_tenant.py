"""The shipped `demo` tenant is the one a stranger can reach: it must be
capped, free, and leave nothing behind that a later decision could learn from."""

from relay.config.loader import load_config

CONFIG = load_config()
DEMO = CONFIG.tenants["demo"]
ROUTE = CONFIG.routes[DEMO.route]


def test_the_demo_tenant_is_capped_per_request_and_per_address() -> None:
    assert DEMO.request_limits.max_messages is not None and DEMO.request_limits.max_messages <= 20
    assert DEMO.request_limits.max_tokens is not None and DEMO.request_limits.max_tokens <= 512
    assert DEMO.limits.per_ip_rpm is not None
    assert DEMO.limits.per_ip_rpm < DEMO.limits.rpm  # one address cannot use the whole allowance
    assert DEMO.budget_mode == "hard" and DEMO.budget_usd_daily <= 1


def test_the_demo_stores_no_prompt_text_and_feeds_no_decision() -> None:
    assert DEMO.logging == "metadata_only"
    assert DEMO.evidence_sampling == "off"
    assert ROUTE.evidence is None  # nothing on its route judges anything either


def test_every_model_the_demo_can_reach_is_free_and_from_an_allowed_provider() -> None:
    reachable = set(ROUTE.aliases.values())
    for edges in ROUTE.fallback_graph.values():
        reachable |= {e.to for e in edges}
    for name in reachable:
        assert CONFIG.models[name].billing == "free", f"{name} would cost money"
        assert CONFIG.models[name].provider in DEMO.allowed_providers


def test_the_demos_fallback_edges_are_seeded_safe() -> None:
    # A strict tenant is refused an `undecided` edge, so unseeded edges mean a 503 on the
    # first failover.
    edges = [e for edges in ROUTE.fallback_graph.values() for e in edges]
    assert edges and all(e.seed_state == "safe" for e in edges)


def test_the_simulator_demo_tenants_are_free_capped_and_keep_nothing() -> None:
    for name in ("demo-sim", "demo-chaos"):
        tenant = CONFIG.tenants[name]
        assert tenant.allowed_providers == ["mock"]  # simulator only: no provider, no cost
        assert tenant.logging == "metadata_only" and tenant.evidence_sampling == "off"
        assert CONFIG.routes[tenant.route].evidence is None
    sim = CONFIG.tenants["demo-sim"]
    assert sim.request_limits.max_messages == 20 and sim.request_limits.max_tokens == 512
    assert sim.limits.per_ip_rpm is not None and sim.budget_mode == "hard"


def test_the_chaos_route_has_a_safe_fallback_for_the_strict_demo_to_use() -> None:
    edges = CONFIG.routes["demo-chaos"].fallback_graph["mock/strong"]
    assert [(e.to, e.seed_state) for e in edges] == [("mock/cheap", "safe")]
