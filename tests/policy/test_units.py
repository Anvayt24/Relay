"""Policy bundles, the pointer transitions, and the runtime, without HTTP (ADRs 0023, 0024)."""

from datetime import date

import pytest

from relay.config.loader import DEFAULT_CONFIG, load_config
from relay.config.schema import FallbackEdge
from relay.pipeline.context import RequestContext
from relay.policy import control
from relay.policy.bundle import (
    BundleError,
    BundleRouter,
    PolicyBundle,
    RoutePolicy,
    artifacts,
    effective_config,
    from_config,
    next_version,
)
from relay.policy.control import ControlError, Pointer
from relay.policy.runtime import PolicyRuntime, PolicySnapshot, canary_arm
from relay.routing.base import RouteDecision
from relay.routing.static import StaticRouter

from ..routing.conftest import chat, length_artifact, routed_config  # type: ignore[import-not-found]
from .conftest import V1, V2, bundle, v1, v2

BASE = routed_config()


def runtime() -> PolicyRuntime:
    return PolicyRuntime(
        BASE,
        dict(BASE.models),
        bootstrap=StaticRouter(BASE, dict(BASE.models)),
        router_factory=lambda cfg, arts: StaticRouter(cfg, dict(cfg.models)),
    )


# ------------------------------------------------------------------ bundles
def test_the_shipped_config_bundles_and_round_trips() -> None:
    config = load_config()
    b = from_config(config, DEFAULT_CONFIG.parent, version="2026-10-28.1", note="first")
    again = PolicyBundle.from_json(b.to_json())
    assert again.content_sha256 == b.content_sha256
    assert set(artifacts(again)) == {"default"}  # the only routed route
    assert effective_config(config, again).routes["default"].router.threshold == 0.82  # type: ignore[union-attr]


def test_the_hash_covers_the_policy_not_the_labels() -> None:
    a, b = bundle(V1, 0.5), bundle(V2, 0.5)
    assert a.content_sha256 == b.content_sha256  # same policy, different version label
    assert bundle(V1, 0.6).content_sha256 != a.content_sha256


def test_versions_are_dated_and_numbered() -> None:
    day = date(2026, 10, 28)
    assert next_version([], day) == "2026-10-28.1"
    assert next_version(["2026-10-28.1", "2026-10-28.3", "2026-10-27.9"], day) == "2026-10-28.4"
    with pytest.raises(BundleError):
        from_config(load_config(), DEFAULT_CONFIG.parent, version="latest")


def test_a_bundle_is_validated_against_this_deployment() -> None:
    with pytest.raises(BundleError, match="unknown routes"):
        effective_config(BASE, PolicyBundle(version=V1, routes={"nope": RoutePolicy()}))
    bad_model = bundle(V1, 0.5, graph={"mock/strong": [FallbackEdge(to="ghost/model")]})
    with pytest.raises(BundleError, match="unknown model"):
        effective_config(BASE, bad_model)
    cycle = bundle(
        V1,
        0.5,
        graph={
            "mock/strong": [FallbackEdge(to="mock/cheap")],
            "mock/cheap": [FallbackEdge(to="mock/strong")],
        },
    )
    with pytest.raises(BundleError, match="cycle"):
        effective_config(BASE, cycle)


def test_a_tampered_artifact_is_refused() -> None:
    art = length_artifact()
    tampered = PolicyBundle(
        version=V1,
        routes={
            "default": RoutePolicy(
                router=BundleRouter(
                    artifact_json=art.dumps(), artifact_sha256="0" * 64, threshold=0.5
                )
            )
        },
    )
    with pytest.raises(BundleError, match="hash mismatch"):
        artifacts(tampered)
    garbage = tampered.model_copy(
        update={
            "routes": {
                "default": RoutePolicy(
                    router=BundleRouter(artifact_json="{}", artifact_sha256="0", threshold=0.5)
                )
            }
        }
    )
    with pytest.raises(BundleError):
        artifacts(garbage)


def test_a_route_the_bundle_omits_keeps_its_yaml_policy() -> None:
    cfg = effective_config(BASE, v2())
    assert cfg.routes["cheap_only"] == BASE.routes["cheap_only"]
    assert cfg.routes["default"].router.threshold == 0.99  # type: ignore[union-attr]
    assert cfg.tenants == BASE.tenants  # infrastructure is never in a bundle


# ------------------------------------------------------------------ pointer transitions
def test_the_operations_move_the_pointer_as_documented() -> None:
    p = control.activate(Pointer(), V1)
    assert (p.active_version, p.previous_version) == (V1, None)
    p = control.canary(p, V2, 10)
    assert (p.candidate_version, p.candidate_mode, p.canary_pct) == (V2, "canary", 10)
    rolled = control.rollback(p)  # with a candidate: drop it, active never changed
    assert (rolled.active_version, rolled.candidate_version) == (V1, None)
    p = control.promote(control.shadow(p, V2))
    assert (p.active_version, p.previous_version, p.candidate_version) == (V2, V1, None)
    p = control.rollback(p)  # without a candidate: back to the previous version
    assert (p.active_version, p.previous_version) == (V1, None)
    assert p.updated_at is not None


@pytest.mark.parametrize(
    ("op", "pointer"),
    [
        (lambda p: control.promote(p), Pointer(active_version=V1)),
        (lambda p: control.rollback(p), Pointer(active_version=V1)),
        (lambda p: control.canary(p, V1, 10), Pointer(active_version=V1)),
        (lambda p: control.canary(p, V2, 0), Pointer(active_version=V1)),
        (lambda p: control.canary(p, V2, 101), Pointer(active_version=V1)),
        (lambda p: control.activate(p, V1), Pointer(active_version=V1)),
    ],
)
def test_operations_that_make_no_sense_are_refused(op: object, pointer: Pointer) -> None:
    with pytest.raises(ControlError):
        op(pointer)  # type: ignore[operator]


# ------------------------------------------------------------------ runtime
def test_a_swap_never_changes_a_snapshot_already_taken() -> None:
    rt = runtime()
    before = rt.snapshot()
    rt.apply(v1(), v2(), mode="canary", canary_pct=10)
    after = rt.snapshot()
    assert before.active.version is None and before.candidate is None  # untouched (frozen)
    assert (after.active.version, after.candidate.version) == (V1, V2)  # type: ignore[union-attr]


def test_an_invalid_bundle_changes_nothing() -> None:
    rt = runtime()
    rt.apply(v1())
    bad = bundle(V2, 0.5, graph={"mock/strong": [FallbackEdge(to="ghost/model")]})
    with pytest.raises(BundleError):
        rt.apply(v1(), bad, mode="canary", canary_pct=50)
    assert rt.snapshot().active.version == V1 and rt.snapshot().candidate is None


def test_built_policies_are_reused_so_rollback_is_instant() -> None:
    rt = runtime()
    first = rt.build(v1())
    rt.apply(v2())
    rt.apply(v1())
    assert rt.snapshot().active is first
    rt.apply(None)  # the pointer names nothing: back to the YAML policy
    assert rt.snapshot().active.version is None


def test_no_candidate_means_no_mode() -> None:
    snap = runtime().apply(v1(), None, mode="canary", canary_pct=50)
    assert (snap.mode, snap.canary_pct) == ("active", 0)
    assert snap.arm("anything") is None and snap.shadow() is None


def test_canary_assignment_is_deterministic_proportional_and_reshuffled() -> None:
    keys = [f"req-{i}" for i in range(20_000)]
    arms = [canary_arm(V2, k, 10) for k in keys]
    share = arms.count("canary") / len(keys)
    assert 0.09 < share < 0.11
    assert arms == [canary_arm(V2, k, 10) for k in keys]  # same key, same arm, every time
    other = [canary_arm("2026-10-28.3", k, 10) for k in keys]
    overlap = sum(a == b == "canary" for a, b in zip(arms, other, strict=True))
    assert overlap < 0.03 * len(keys)  # a new canary draws a fresh 10%, not the same users
    assert all(canary_arm(V2, k, 100) == "canary" for k in keys[:100])


def test_a_snapshot_applies_the_candidate_only_to_its_arm() -> None:
    rt = runtime()
    snap: PolicySnapshot = rt.apply(v1(), v2(), mode="canary", canary_pct=50)
    assert snap.applied("canary").version == V2
    assert snap.applied("control").version == V1
    assert snap.applied(None).version == V1


async def test_the_dispatcher_plans_with_the_applied_bundles_graph() -> None:
    from relay.admission.store import MemoryStore
    from relay.providers.mock import MockProvider
    from relay.reliability.breaker import BreakerRegistry
    from relay.reliability.dispatch import ReliableDispatcher
    from relay.reliability.edges import SeededEdgeStates
    from relay.reliability.retry_budget import RetryBudget

    rt = runtime()
    graph = {"mock/strong": [FallbackEdge(to="mock/cheap", quality_gate="none")]}
    built = rt.build(bundle(V2, 0.5, graph=graph))
    dispatcher = ReliableDispatcher(
        BASE,
        dict(BASE.models),
        {"mock": MockProvider()},
        store=MemoryStore(),
        pause=None,
        breakers=BreakerRegistry(BASE.reliability),
        retry_budget=RetryBudget(ratio=0.1, minimum=3, window_s=10),
        edge_states=SeededEdgeStates(BASE),
        first_token_ceiling_s=5,
    )
    ctx = RequestContext(
        request_id="r",
        tenant=BASE.tenants["test"],
        request=chat("hi"),
        route="default",
        priority="interactive",
        cohort=None,
        policy_version=V2,
        deadline=1e12,
        stream=False,
        decision=RouteDecision("strong", "mock/strong", None, "alias", V2),
    )
    assert [c.model for c in dispatcher.plan(ctx)] == ["mock/strong"]  # the YAML: no graph
    ctx.applied = built
    assert [c.model for c in dispatcher.plan(ctx)] == ["mock/strong", "mock/cheap"]
