"""FR-11 acceptance through the gateway: a shadow bundle never changes a response, a 10% canary
routes about 10%, and a rollback lands within one poll interval. Plus the cache scope, the
admin API, readiness and the automatic rollback of an unsafe canary."""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from scipy.stats import binomtest

from relay.api.app import create_app
from relay.evidence.cards import RouteCard
from relay.policy import control
from relay.policy.auto import rollback_unsafe_canary
from relay.policy.store import MemoryPolicyStore
from relay.providers.mock import MockProvider
from relay.routing.router import TrainedRouter
from relay.telemetry.log_writer import MemorySink

from ..conftest import test_settings as relay_settings  # type: ignore[import-not-found]
from ..routing.conftest import (  # type: ignore[import-not-found]
    StubEmbedder,
    length_artifact,
    routed_config,
)
from .conftest import V1, V2, v1, v2

CHAT = "/v1/chat/completions"


def body(content: str = "hi", model: str = "relay/auto", **kw: Any) -> dict[str, Any]:
    return {"model": model, "messages": [{"role": "user", "content": content}], **kw}


def build(
    store: MemoryPolicyStore | None = None,
    sink: MemorySink | None = None,
    config: Any = None,
    **settings: Any,
) -> tuple[FastAPI, MemoryPolicyStore]:
    store = store if store is not None else MemoryPolicyStore()
    config = config or routed_config()
    emb = StubEmbedder()
    router = TrainedRouter(
        config, dict(config.models), artifacts={"default": length_artifact()}, embedder=emb
    )
    app = create_app(
        relay_settings(**settings),
        providers={"mock": MockProvider()},
        config=config,
        router=router,
        embedder=emb,
        log_sink=sink,
        policy_store=store,
    )
    return app, store


@asynccontextmanager
async def serving(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://relay.test") as c:
            yield c


async def prepared(*ops: Any) -> MemoryPolicyStore:
    """A store with both bundles and the pointer moved by ``ops``."""
    store = MemoryPolicyStore()
    await store.create(v1(), created_by="t")
    await store.create(v2(), created_by="t")
    for op in ops:
        await store.change("test", op, actor="t")
    return store


def activate_v1(p: control.Pointer) -> control.Pointer:
    return control.activate(p, V1)


# ------------------------------------------------------------------ no bundle
async def test_without_a_bundle_nothing_changes() -> None:
    sink = MemorySink()
    app, _ = build(sink=sink)
    async with serving(app) as c:
        await app.state.policy_poller.refresh()
        r = await c.post(CHAT, json=body())
    assert r.headers["x-relay-model"] == "mock/cheap"
    assert "x-relay-policy" not in r.headers
    [row] = sink.rows
    assert row.policy_version == "t-v1"  # the router artifact's version, as before
    assert row.canary_arm is None and row.shadow_tier is None


async def test_an_active_bundle_serves_and_is_logged() -> None:
    sink = MemorySink()
    app, _ = build(await prepared(lambda p: control.activate(p, V2)), sink=sink)
    async with serving(app) as c:
        await app.state.policy_poller.refresh()
        r = await c.post(CHAT, json=body())
    assert r.headers["x-relay-policy"] == V2
    assert r.headers["x-relay-model"] == "mock/strong"  # V2's threshold sends "hi" strong
    assert sink.rows[0].policy_version == V2


# ------------------------------------------------------------------ shadow
async def test_a_shadow_bundle_never_changes_a_response() -> None:
    plain_sink, shadow_sink = MemorySink(), MemorySink()
    plain, _ = build(await prepared(activate_v1), sink=plain_sink)
    shadowed, _ = build(
        await prepared(activate_v1, lambda p: control.shadow(p, V2)), sink=shadow_sink
    )
    prompts = ["hi", "x" * 400, "what is 2+2?", "ok"]
    out: dict[str, list[Any]] = {"plain": [], "shadow": []}
    for name, app in (("plain", plain), ("shadow", shadowed)):
        async with serving(app) as c:
            await app.state.policy_poller.refresh()
            for p in prompts:
                r = await c.post(CHAT, json=body(p))
                out[name].append((r.status_code, r.headers["x-relay-model"], r.json()["choices"]))
                s = await c.post(CHAT, json=body(p, stream=True))
                out[name].append((s.status_code, s.headers["x-relay-model"], s.text.count("data:")))
    assert out["plain"] == out["shadow"]
    # The candidate decided too, and only the log knows: "hi" would have gone strong under V2.
    first = shadow_sink.rows[0]
    assert (first.model, first.shadow_policy_version, first.shadow_tier) == (
        "mock/cheap",
        V2,
        "strong",
    )
    assert first.shadow_score is not None
    assert all(row.shadow_policy_version is None for row in plain_sink.rows)


async def test_a_failing_shadow_router_is_swallowed() -> None:
    app, _ = build(await prepared(activate_v1, lambda p: control.shadow(p, V2)))
    async with serving(app) as c:
        await app.state.policy_poller.refresh()
        snap = app.state.policy_runtime.snapshot()

        async def broken(*a: Any, **k: Any) -> Any:
            raise RuntimeError("candidate router exploded")

        object.__setattr__(snap.candidate.router, "decide", broken)
        r = await c.post(CHAT, json=body())
    assert r.status_code == 200


# ------------------------------------------------------------------ canary
async def test_a_ten_percent_canary_routes_about_ten_percent() -> None:
    sink = MemorySink(maxlen=10_000)
    app, _ = build(
        await prepared(activate_v1, lambda p: control.canary(p, V2, 10)),
        sink=sink,
        log_queue_max=10_000,
    )
    n = 1500
    async with serving(app) as c:
        await app.state.policy_poller.refresh()
        for i in range(n):
            r = await c.post(CHAT, json=body(), headers={"x-request-id": f"req-{i}"})
            # The arm decides the bundle, and the bundle decides the model.
            expected = "mock/strong" if r.headers["x-relay-policy"] == V2 else "mock/cheap"
            assert r.headers["x-relay-model"] == expected
    arms = [row.canary_arm for row in sink.rows]
    assert set(arms) == {"control", "canary"}
    k = arms.count("canary")
    assert binomtest(k, n, 0.10).pvalue > 0.001  # FR-11: "verified statistically"
    canary_rows = [row for row in sink.rows if row.canary_arm == "canary"]
    assert all(row.policy_version == V2 for row in canary_rows)


async def test_a_session_key_keeps_a_client_in_one_arm() -> None:
    sink = MemorySink()
    app, _ = build(await prepared(activate_v1, lambda p: control.canary(p, V2, 50)), sink=sink)
    async with serving(app) as c:
        await app.state.policy_poller.refresh()
        for i in range(30):
            await c.post(
                CHAT, json=body(), headers={"x-relay-session": "user-7", "x-request-id": f"r{i}"}
            )
    assert len({row.canary_arm for row in sink.rows}) == 1


# ------------------------------------------------------------------ rollback and propagation
async def test_rollback_lands_within_one_poll_interval() -> None:
    store = await prepared(activate_v1, lambda p: control.activate(p, V2))
    app, _ = build(store, policy_poll_s=0.05)
    async with serving(app) as c:
        for _ in range(100):
            if app.state.policy_poller.first_attempt_done:
                break
            await asyncio.sleep(0.01)
        assert (await c.post(CHAT, json=body())).headers["x-relay-policy"] == V2
        await store.change("rollback", control.rollback, actor="op")
        started = time.monotonic()
        while app.state.policy_runtime.snapshot().active.version != V1:
            assert time.monotonic() - started < 0.05 * 3  # one interval, plus scheduling slack
            await asyncio.sleep(0.005)
        assert (await c.post(CHAT, json=body())).headers["x-relay-policy"] == V1


async def test_a_database_outage_keeps_the_last_policy() -> None:
    store = await prepared(activate_v1)
    app, _ = build(store)
    async with serving(app) as c:
        poller = app.state.policy_poller
        assert await poller.refresh()
        await store.change("canary", lambda p: control.canary(p, V2, 50), actor="op")
        store.fail = True
        assert not await poller.refresh()
        r = await c.post(CHAT, json=body())
    assert r.status_code == 200 and r.headers["x-relay-policy"] == V1
    assert app.state.policy_runtime.snapshot().candidate is None


async def test_an_invalid_bundle_is_never_applied() -> None:
    from relay.config.schema import FallbackEdge

    from .conftest import bundle

    store = await prepared(activate_v1)
    bad = bundle("2026-10-28.9", 0.5, graph={"mock/strong": [FallbackEdge(to="ghost/model")]})
    await store.create(bad, created_by="t")
    await store.change("canary", lambda p: control.canary(p, bad.version, 10), actor="op")
    app, _ = build(store)
    async with serving(app) as c:
        assert not await app.state.policy_poller.refresh()
        r = await c.post(CHAT, json=body())
    assert r.status_code == 200
    assert app.state.policy_runtime.snapshot().active.version is None  # still the YAML policy


async def test_readiness_waits_for_the_first_poll() -> None:
    app, _ = build(await prepared(activate_v1), policy_poll_s=0.05)
    async with serving(app) as c:
        for _ in range(100):
            ready = (await c.get("/readyz")).json()
            if ready["status"] == "ready":
                break
            assert ready["status"] == "loading_policy"
            await asyncio.sleep(0.01)
    assert ready["status"] == "ready"
    assert ready["policy"]["active"] == V1


# ------------------------------------------------------------------ cache scope
async def test_cached_answers_never_cross_a_policy_change() -> None:
    from relay.config.tenants import TenantCache

    base = routed_config()
    tenant = base.tenants["test"].model_copy(update={"cache": TenantCache(enabled=True)})
    config = base.model_copy(update={"tenants": {**base.tenants, "test": tenant}})
    store = await prepared(activate_v1)
    app, _ = build(store, config=config)
    auto, cheap = body(temperature=0), body(model="relay/cheap", temperature=0)
    async with serving(app) as c:
        await app.state.policy_poller.refresh()
        assert (await c.post(CHAT, json=auto)).headers["x-relay-cache"] == "miss"
        assert (await c.post(CHAT, json=auto)).headers["x-relay-cache"] == "hit"
        assert (await c.post(CHAT, json=cheap)).headers["x-relay-cache"] == "miss"
        await store.change("activate", lambda p: control.activate(p, V2), actor="op")
        await app.state.policy_poller.refresh()
        # relay/auto: V2 routes differently, so V1's answer must not be served.
        after = await c.post(CHAT, json=auto)
        assert after.headers["x-relay-cache"] == "miss"
        assert after.headers["x-relay-model"] == "mock/strong"
        # An explicit alias doesn't depend on the router: its entry survives.
        assert (await c.post(CHAT, json=cheap)).headers["x-relay-cache"] == "hit"


# ------------------------------------------------------------------ admin API
async def test_the_admin_api_is_off_without_a_token() -> None:
    app, _ = build(await prepared())
    async with serving(app) as c:
        assert (await c.get("/v1/admin/policy")).status_code == 404
        assert (await c.post("/v1/admin/policy/rollback", json={})).status_code == 404


async def test_the_admin_api_needs_the_right_token() -> None:
    app, store = build(await prepared(activate_v1), admin_token="s3cret")
    good = {"authorization": "Bearer s3cret"}
    async with serving(app) as c:
        wrong = await c.get("/v1/admin/policy", headers={"authorization": "Bearer nope"})
        assert wrong.status_code == 401
        r = await c.post("/v1/admin/policy/canary", json={"version": V2, "pct": 10}, headers=good)
        assert r.status_code == 200
        # This replica applied it at once (no wait for the poll).
        assert app.state.policy_runtime.snapshot().candidate.version == V2
        status = (await c.get("/v1/admin/policy", headers=good)).json()
        assert status["serving"] == {"active": V1, "candidate": V2, "mode": "canary",
                                     "canary_pct": 10}  # fmt: skip
        assert status["events"][0]["actor"] == "admin-api"
        assert (await c.post("/v1/admin/policy/promote", json={}, headers=good)).status_code == 200
        bad = await c.post("/v1/admin/policy/promote", json={}, headers=good)
        assert bad.status_code == 409  # nothing left to promote
        missing = await c.post("/v1/admin/policy/canary", json={"version": V1}, headers=good)
        assert missing.status_code == 422
        ghost = await c.post(
            "/v1/admin/policy/activate", json={"version": "2099-01-01.1"}, headers=good
        )
        assert ghost.status_code == 409
    assert (await store.pointer()).active_version == V2


async def test_the_admin_api_without_a_database_says_so() -> None:
    config = routed_config()
    app = create_app(
        relay_settings(admin_token="t"),
        providers={"mock": MockProvider()},
        config=config,
        router=TrainedRouter(
            config, dict(config.models), artifacts={"default": length_artifact()},
            embedder=StubEmbedder(),
        ),
    )  # fmt: skip
    async with serving(app) as c:
        r = await c.get("/v1/admin/policy", headers={"authorization": "Bearer t"})
    assert r.status_code == 503


# ------------------------------------------------------------------ automatic rollback
def canary_card(state: str, version: str = V2, action: str = "rollback") -> RouteCard:
    return RouteCard(
        route="default", cohort="all", comparison=f"canary:{version}", estimate=0.84,
        ci=(0.80, 0.88), floor=0.9, n_judge=500, n_human=100, judge="j@v1", judge_kappa=0.7,
        state=state, recommended_action=action, look=2, max_looks=4, policy_version=version,
        updated_at=datetime.now(UTC),
    )  # type: ignore[arg-type]  # fmt: skip


async def test_an_unsafe_canary_is_rolled_back_by_the_worker() -> None:
    store = await prepared(activate_v1, lambda p: control.canary(p, V2, 10))
    await rollback_unsafe_canary(store)(canary_card("unsafe"))
    p = await store.pointer()
    assert (p.active_version, p.candidate_version) == (V1, None)
    [event] = await store.events(1)
    assert event.actor == "evidence-worker" and "unsafe" in event.reason


@pytest.mark.parametrize(
    ("card", "pointer_ops"),
    [
        # Not a rollback card.
        (canary_card("undecided", action="collect_more"), [lambda p: control.canary(p, V2, 10)]),
        # A canary that is no longer running (promoted since): never undo a human's decision.
        (canary_card("unsafe"), [lambda p: control.canary(p, V2, 10), control.promote]),
        # A different version is the canary now.
        (canary_card("unsafe", version="2026-10-27.1"), [lambda p: control.canary(p, V2, 10)]),
    ],
)
async def test_the_worker_never_rolls_back_anything_else(card: RouteCard, pointer_ops: Any) -> None:
    store = await prepared(activate_v1, *pointer_ops)
    before = await store.pointer()
    await rollback_unsafe_canary(store)(card)
    assert await store.pointer() == before
