"""What Relay decided and why, in every response: the router's score, what the
always-strong model would have cost, the fallback hops used, and the latency.

The demo page reads these; so can any client. They are headers on a JSON response and, for a
stream, fields of the final `relay` chunk, because a stream's headers are sent before its cost
is known.
"""

import json

import httpx
import pytest

from relay.api.app import create_app
from relay.api.decision import decision_fields
from relay.pipeline.context import RequestContext
from relay.providers.mock import MockProvider
from relay.routing.base import RouteDecision

from ..conftest import test_settings as relay_settings  # type: ignore[import-not-found]
from ..reliability.conftest import make_ctx  # type: ignore[import-not-found]

CHAT = "/v1/chat/completions"


def _client() -> httpx.AsyncClient:
    app = create_app(relay_settings(), providers={"mock": MockProvider()})
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://relay.test")


async def test_a_json_response_carries_the_decision_fields() -> None:
    body = {"model": "relay/cheap", "messages": [{"role": "user", "content": "hi"}]}
    async with _client() as c:
        r = await c.post(CHAT, json=body)
    assert r.status_code == 200
    assert r.headers["x-relay-fallback-hops"] == "0"
    assert int(r.headers["x-relay-latency-ms"]) >= 0
    # Served cheap, so always-strong would have cost more (a list-price counterfactual).
    assert float(r.headers["x-relay-strong-cost-usd"]) > float(r.headers["x-relay-cost-usd"])
    assert "x-relay-router-score" not in r.headers  # no router on this route: absent, not "None"


async def test_a_stream_carries_them_in_the_final_chunk() -> None:
    body = {
        "model": "relay/cheap",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    async with _client() as c:
        r = await c.post(CHAT, json=body)
    last = [line for line in r.text.splitlines() if line.startswith("data: {")][-1]
    extension = json.loads(last[len("data: ") :])["relay"]
    assert extension["fallback_hops"] == "0"
    assert float(extension["strong_cost_usd"]) > float(extension["cost_usd"])
    assert int(extension["latency_ms"]) >= 0
    # The headers are sent with the first chunk, so they carry what was known by then.
    assert r.headers["x-relay-fallback-hops"] == "0"
    assert "x-relay-strong-cost-usd" not in r.headers


def _ctx() -> RequestContext:
    ctx = make_ctx()
    ctx.decision = RouteDecision(
        tier="cheap", model="p/model", score=0.87654, reason="router", policy_version="t"
    )
    return ctx


def test_the_router_score_is_reported_to_four_places_when_a_router_decided() -> None:
    assert decision_fields(_ctx())["router_score"] == "0.8765"


@pytest.mark.parametrize("score", [None])
def test_no_score_means_no_field(score: float | None) -> None:
    ctx = _ctx()
    assert ctx.decision is not None
    ctx.decision = RouteDecision(
        tier="cheap", model="p/model", score=score, reason="alias", policy_version="t"
    )
    assert "router_score" not in decision_fields(ctx)


async def test_a_fallback_shows_in_the_hops() -> None:
    from relay.schemas.errors import ErrorClass
    from relay.simulator.plan import MockBehavior

    from ..reliability.conftest import FALLBACK_CONFIG, ScriptedProvider  # type: ignore[import-not-found]

    providers = {
        "primary": ScriptedProvider(then=MockBehavior(error=ErrorClass.SERVER_ERROR)),
        "backup": ScriptedProvider(),
        "big": ScriptedProvider(),
        "capped": ScriptedProvider(),
    }
    app = create_app(
        relay_settings(auth_disabled_tenant="strict", first_token_timeout_s=1.0),
        providers=providers,  # type: ignore[arg-type]
        config=FALLBACK_CONFIG,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://relay.test") as c:
        r = await c.post(
            CHAT, json={"model": "relay/auto", "messages": [{"role": "user", "content": "x"}]}
        )
    assert r.status_code == 200 and r.headers["x-relay-fallback-hops"] == "1"


def test_a_router_fallback_says_why_so_a_missing_score_is_not_a_mystery() -> None:
    ctx = _ctx()
    ctx.decision = RouteDecision(
        tier="strong",
        model="p/model",
        score=None,
        reason="router_fallback:embedder_not_ready",
        policy_version="t",
    )
    assert decision_fields(ctx)["router_reason"] == "router_fallback:embedder_not_ready"
    ctx.decision = RouteDecision(
        tier="cheap", model="p/model", score=0.9, reason="router", policy_version="t"
    )
    assert decision_fields(ctx)["router_reason"] == "router"
    ctx.decision = RouteDecision(
        tier="cheap", model="p/model", score=None, reason="alias", policy_version="t"
    )
    assert "router_reason" not in decision_fields(ctx)  # not decided by a router
