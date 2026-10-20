"""Tenant rules (FR-3) and per-request cost (FR-9), through the HTTP API.

The `limited` tenant in tests/config/test.yaml may call only mock/cheap, defaults to batch
priority, and allows one cohort tag. The `test` tenant may call both mock models.
"""

import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI

from relay.providers.mock import MockProvider
from relay.telemetry.log_writer import MemorySink

CHAT = "/v1/chat/completions"


def _body(model: str, **extra: Any) -> dict[str, Any]:
    return {"model": model, "messages": [{"role": "user", "content": "hello there"}], **extra}


@asynccontextmanager
async def serving(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    # Run the lifespan, so the log writer is running and rows reach the sink.
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://relay.test") as c:
            yield c


async def test_disallowed_model_is_404_like_a_missing_one(
    make_app: Callable[..., FastAPI],
) -> None:
    app = make_app(MockProvider(), auth_disabled_tenant="limited")
    async with serving(app) as client:
        explicit = await client.post(CHAT, json=_body("mock/strong"))
        missing = await client.post(CHAT, json=_body("mock/nope"))
        alias = await client.post(CHAT, json=_body("relay/strong"))  # not on its route
        allowed = await client.post(CHAT, json=_body("relay/cheap"))
    assert explicit.status_code == missing.status_code == alias.status_code == 404
    # Same message: probing can't tell "exists but not yours" from "doesn't exist".
    assert explicit.json()["error"]["message"].endswith(missing.json()["error"]["message"][-60:])
    assert allowed.status_code == 200


async def test_models_list_is_per_tenant(make_app: Callable[..., FastAPI]) -> None:
    async with serving(make_app(MockProvider(), auth_disabled_tenant="limited")) as client:
        limited = [m["id"] for m in (await client.get("/v1/models")).json()["data"]]
    async with serving(make_app(MockProvider())) as client:
        full = [m["id"] for m in (await client.get("/v1/models")).json()["data"]]
    assert limited == ["relay/cheap", "mock/cheap"]
    assert set(full) == {"relay/cheap", "relay/strong", "relay/auto", "mock/cheap", "mock/strong"}


async def test_cohort_allowlist_and_default_priority(make_app: Callable[..., FastAPI]) -> None:
    sink = MemorySink()
    app = make_app(MockProvider(), sink=sink, auth_disabled_tenant="limited")
    async with serving(app) as client:
        await client.post(CHAT, json=_body("relay/cheap"), headers={"X-Relay-Cohort": "lang=hi"})
        await client.post(CHAT, json=_body("relay/cheap"), headers={"X-Relay-Cohort": "user-123"})
        await client.post(
            CHAT, json=_body("relay/cheap"), headers={"X-Relay-Priority": "interactive"}
        )
    cohorts = [r.cohort for r in sink.rows]
    priorities = [r.priority for r in sink.rows]
    assert cohorts == ["lang=hi", "other", None]  # unknown tags are bucketed, never stored raw
    assert priorities == ["batch", "batch", "interactive"]  # tenant default, unless asked


# ---------------------------------------------------------------- cost
# mock/cheap is $0.10/1M in, $0.40/1M out. "hello there" = 2 prompt tokens; the mock replies
# "mock: hello there" = 3 tokens. List cost = 2*0.10 + 3*0.40 = 1.4 $/1M = 1,400 nano-USD.
# Counterfactual on mock/strong ($1/$4): 2*1 + 3*4 = 14 $/1M = 14,000 nano-USD.


async def test_cost_header_and_log_row(make_app: Callable[..., FastAPI]) -> None:
    sink = MemorySink()
    async with serving(make_app(MockProvider(), sink=sink)) as client:
        r = await client.post(CHAT, json=_body("relay/cheap"))
    assert r.headers["x-relay-cost-usd"] == "0.0000014"
    [row] = sink.rows
    assert (row.billed_cost_nano_usd, row.list_cost_nano_usd) == (1400, 1400)
    assert row.counterfactual_strong_nano_usd == 14_000
    assert row.price_table_version == "test-1"


async def test_stream_cost_is_in_the_final_usage_chunk(make_app: Callable[..., FastAPI]) -> None:
    sink = MemorySink()
    async with serving(make_app(MockProvider(), sink=sink)) as client:
        r = await client.post(
            CHAT,
            json=_body("relay/cheap", stream=True, stream_options={"include_usage": True}),
        )
    assert "x-relay-cost-usd" not in r.headers  # headers left before the cost was known
    events = [
        line.removeprefix("data: ") for line in r.text.splitlines() if line.startswith("data: {")
    ]
    final = json.loads(events[-1])
    assert final["relay"]["cost_usd"] == "0.0000014"
    assert final["relay"]["list_cost_usd"] == "0.0000014"
    assert sink.rows[0].list_cost_nano_usd == 1400


async def test_failed_request_has_no_cost(make_app: Callable[..., FastAPI]) -> None:
    sink = MemorySink()
    async with serving(make_app(MockProvider(), sink=sink)) as client:
        await client.post(CHAT, json=_body("mock/nope"))
    [row] = sink.rows
    assert row.list_cost_nano_usd is None
