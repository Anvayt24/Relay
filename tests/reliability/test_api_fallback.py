"""Fallback through the real HTTP layer: headers, log rows, the OpenAI SDK, drain."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import openai
from fastapi import FastAPI

from relay.api.app import create_app
from relay.providers.base import ProviderAdapter
from relay.schemas.errors import ErrorClass
from relay.simulator.plan import MockBehavior
from relay.telemetry.log_writer import MemorySink

from ..conftest import test_settings as relay_settings  # type: ignore[import-not-found]
from .conftest import FALLBACK_CONFIG, ScriptedProvider  # type: ignore[import-not-found]

CHAT = "/v1/chat/completions"
BODY = {"model": "relay/auto", "messages": [{"role": "user", "content": "hi"}]}


def build(
    tenant: str, primary: ScriptedProvider, sink: MemorySink | None = None
) -> tuple[FastAPI, dict[str, ScriptedProvider]]:
    provs = {
        "primary": primary,
        "backup": ScriptedProvider(),
        "big": ScriptedProvider(),
        "capped": ScriptedProvider(),
    }
    adapters: dict[str, ProviderAdapter] = dict(provs)
    app = create_app(
        relay_settings(auth_disabled_tenant=tenant, first_token_timeout_s=1.0),
        providers=adapters,
        log_sink=sink,
        config=FALLBACK_CONFIG,
    )
    return app, provs


@asynccontextmanager
async def serving(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://relay.test") as c:
            yield c


def broken() -> ScriptedProvider:
    return ScriptedProvider(then=MockBehavior(error=ErrorClass.SERVER_ERROR))


async def test_headers_and_log_row_name_the_model_that_answered() -> None:
    sink = MemorySink()
    app, provs = build("strict", broken(), sink)
    async with serving(app) as client:
        r = await client.post(CHAT, json=BODY)
    assert r.status_code == 200
    assert r.headers["x-relay-provider"] == "backup"
    assert r.headers["x-relay-model"] == "b/model"
    assert "x-relay-degraded" not in r.headers  # a safe edge: absent, never "false"
    assert r.json()["model"] == "b/model"

    [row] = sink.rows
    assert (row.provider, row.model, row.tier) == ("backup", "b/model", "cheap")
    assert row.fallback_hops == 1 and not row.degraded
    assert [(a.model, a.retry, a.error_class) for a in row.attempts] == [
        ("p/model", 0, "server_error"),
        ("p/model", 1, "server_error"),
        ("b/model", 0, None),
    ]
    assert provs["primary"].calls == 2


async def test_degraded_edge_sets_the_header_and_the_row_flag() -> None:
    sink = MemorySink()
    app, _ = build("degraded", broken(), sink)
    async with serving(app) as client:
        r = await client.post(CHAT, json=BODY)
    assert r.status_code == 200 and r.headers["x-relay-degraded"] == "true"
    assert sink.rows[0].degraded


async def test_strict_cold_start_returns_the_primarys_error_with_no_retry_header() -> None:
    app, provs = build("strict", broken())
    # `degraded` tenant is allow_degraded; build a strict tenant on the `gated` route.
    config = FALLBACK_CONFIG.model_copy(
        update={
            "tenants": {
                **FALLBACK_CONFIG.tenants,
                "cold": FALLBACK_CONFIG.tenants["strict"].model_copy(
                    update={"id": "cold", "route": "gated"}
                ),
            }
        }
    )
    app = create_app(
        relay_settings(auth_disabled_tenant="cold"),
        providers=dict(provs),  # type: ignore[arg-type]
        config=config,
    )
    async with serving(app) as client:
        r = await client.post(CHAT, json=BODY)
    assert r.status_code == 502 and r.headers["x-should-retry"] == "false"
    assert provs["backup"].calls == 0


async def test_total_failure_is_503_no_usable_fallback() -> None:
    app, provs = build("strict", broken())
    provs["backup"]._then = MockBehavior(error=ErrorClass.OVERLOADED)
    async with serving(app) as client:
        r = await client.post(CHAT, json=BODY)
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "no_usable_fallback"
    assert r.headers["x-should-retry"] == "false" and "retry-after" in r.headers


async def test_stream_falls_back_before_first_byte_and_names_the_server() -> None:
    app, provs = build("strict", ScriptedProvider(then=MockBehavior(first_chunk_delay_s=5)))
    async with serving(app) as client:
        r = await client.post(CHAT, json={**BODY, "stream": True})
    assert r.status_code == 200
    assert r.headers["x-relay-model"] == "b/model"
    assert r.text.rstrip().endswith("data: [DONE]")
    assert provs["primary"].calls == 1


async def test_readyz_reports_draining_during_shutdown() -> None:
    app, _ = build("strict", ScriptedProvider())
    async with serving(app) as client:
        assert (await client.get("/readyz")).status_code == 200
        app.state.draining = True
        r = await client.get("/readyz")
    assert r.status_code == 503 and r.json()["status"] == "draining"


async def test_sdk_adds_no_retries_on_top_of_relays_own() -> None:
    app, provs = build("strict", broken())
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://relay.test/v1") as http,
    ):
        sdk = openai.AsyncOpenAI(
            api_key="k",
            base_url="http://relay.test/v1",
            http_client=http,  # type: ignore[arg-type]
        )
        # Fully failing route: primary 2 calls (call + retry), backup 1 (fails) -> 503.
        provs["backup"]._then = MockBehavior(error=ErrorClass.OVERLOADED)
        try:
            await sdk.chat.completions.create(
                model="relay/auto", messages=[{"role": "user", "content": "hi"}]
            )
        except openai.APIStatusError as e:
            assert e.status_code == 503
    assert provs["primary"].calls == 2
    assert provs["backup"].calls == 2  # overloaded is also retried once on the fallback
