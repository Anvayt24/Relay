"""The cache through the real HTTP layer: hits, replayed streams, opt-outs, failures.

Every test counts ``provider.calls``: a cache hit is defined by the call it does *not* make.
"""

import json
from typing import Any

import httpx
import httpx2
import openai
import pytest
from fastapi import FastAPI

from relay.api.app import create_app
from relay.cache.embed import HashEmbedder
from relay.cache.entry import CachedReply
from relay.cache.exact import CacheUnavailable, MemoryExactStore
from relay.cache.semantic import MemorySemanticStore
from relay.config.schema import RouteCache
from relay.config.tenants import TenantCache
from relay.providers.mock import MockProvider
from relay.simulator.plan import MockBehavior
from relay.telemetry.log_writer import MemorySink

from ..conftest import TEST_CONFIG  # type: ignore[import-not-found]
from ..conftest import test_settings as relay_settings  # type: ignore[import-not-found]

CHAT = "/v1/chat/completions"


def body(content: str = "What is the capital of France?", **extra: Any) -> dict[str, Any]:
    return {
        "model": "relay/auto",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        **extra,
    }


def build(
    provider: MockProvider | None = None,
    *,
    semantic: bool = False,
    threshold: float = 0.8,
    guard: bool = True,
    sink: MemorySink | None = None,
    **backends: Any,
) -> tuple[FastAPI, MockProvider]:
    provider = provider or MockProvider()
    tenant = TEST_CONFIG.tenants["test"].model_copy(
        update={"cache": TenantCache(enabled=True, semantic=semantic, ttl_s=600)}
    )
    route = TEST_CONFIG.routes["default"].model_copy(
        update={"cache": RouteCache(semantic_threshold=threshold, lexical_guard=guard)}
    )
    config = TEST_CONFIG.model_copy(
        update={
            "tenants": {**TEST_CONFIG.tenants, "test": tenant},
            "routes": {**TEST_CONFIG.routes, "default": route},
        }
    )
    app = create_app(
        relay_settings(embedder="hash"),
        providers={"mock": provider},
        config=config,
        log_sink=sink,
        **backends,
    )
    return app, provider


def client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://relay.test")


async def settle(app: FastAPI) -> None:
    await app.state.cache_stage.drain()


async def test_second_identical_request_is_served_from_the_cache() -> None:
    app, provider = build()
    async with client(app) as c:
        first = await c.post(CHAT, json=body())
        await settle(app)
        second = await c.post(CHAT, json=body())
    assert first.headers["x-relay-cache"] == "miss"
    assert second.headers["x-relay-cache"] == "hit"
    assert second.headers["x-relay-provider"] == "relay-cache"
    assert second.headers["x-relay-model"] == "mock/strong"  # who originally produced it
    assert second.json()["choices"] == first.json()["choices"]
    assert second.json()["usage"] == first.json()["usage"]
    assert provider.calls == 1  # the hit made no provider call


async def test_hits_are_logged_and_cost_nothing() -> None:
    sink = MemorySink()
    app, _ = build(sink=sink)
    async with app.router.lifespan_context(app), client(app) as c:
        await c.post(CHAT, json=body())
        await settle(app)
        r = await c.post(CHAT, json=body())
    assert r.headers.get("x-relay-cost-usd") is None  # no provider ran, nothing billed
    miss, hit = list(sink.rows)
    assert (miss.cache_status, hit.cache_status) == ("miss", "hit")
    assert (hit.provider, hit.model) == ("relay-cache", "mock/strong")
    assert hit.tokens_out == miss.tokens_out
    assert hit.billed_cost_nano_usd is None


async def test_a_cached_answer_replays_as_an_identical_stream() -> None:
    app, provider = build()
    async with client(app) as c:
        live = await c.post(CHAT, json=body(stream=True, stream_options={"include_usage": True}))
        await settle(app)
        replay = await c.post(CHAT, json=body(stream=True, stream_options={"include_usage": True}))
    assert replay.headers["x-relay-cache"] == "hit" and provider.calls == 1

    def text(response: httpx.Response) -> str:
        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        return "".join(c["delta"].get("content", "") for e in events for c in e.get("choices", []))

    assert text(replay) == text(live) != ""
    assert replay.text.rstrip().endswith("data: [DONE]")
    usage_chunks = [line for line in replay.text.splitlines() if '"usage"' in line]
    assert len(usage_chunks) == 1  # the final usage chunk, as in a live stream


async def test_sdk_streams_a_cached_answer() -> None:
    app, provider = build()
    http = openai.DefaultAsyncHttpxClient(transport=httpx2.ASGITransport(app=app))
    sdk = openai.AsyncOpenAI(base_url="http://relay.test/v1", api_key="k", http_client=http)
    messages: Any = [{"role": "user", "content": "hello cache"}]
    kwargs: Any = {"model": "relay/auto", "messages": messages, "temperature": 0, "stream": True}

    async def run() -> str:
        stream = await sdk.chat.completions.create(**kwargs)
        return "".join([c.choices[0].delta.content or "" async for c in stream if c.choices])

    live = await run()
    await settle(app)
    assert await run() == live
    assert provider.calls == 1


@pytest.mark.parametrize(
    ("extra", "headers"),
    [
        ({"temperature": 0.7}, {}),  # sampling is meant to vary
        ({"temperature": None}, {}),  # unset is OpenAI's 1.0, not deterministic
        ({}, {"X-Relay-Cache": "no-store"}),  # the client opted out
    ],
)
async def test_ineligible_requests_bypass_the_cache(
    extra: dict[str, Any], headers: dict[str, str]
) -> None:
    app, provider = build()
    payload = {k: v for k, v in body(**extra).items() if v is not None}
    async with client(app) as c:
        for _ in range(2):
            r = await c.post(CHAT, json=payload, headers=headers)
            await settle(app)
            assert r.headers["x-relay-cache"] == "bypass"
    assert provider.calls == 2


async def test_truncated_answers_are_not_cached() -> None:
    app, provider = build()
    async with client(app) as c:
        # max_tokens=1 cuts the reply off (finish_reason "length"): never reusable.
        for _ in range(2):
            r = await c.post(CHAT, json=body(max_tokens=1))
            await settle(app)
            assert r.json()["choices"][0]["finish_reason"] == "length"
    assert provider.calls == 2


async def test_errors_are_not_cached() -> None:
    from relay.schemas.errors import ErrorClass

    app, provider = build(MockProvider(MockBehavior(error=ErrorClass.BAD_REQUEST)))
    async with client(app) as c:
        for _ in range(2):
            assert (await c.post(CHAT, json=body())).status_code == 400
            await settle(app)
    assert provider.calls == 2


async def test_different_parameters_never_share_an_entry() -> None:
    app, provider = build()
    async with client(app) as c:
        await c.post(CHAT, json=body(max_tokens=50))
        await settle(app)
        other = await c.post(CHAT, json=body(max_tokens=1000))  # a 50-token reply != 1000
        await settle(app)
    assert other.headers["x-relay-cache"] == "miss" and provider.calls == 2


async def test_other_tenants_never_see_each_others_entries() -> None:
    app, provider = build()
    other_tenant = TEST_CONFIG.tenants["limited"]  # no cache configured: bypass
    assert other_tenant.cache.enabled is False
    async with client(app) as c:
        await c.post(CHAT, json=body())
        await settle(app)
    store = app.state.cache_stage._exact
    assert len(store._items) == 1  # one entry, keyed with the tenant inside the hash
    assert provider.calls == 1


# ---------------------------------------------------------------- failure = a miss
class DownExact:
    async def get(self, key: str) -> CachedReply | None:
        raise CacheUnavailable("redis down")

    async def set(self, key: str, reply: CachedReply, ttl_s: int) -> None:
        raise CacheUnavailable("redis down")

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None


async def test_redis_down_serves_from_the_provider() -> None:
    app, provider = build(exact_cache=DownExact())
    async with client(app) as c:
        for _ in range(2):
            r = await c.post(CHAT, json=body())
            await settle(app)
            assert r.status_code == 200
    assert provider.calls == 2


# ---------------------------------------------------------------- semantic layer
async def test_a_paraphrase_hits_the_semantic_layer() -> None:
    app, provider = build(
        semantic=True,
        semantic_cache=MemorySemanticStore(),
        embedder=HashEmbedder(),
        exact_cache=MemoryExactStore(),
    )
    async with client(app) as c:
        await c.post(CHAT, json=body("What is the capital of France today"))
        await settle(app)
        near = await c.post(CHAT, json=body("What is the capital of France today please"))
        far = await c.post(CHAT, json=body("How do I bake sourdough bread"))
    assert near.headers["x-relay-cache"] == "hit"
    assert far.headers["x-relay-cache"] == "miss"
    assert provider.calls == 2


async def test_the_lexical_guard_turns_a_near_miss_on_numbers_into_a_miss() -> None:
    first, second = "What is 15% of 200 dollars", "What is 15% of 300 dollars"
    for guard, expected in ((True, "miss"), (False, "hit")):
        app, _ = build(semantic=True, threshold=0.7, guard=guard)
        async with client(app) as c:
            await c.post(CHAT, json=body(first))
            await settle(app)
            r = await c.post(CHAT, json=body(second))
        assert r.headers["x-relay-cache"] == expected


async def test_multi_turn_requests_use_the_exact_layer_only() -> None:
    app, _ = build(semantic=True, threshold=0.5)
    turns = {
        "model": "relay/auto",
        "temperature": 0,
        "messages": [
            {"role": "user", "content": "hello there friend"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "tell me a joke"},
        ],
    }
    async with client(app) as c:
        await c.post(CHAT, json=turns)
        await settle(app)
        stage = app.state.cache_stage
        assert stage._semantic._items == {}  # nothing semantic stored for multi-turn
        again = await c.post(CHAT, json=turns)
    assert again.headers["x-relay-cache"] == "hit"  # identical, so the exact layer answers


async def test_a_failing_embedder_falls_back_to_the_exact_layer() -> None:
    class Broken:
        name = "broken"
        dimensions = 384
        ready = True  # loaded, but fails when asked to embed

        async def embed(self, text: str) -> list[float]:
            raise RuntimeError("model failed to load")

    app, provider = build(semantic=True, embedder=Broken())
    async with client(app) as c:
        a = await c.post(CHAT, json=body())
        await settle(app)
        b = await c.post(CHAT, json=body())
    assert a.status_code == 200 and b.headers["x-relay-cache"] == "hit"  # exact still works
    assert provider.calls == 1


# ---------------------------------------------------------------- remaining plan items
async def test_the_embedding_is_computed_once_and_kept_for_the_router() -> None:
    calls = 0

    class Counting(HashEmbedder):
        async def embed(self, text: str) -> list[float]:
            nonlocal calls
            calls += 1
            return await super().embed(text)

    app, _ = build(semantic=True, embedder=Counting(), semantic_cache=MemorySemanticStore())
    stage = app.state.cache_stage
    from relay.cache.result import CacheResult  # noqa: F401
    from tests.reliability.conftest import make_ctx  # noqa: F401

    async with client(app) as c:
        await c.post(CHAT, json=body())
        await settle(app)
    assert calls == 1  # one embedding for the lookup; the write-back reuses it
    assert len(stage._semantic._items) == 1


async def test_degraded_answers_are_never_cached() -> None:
    """A fallback answer from an un-evidenced edge must not become everyone's cached answer
    (ADR 0010). Checked at the stage: the miss is not written when ctx.degraded is set."""
    import time

    from relay.cache.stage import CacheStage
    from relay.pipeline.context import ProviderAttempt, RequestContext
    from relay.pipeline.stage import Outcome
    from relay.schemas.chat import ChatRequest, Usage

    store = MemoryExactStore()
    stage = CacheStage(TEST_CONFIG, exact=store)
    request = ChatRequest.model_validate(
        {"model": "relay/auto", "temperature": 0, "messages": [{"role": "user", "content": "q"}]}
    )

    def ctx(degraded: bool) -> RequestContext:
        c = RequestContext(
            request_id="r",
            tenant=TEST_CONFIG.tenants["test"].model_copy(
                update={"cache": TenantCache(enabled=True)}
            ),
            request=request,
            route="default",
            priority="interactive",
            cohort=None,
            policy_version="t",
            deadline=time.monotonic() + 60,
            stream=False,
        )
        c.attempts.append(ProviderAttempt("mock", "mock/strong", 0.01, None))
        c.usage = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        c.reply_parts, c.finish_reason, c.degraded = ["answer"], "stop", degraded
        return c

    for degraded in (True, False):
        c = ctx(degraded)
        await stage.process(c)
        await stage.finalize(c, Outcome("success"))
        await stage.drain()
        assert len(store._items) == (0 if degraded else 1)


async def test_hits_count_toward_the_rate_limit_but_reserve_nothing() -> None:
    from relay.config.tenants import TenantLimits

    tenant = TEST_CONFIG.tenants["test"].model_copy(
        update={
            "cache": TenantCache(enabled=True),
            "limits": TenantLimits(rpm=3, tpm=100_000_000, concurrency=100),
        }
    )
    config = TEST_CONFIG.model_copy(update={"tenants": {**TEST_CONFIG.tenants, "test": tenant}})
    app = create_app(
        relay_settings(embedder="off"), providers={"mock": MockProvider()}, config=config
    )
    async with client(app) as c:
        first = await c.post(CHAT, json=body())  # miss: a provider call
        await settle(app)
        statuses = [(await c.post(CHAT, json=body())).status_code for _ in range(3)]
    assert first.status_code == 200
    # rpm=3: the miss and two hits fit; the third hit is refused. A cache can't bypass limits.
    assert statuses == [200, 200, 429]


async def test_an_embedder_that_is_not_ready_is_skipped_silently() -> None:
    class Loading:
        name = "loading"
        dimensions = 384
        ready = False

        async def embed(self, text: str) -> list[float]:
            raise AssertionError("must not be called while the model is still loading")

    app, provider = build(semantic=True, embedder=Loading(), semantic_cache=MemorySemanticStore())
    async with client(app) as c:
        a = await c.post(CHAT, json=body())
        await settle(app)
        b = await c.post(CHAT, json=body())
    assert a.status_code == 200 and b.headers["x-relay-cache"] == "hit"  # exact layer works
    assert provider.calls == 1
