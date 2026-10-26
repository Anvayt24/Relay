"""The exact and semantic stores: one contract, run against the in-memory implementations
always and against Redis / Postgres+pgvector under ``-m integration``."""

import uuid
from collections.abc import AsyncIterator

import asyncpg
import pytest

from relay.cache.embed import HashEmbedder, cosine
from relay.cache.entry import CachedReply
from relay.cache.exact import CacheUnavailable, MemoryExactStore, RedisExactStore
from relay.cache.semantic import MemorySemanticStore, PgVectorStore, SemanticStore
from relay.settings import Settings
from relay.telemetry.migrate import migrate


def reply(text: str = "answer", tokens: int = 5) -> CachedReply:
    return CachedReply(
        text=text,
        finish_reason="stop",
        prompt_tokens=3,
        completion_tokens=tokens,
        cached_tokens=0,
        served_model="mock/strong",
        policy_version="static-v0",
        created_at=0.0,
    )


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# ---------------------------------------------------------------- exact
async def test_memory_exact_store_round_trip_and_ttl() -> None:
    clock = Clock()
    store = MemoryExactStore(clock)
    await store.set("k", reply(), ttl_s=60)
    assert await store.get("k") == reply()
    assert await store.get("missing") is None
    clock.now += 61
    assert await store.get("k") is None  # expired


async def test_memory_exact_store_refuses_huge_values() -> None:
    store = MemoryExactStore()
    await store.set("big", reply("x" * (300 * 1024)), ttl_s=60)
    assert await store.get("big") is None


@pytest.mark.integration
async def test_redis_exact_store_matches_the_memory_contract() -> None:
    url = Settings().redis_url
    if not url:
        pytest.fail("RELAY_REDIS_URL must be set for integration tests")
    store = RedisExactStore(url)
    key = f"test-{uuid.uuid4().hex}"
    try:
        assert await store.get(key) is None
        await store.set(key, reply("from redis"), ttl_s=30)
        assert await store.get(key) == reply("from redis")
    finally:
        await store.close()


async def test_redis_exact_store_fails_open_as_unavailable() -> None:
    store = RedisExactStore("redis://127.0.0.1:1/0", timeout_s=0.1)  # nothing listens there
    try:
        with pytest.raises(CacheUnavailable):
            await store.get("k")
        with pytest.raises(CacheUnavailable):
            await store.set("k", reply(), ttl_s=10)
    finally:
        await store.close()


# ---------------------------------------------------------------- semantic
async def vec(text: str) -> list[float]:
    return await HashEmbedder().embed(text)


async def contract(store: SemanticStore) -> None:
    ns = f"ns-{uuid.uuid4().hex}"
    other = f"ns-{uuid.uuid4().hex}"
    await store.insert(
        ns, await vec("capital of france today"), reply("Paris"), guard="g", ttl_s=60
    )
    await store.insert(
        other, await vec("capital of france today"), reply("other"), guard="g", ttl_s=60
    )

    near = await vec("capital of france today please")
    hit = await store.lookup(ns, near, threshold=0.7, max_tokens=100, guard="g")
    assert hit is not None and hit.reply.text == "Paris"  # its own namespace's entry only
    assert hit.similarity == pytest.approx(
        cosine(near, await vec("capital of france today")), abs=1e-3
    )
    # Below the threshold: a miss.
    assert await store.lookup(ns, await vec("sourdough bread recipe"), threshold=0.7,
                              max_tokens=100, guard="g") is None  # fmt: skip
    # The length guard: a 5-token reply may not answer a request capped at 3.
    assert await store.lookup(ns, near, threshold=0.7, max_tokens=3, guard="g") is None
    # The lexical guard: a different numbers/names signature is a miss; no guard means hit.
    assert await store.lookup(ns, near, threshold=0.7, max_tokens=100, guard="DIFFERENT") is None
    assert await store.lookup(ns, near, threshold=0.7, max_tokens=100, guard=None) is not None
    # An empty namespace never matches.
    assert (
        await store.lookup("nothing-here", near, threshold=0.0, max_tokens=100, guard=None) is None
    )


async def test_memory_semantic_store_contract() -> None:
    await contract(MemorySemanticStore())


async def test_memory_semantic_store_entries_expire() -> None:
    clock = Clock()
    store = MemorySemanticStore(clock)
    v = await vec("hello world")
    await store.insert("ns", v, reply(), guard="g", ttl_s=10)
    assert await store.lookup("ns", v, threshold=0.9, max_tokens=100, guard="g") is not None
    clock.now += 11
    assert await store.lookup("ns", v, threshold=0.9, max_tokens=100, guard="g") is None


@pytest.fixture
async def pg_store() -> AsyncIterator[PgVectorStore]:
    dsn = Settings().database_url
    if not dsn:
        pytest.fail("RELAY_DATABASE_URL must be set for integration tests")
    schema = f"test_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(dsn)
    await admin.execute(f"CREATE SCHEMA {schema}")
    # The vector type lives in `public`; the throwaway schema's search_path must reach it.
    conn = await asyncpg.connect(dsn, server_settings={"search_path": f"{schema},public"})
    store = PgVectorStore(f"{dsn}?options=-csearch_path%3D{schema}%2Cpublic")
    try:
        await migrate(conn)
        await store.start()
        yield store
    finally:
        await store.close()
        await conn.close()
        await admin.execute(f"DROP SCHEMA {schema} CASCADE")
        await admin.close()


@pytest.mark.integration
async def test_pgvector_store_matches_the_memory_contract(pg_store: PgVectorStore) -> None:
    await contract(pg_store)


@pytest.mark.integration
async def test_pgvector_expired_entries_are_not_returned(pg_store: PgVectorStore) -> None:
    v = await vec("hello world")
    await pg_store.insert("ns", v, reply(), guard="g", ttl_s=0)  # already expired
    assert await pg_store.lookup("ns", v, threshold=0.5, max_tokens=100, guard="g") is None
