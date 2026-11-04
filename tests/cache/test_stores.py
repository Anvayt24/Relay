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


@pytest.mark.integration
async def test_pgvector_sweep_deletes_only_expired_entries(pg_store: PgVectorStore) -> None:
    # Found in review: expired rows were filtered out of lookups but never deleted.
    v = await vec("hello world")
    await pg_store.insert("ns", v, reply(), guard="g", ttl_s=0)  # already expired
    await pg_store.insert("ns", v, reply(), guard="g", ttl_s=600)
    assert await pg_store.sweep() == 1
    assert await pg_store.sweep() == 0
    assert await pg_store.lookup("ns", v, threshold=0.5, max_tokens=100, guard="g") is not None


@pytest.mark.integration
async def test_pgvector_connections_scan_the_index_iteratively(pg_store: PgVectorStore) -> None:
    # Found in review: filtered HNSW lookups could miss a namespace's only match, and ADR 0015
    # claimed a setting no code made. Every connection of the store must have it. (A test that
    # makes Postgres pick the HNSW plan on purpose needs thousands of rows and planner flags;
    # this checks the setting itself.)
    pool = pg_store._need_pool()
    async with pool.acquire() as conn:
        assert await conn.fetchval("SHOW hnsw.iterative_scan") == "strict_order"


# ---------------------------------------------------------------- bounded in-process stores
class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def test_memory_exact_store_is_bounded() -> None:
    # Found in review: an entry only left when its own key was read again.
    clock = _Clock()
    store = MemoryExactStore(clock, max_items=3)
    await store.set("old", reply(), ttl_s=10)
    clock.now = 20  # "old" has expired
    for k in ("a", "b", "c", "d"):
        await store.set(k, reply(), ttl_s=600)
    assert len(store._items) == 3
    assert await store.get("a") is None  # the oldest live entry made room
    assert await store.get("d") is not None


async def test_memory_semantic_store_is_bounded() -> None:
    clock = _Clock()
    store = MemorySemanticStore(clock, max_entries=3)
    v = await vec("hello world")
    await store.insert("ns", v, reply("expired"), guard="g", ttl_s=5)
    clock.now = 10
    for i in range(4):
        await store.insert(f"ns{i % 2}", v, reply(f"r{i}"), guard="g", ttl_s=600 + i)
    assert sum(len(e) for e in store._items.values()) == 3
    assert "ns" not in store._items or store._items["ns"] == []
