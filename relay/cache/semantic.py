"""The semantic layer: find a stored prompt that is *close enough* in meaning.

An exact cache only helps when the same words are repeated. The semantic layer also catches
paraphrases ("what's the capital of France" / "France's capital city?"), by comparing prompt
embeddings. The price is a risk the exact cache doesn't have: two prompts can be close in
embedding space and still need *different* answers ("15% of 200" / "15% of 300"). So:

- A hit needs cosine similarity >= the route's threshold, which is **measured** to reach
  >= 99% precision on a labeled set (`make eval-cache`), not picked by feel.
- A hit is also rejected when the stored entry's completion is longer than the request's own
  ``max_tokens`` (``keys.fits``), and, if the lexical guard is on, when the numbers and names
  in the two prompts differ.
- Entries live in a **namespace** (tenant, alias, system prompt, parameters, ``keys.py``):
  a lookup only ever sees its own namespace's entries.

Two implementations behind one protocol: ``PgVectorStore`` (Postgres + pgvector, HNSW index)
for the running gateway, and ``MemorySemanticStore`` (brute force) for tests and the
evaluation, which share the exact matching rules.

**Two things keep the Postgres lookup honest over time** (both found in review):
- *Iterative index scans.* HNSW finds the ``ef_search`` nearest vectors first and filters them
  afterwards. Rows of other namespaces (or expired ones) can fill all of those places, and the
  lookup then returns nothing although a match exists. ``hnsw.iterative_scan`` (pgvector 0.8)
  makes the scan continue until the filter is satisfied.
- *An expiry sweep.* Expired rows are filtered out of every lookup but were never deleted, so
  the table, the index and the false misses above grew with traffic. A background task now
  deletes them in batches.
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import asyncpg
import numpy as np

from relay.cache.entry import CachedReply
from relay.cache.exact import CacheUnavailable
from relay.cache.keys import fits

logger = logging.getLogger(__name__)

COMMAND_TIMEOUT_S = 0.5
SWEEP_INTERVAL_S = 300.0
SWEEP_BATCH = 1000


@dataclass(frozen=True)
class SemanticHit:
    reply: CachedReply
    similarity: float


class SemanticStore(Protocol):
    async def lookup(
        self,
        namespace: str,
        embedding: list[float],
        *,
        threshold: float,
        max_tokens: int,
        guard: str | None,
    ) -> SemanticHit | None: ...

    async def insert(
        self,
        namespace: str,
        embedding: list[float],
        reply: CachedReply,
        *,
        guard: str,
        ttl_s: int,
    ) -> None: ...

    async def start(self) -> None: ...
    async def close(self) -> None: ...


def _accept(
    similarity: float,
    load_reply: Callable[[], CachedReply],
    stored_guard: str,
    *,
    threshold: float,
    max_tokens: int,
    guard: str | None,
) -> SemanticHit | None:
    """The accept rule both stores share, in this order: similarity >= threshold, then the
    stored completion fits ``max_tokens``, then the lexical guard. The reply is loaded only
    once the threshold passes (pgvector stores it serialized)."""
    if similarity < threshold:
        return None
    reply = load_reply()
    if not fits(reply.completion_tokens, max_tokens):
        return None
    if guard is not None and guard != stored_guard:
        return None  # lexical guard: the numbers or names differ
    return SemanticHit(reply, float(similarity))


@dataclass
class _Entry:
    embedding: np.ndarray
    reply: CachedReply
    guard: str
    expires_at: float


class MemorySemanticStore:
    """Brute-force cosine over a namespace. Exact (no approximation), so it is also the
    reference the pgvector store's results are compared with.

    Bounded, because a lookup is a loop over the namespace on the event loop: an insert drops
    its namespace's expired entries, and past ``max_entries`` every expired entry and then the
    oldest ones go (found in review: nothing was ever removed, so memory and per-lookup CPU
    grew with traffic)."""

    def __init__(
        self, clock: Callable[[], float] = time.monotonic, *, max_entries: int = 10_000
    ) -> None:
        self._clock = clock
        self._max_entries = max_entries
        self._items: dict[str, list[_Entry]] = {}

    async def lookup(
        self,
        namespace: str,
        embedding: list[float],
        *,
        threshold: float,
        max_tokens: int,
        guard: str | None,
    ) -> SemanticHit | None:
        now = self._clock()
        query = np.asarray(embedding, dtype=np.float32)
        best: tuple[float, _Entry] | None = None
        for entry in self._items.get(namespace, []):
            if entry.expires_at <= now:
                continue
            sim = float(query @ entry.embedding)
            if best is None or sim > best[0]:
                best = (sim, entry)
        if best is None:
            return None
        sim, entry = best
        return _accept(
            sim,
            lambda: entry.reply,
            entry.guard,
            threshold=threshold,
            max_tokens=max_tokens,
            guard=guard,
        )

    async def insert(
        self,
        namespace: str,
        embedding: list[float],
        reply: CachedReply,
        *,
        guard: str,
        ttl_s: int,
    ) -> None:
        now = self._clock()
        live = [e for e in self._items.get(namespace, []) if e.expires_at > now]
        live.append(_Entry(np.asarray(embedding, dtype=np.float32), reply, guard, now + ttl_s))
        self._items[namespace] = live
        if sum(len(entries) for entries in self._items.values()) > self._max_entries:
            self._evict(now)

    def _evict(self, now: float) -> None:
        """Drop every expired entry; if still too many, the oldest ones (by expiry)."""
        everything = [
            (e.expires_at, ns, e)
            for ns, entries in self._items.items()
            for e in entries
            if e.expires_at > now
        ]
        everything.sort(key=lambda t: t[0])
        keep = everything[len(everything) - self._max_entries :] if everything else []
        self._items = {}
        for _, ns, e in keep:
            self._items.setdefault(ns, []).append(e)

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None


class PgVectorStore:
    """Postgres + pgvector. Cosine distance is ``<=>``; similarity = ``1 - distance``."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None
        self._sweeper: asyncio.Task[None] | None = None

    async def start(self) -> None:
        # min_size=0: nothing connects at startup, so Relay starts with Postgres down; the
        # first lookup that fails just disables the semantic layer for that request.
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=0,
            max_size=4,
            command_timeout=COMMAND_TIMEOUT_S,
            init=_init_connection,
        )
        self._sweeper = asyncio.create_task(self._sweep_forever(), name="relay-cache-sweep")

    async def close(self) -> None:
        if self._sweeper is not None:
            self._sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sweeper
        if self._pool is not None:
            await self._pool.close()

    async def sweep(self) -> int:
        """Delete expired entries, in batches so no single statement holds locks for long.
        Every replica sweeps; the deletes are idempotent, so that is harmless."""
        pool = self._need_pool()
        deleted = 0
        async with pool.acquire() as conn:
            while True:
                status = await conn.execute(
                    "DELETE FROM cache_entries WHERE id IN (SELECT id FROM cache_entries"
                    " WHERE expires_at <= now() LIMIT $1)",
                    SWEEP_BATCH,
                    timeout=30,
                )
                n = int(status.split()[-1])  # "DELETE <n>"
                deleted += n
                if n < SWEEP_BATCH:
                    return deleted

    async def _sweep_forever(self) -> None:
        while True:
            await asyncio.sleep(SWEEP_INTERVAL_S)
            try:
                await self.sweep()
            except Exception as e:  # Postgres down: the next sweep tries again
                logger.warning("semantic cache sweep failed: %r", e)

    async def lookup(
        self,
        namespace: str,
        embedding: list[float],
        *,
        threshold: float,
        max_tokens: int,
        guard: str | None,
    ) -> SemanticHit | None:
        pool = self._need_pool()
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT 1 - (embedding <=> $1) AS similarity, reply, guard"
                    " FROM cache_entries WHERE namespace = $2 AND expires_at > now()"
                    " ORDER BY embedding <=> $1 LIMIT 1",
                    np.asarray(embedding, dtype=np.float32),
                    namespace,
                )
        except (asyncpg.PostgresError, OSError, TimeoutError) as e:
            raise CacheUnavailable(str(e)) from e
        if row is None:
            return None
        return _accept(
            row["similarity"],
            lambda: CachedReply.loads(row["reply"]),
            row["guard"],
            threshold=threshold,
            max_tokens=max_tokens,
            guard=guard,
        )

    async def insert(
        self,
        namespace: str,
        embedding: list[float],
        reply: CachedReply,
        *,
        guard: str,
        ttl_s: int,
    ) -> None:
        pool = self._need_pool()
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO cache_entries (namespace, embedding, reply, guard,"
                    " served_model, policy_version, expires_at)"
                    " VALUES ($1, $2, $3, $4, $5, $6, now() + make_interval(secs => $7))",
                    namespace,
                    np.asarray(embedding, dtype=np.float32),
                    reply.dumps(),
                    guard,
                    reply.served_model,
                    reply.policy_version,
                    float(ttl_s),
                )
        except (asyncpg.PostgresError, OSError, TimeoutError) as e:
            raise CacheUnavailable(str(e)) from e

    def _need_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise CacheUnavailable("semantic store not started")
        return self._pool


async def _init_connection(conn: asyncpg.Connection) -> None:
    from pgvector.asyncpg import register_vector

    await register_vector(conn)
    # Keep scanning the index until the namespace filter finds its nearest row, instead of
    # filtering a fixed ef_search candidates and possibly returning none (see the docstring).
    # strict_order keeps the results exactly ordered, which LIMIT 1 relies on.
    await conn.execute("SET hnsw.iterative_scan = strict_order")
