"""The semantic layer: find a stored prompt that is *close enough* in meaning (PRD FR-6).

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

Two implementations behind one protocol: ``PgVectorStore`` (Postgres + pgvector, HNSW index;
ADR 0015) for the running gateway, and ``MemorySemanticStore`` (brute force) for tests and the
evaluation, which share the exact matching rules.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import asyncpg
import numpy as np

from relay.cache.entry import CachedReply
from relay.cache.exact import CacheUnavailable
from relay.cache.keys import fits

DIMENSIONS = 384


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


@dataclass
class _Entry:
    embedding: np.ndarray
    reply: CachedReply
    guard: str
    expires_at: float


class MemorySemanticStore:
    """Brute-force cosine over a namespace. Exact (no approximation), so it is also the
    reference the pgvector store's results are compared with."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
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
        if best is None or best[0] < threshold:
            return None
        sim, entry = best
        if not fits(entry.reply.completion_tokens, max_tokens):
            return None
        if guard is not None and guard != entry.guard:
            return None  # lexical guard: the numbers or names differ
        return SemanticHit(entry.reply, sim)

    async def insert(
        self,
        namespace: str,
        embedding: list[float],
        reply: CachedReply,
        *,
        guard: str,
        ttl_s: int,
    ) -> None:
        self._items.setdefault(namespace, []).append(
            _Entry(np.asarray(embedding, dtype=np.float32), reply, guard, self._clock() + ttl_s)
        )

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None


class PgVectorStore:
    """Postgres + pgvector. Cosine distance is ``<=>``; similarity = ``1 - distance``."""

    def __init__(self, dsn: str, *, command_timeout_s: float = 0.5) -> None:
        self._dsn = dsn
        self._timeout = command_timeout_s
        self._pool: asyncpg.Pool | None = None

    async def start(self) -> None:
        # min_size=0: nothing connects at startup, so Relay starts with Postgres down; the
        # first lookup that fails just disables the semantic layer for that request.
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=0,
            max_size=4,
            command_timeout=self._timeout,
            init=_register_vector,
        )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()

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
        if row is None or row["similarity"] < threshold:
            return None
        reply = CachedReply.loads(row["reply"])
        if not fits(reply.completion_tokens, max_tokens):
            return None
        if guard is not None and guard != row["guard"]:
            return None
        return SemanticHit(reply, float(row["similarity"]))

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


async def _register_vector(conn: asyncpg.Connection) -> None:
    from pgvector.asyncpg import register_vector

    await register_vector(conn)
