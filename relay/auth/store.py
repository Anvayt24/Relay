"""Where key hashes live, and the cache that keeps auth working when that place is down.

``KeyStore.lookup(key_hash)`` returns the tenant id for an active key, ``None`` for an
unknown or revoked one, and raises ``KeyStoreUnavailable`` when it cannot tell.

**The difference between "no" and "can't tell" is the point.**
- *No* means 401: the key is wrong.
- *Can't tell* (Postgres down) means 503 ``auth_unavailable``: the key may well be valid,
  and telling a paying client its good key is invalid would be both wrong and alarming.

``CachedKeyStore`` handles that degradation (PRD §8, "Postgres down: auth uses a TTL in-memory
key cache; keys not yet cached fail until recovery"):

- **Fresh positive entries** (≤ 5 min) are served without a query, which takes load off
  Postgres. The cost: revoking a key takes up to 5 minutes to bite on each replica.
- **During an outage,** a positive entry up to 1 hour old is still honored (stale-if-error):
  known clients keep working through a short outage.
- **Negative entries** (unknown keys, 10 s) stop a client with a bad key, or someone spraying
  guesses, from turning every request into a database query.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import asyncpg


class KeyStoreUnavailable(Exception):
    """The store could not answer (as opposed to answering "no such key")."""


class KeyStore(Protocol):
    async def lookup(self, key_hash: bytes) -> str | None: ...
    async def start(self) -> None: ...
    async def close(self) -> None: ...


class MemoryKeyStore:
    """For tests and for the CLI's dry runs: hash -> tenant id."""

    def __init__(self, keys: dict[bytes, str] | None = None) -> None:
        self.keys = dict(keys or {})
        self.down = False  # tests flip this to simulate an outage
        self.lookups = 0

    async def lookup(self, key_hash: bytes) -> str | None:
        self.lookups += 1
        if self.down:
            raise KeyStoreUnavailable("memory store marked down")
        return self.keys.get(key_hash)

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None


LOOKUP_TIMEOUT_S = 3.0


class PostgresKeyStore:
    QUERY = "SELECT tenant FROM api_keys WHERE key_hash = $1 AND status = 'active'"

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def start(self) -> None:
        # min_size=0, as for the log sink: creating the pool connects to nothing, so Relay
        # starts even when Postgres is down.
        self._pool = await asyncpg.create_pool(
            self._dsn, min_size=0, max_size=4, timeout=2.0, command_timeout=2.0
        )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()

    async def lookup(self, key_hash: bytes) -> str | None:
        if self._pool is None:
            raise KeyStoreUnavailable("key store not started")
        try:
            # Bounded as a whole: waiting for one of the pool's 4 connections has no timeout of
            # its own, and command_timeout only starts once a connection is held.
            async with asyncio.timeout(LOOKUP_TIMEOUT_S):
                tenant: str | None = await self._pool.fetchval(self.QUERY, key_hash)
        except (OSError, asyncpg.PostgresError, asyncpg.InterfaceError, TimeoutError) as exc:
            # InterfaceError (a pool closing, a connection in use) is not a PostgresError:
            # uncaught, it became a 500 instead of 503 auth_unavailable (found in review).
            raise KeyStoreUnavailable(repr(exc)) from exc
        return tenant


@dataclass
class _Entry:
    tenant: str | None
    at: float


class CachedKeyStore:
    def __init__(
        self,
        inner: KeyStore,
        *,
        positive_ttl_s: float = 300.0,
        stale_if_error_s: float = 3600.0,
        negative_ttl_s: float = 10.0,
        max_entries: int = 100_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.inner = inner
        self._positive_ttl = positive_ttl_s
        self._stale_if_error = stale_if_error_s
        self._negative_ttl = negative_ttl_s
        self._max_entries = max_entries
        self._clock = clock
        self._entries: dict[bytes, _Entry] = {}

    async def start(self) -> None:
        await self.inner.start()

    async def close(self) -> None:
        await self.inner.close()

    async def lookup(self, key_hash: bytes) -> str | None:
        now = self._clock()
        entry = self._entries.get(key_hash)
        if entry is not None:
            ttl = self._positive_ttl if entry.tenant is not None else self._negative_ttl
            if now - entry.at < ttl:
                return entry.tenant
        try:
            tenant = await self.inner.lookup(key_hash)
        except KeyStoreUnavailable:
            if (
                entry is not None
                and entry.tenant is not None
                and now - entry.at < self._stale_if_error
            ):
                return entry.tenant  # stale, but this key was valid moments ago
            raise
        if len(self._entries) >= self._max_entries:
            # Bound on memory. Unknown keys go first: a client spraying random keys fills the
            # cache with them, and clearing everything would also drop the valid keys that let
            # clients through a Postgres outage (stale_if_error).
            self._entries = {k: e for k, e in self._entries.items() if e.tenant is not None}
            if len(self._entries) >= self._max_entries:
                self._entries.clear()
        self._entries[key_hash] = _Entry(tenant, now)
        return tenant
