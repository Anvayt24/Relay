"""The exact-match layer: a key-value cache in Redis (PRD FR-6).

The first thing a cacheable request checks, because it is a single O(1) lookup with no model
involved: identical requests (same tenant, alias, messages and parameters) hit it.

**Fail-open.** A cache is an optimization. If Redis is down, the layer reports
``CacheUnavailable`` and the stage serves the request from the provider as if the cache
weren't there (degradation matrix: "exact cache bypassed (fail-open)"). It must never turn a
Redis outage into request failures, which is the opposite of budgets (those fail closed,
because overspending is worse than being unavailable).

Entries expire by Redis TTL (per tenant). A value is limited in size, so one huge answer can't
fill the cache.
"""

import time
from collections.abc import Callable
from typing import Protocol

import redis.asyncio as redis

from relay.cache.entry import CachedReply

MAX_VALUE_BYTES = 256 * 1024
KEY_PREFIX = "relay:cache:x:"


class CacheUnavailable(Exception):
    """The cache backend can't be reached right now. Callers treat this as a miss."""


class ExactStore(Protocol):
    async def get(self, key: str) -> CachedReply | None: ...
    async def set(self, key: str, reply: CachedReply, ttl_s: int) -> None: ...
    async def start(self) -> None: ...
    async def close(self) -> None: ...


class MemoryExactStore:
    """In-process, for tests and for running without Redis."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._items: dict[str, tuple[float, CachedReply]] = {}

    async def get(self, key: str) -> CachedReply | None:
        item = self._items.get(key)
        if item is None or item[0] <= self._clock():
            self._items.pop(key, None)
            return None
        return item[1]

    async def set(self, key: str, reply: CachedReply, ttl_s: int) -> None:
        if len(reply.dumps().encode("utf-8")) <= MAX_VALUE_BYTES:
            self._items[key] = (self._clock() + ttl_s, reply)

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None


class RedisExactStore:
    def __init__(self, url: str, *, timeout_s: float = 0.3) -> None:
        # Short timeouts: a slow Redis must cost a cache lookup at most this long, not stall
        # the request. Decode off: values are stored as UTF-8 JSON bytes.
        self._client = redis.from_url(
            url, socket_timeout=timeout_s, socket_connect_timeout=timeout_s
        )

    async def get(self, key: str) -> CachedReply | None:
        try:
            blob = await self._client.get(KEY_PREFIX + key)
        except (redis.RedisError, OSError, TimeoutError) as e:
            raise CacheUnavailable(str(e)) from e
        return None if blob is None else CachedReply.loads(blob)

    async def set(self, key: str, reply: CachedReply, ttl_s: int) -> None:
        blob = reply.dumps().encode("utf-8")
        if len(blob) > MAX_VALUE_BYTES:
            return
        try:
            await self._client.set(KEY_PREFIX + key, blob, ex=ttl_s)
        except (redis.RedisError, OSError, TimeoutError) as e:
            raise CacheUnavailable(str(e)) from e

    async def start(self) -> None:
        return None  # connects lazily, on first use: Relay must start with Redis down

    async def close(self) -> None:
        await self._client.aclose()
