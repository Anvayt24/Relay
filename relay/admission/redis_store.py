"""``CapacityStore`` on Redis, shared by every replica. Each operation is one Lua script.

**Why Lua.** "Check the budget, then reserve" as two commands is a race: two replicas both
read "$0.10 left", both reserve $0.08, and the budget is overspent. Redis runs a Lua script
atomically, with no other command in between, so the check and the write happen as one
step for the whole fleet. That is what makes FR-4's invariant hold across replicas, not just
within one process.

**Why Redis's clock** (``redis.call('TIME')``) inside the scripts: replicas' clocks drift. A
bucket refilled by one replica's clock and drained by another's would leak or lose tokens.
One clock, Redis's, for everything time-based.

**Details:**
- Scripts return strings when the value is fractional: Redis converts a Lua number reply to an
  *integer*, so ``0.25`` seconds would come back as ``0``.
- Money in Lua is a double, exact up to 2^53, i.e. about $9M in nano-USD per tenant-day.
- Budget keys share a hash tag, ``{tenant:day}``, so all of a budget's keys land on one shard
  if this ever runs on Redis Cluster (a multi-key script requires that).
- Every key gets an expiry, so abandoned state cleans itself up.
"""

from typing import Any

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from relay.admission.store import BudgetState, StoreUnavailable

_NOW_S = "local _t = redis.call('TIME') local now = tonumber(_t[1]) + tonumber(_t[2]) / 1e6\n"
_TWO_DAYS = 172800

TAKE = (
    _NOW_S
    + """
-- KEYS[1] bucket; ARGV capacity, refill_per_s, amount
local capacity, rate, amount = tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3])
local b = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(b[1]) or capacity        -- a new bucket starts full
local ts = tonumber(b[2]) or now
tokens = math.min(capacity, tokens + math.max(0, now - ts) * rate)   -- lazy refill
local wait = 0
if tokens >= amount then tokens = tokens - amount else wait = (amount - tokens) / rate end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', tostring(now))
redis.call('EXPIRE', KEYS[1], math.ceil(capacity / rate) + 60)
return tostring(wait)
"""
)

REFUND = """
-- KEYS[1] bucket; ARGV capacity, amount
local tokens = tonumber(redis.call('HGET', KEYS[1], 'tokens'))
if tokens then
  local refilled = math.min(tonumber(ARGV[1]), tokens + tonumber(ARGV[2]))
  redis.call('HSET', KEYS[1], 'tokens', tostring(refilled))
end
return 1
"""

_SWEEP = """
local now_ms = math.floor(now * 1000)
for _, id in ipairs(redis.call('ZRANGEBYSCORE', KEYS[2], '-inf', now_ms)) do
  redis.call('HDEL', KEYS[3], id)        -- expired: its owner died or hung; free the money
  redis.call('ZREM', KEYS[2], id)
end
local committed = tonumber(redis.call('GET', KEYS[1]) or '0')
local outstanding = 0
for _, v in ipairs(redis.call('HVALS', KEYS[3])) do outstanding = outstanding + tonumber(v) end
"""

RESERVE = (
    _NOW_S
    + _SWEEP
    + """
-- KEYS committed, expiries (zset), amounts (hash); ARGV res_id, amount, limit, ttl_ms
local amount = tonumber(ARGV[2])
if committed + outstanding + amount > tonumber(ARGV[3]) then return 0 end
redis.call('ZADD', KEYS[2], now_ms + tonumber(ARGV[4]), ARGV[1])
redis.call('HSET', KEYS[3], ARGV[1], ARGV[2])
for i = 1, 3 do redis.call('EXPIRE', KEYS[i], TWO_DAYS) end
return 1
""".replace("TWO_DAYS", str(_TWO_DAYS))
)

SETTLE = """
-- KEYS committed, expiries, amounts, settled (set); ARGV res_id, actual
if redis.call('SADD', KEYS[4], ARGV[1]) == 0 then return 0 end   -- settled before: no-op
redis.call('HDEL', KEYS[3], ARGV[1])
redis.call('ZREM', KEYS[2], ARGV[1])
redis.call('INCRBY', KEYS[1], ARGV[2])
for i = 1, 4 do redis.call('EXPIRE', KEYS[i], TWO_DAYS) end
return 1
""".replace("TWO_DAYS", str(_TWO_DAYS))

STATE = (
    _NOW_S
    + _SWEEP
    + """
return {tostring(committed), tostring(outstanding)}
"""
)

PAUSE = """
-- KEYS[1] pause flag; ARGV ttl_ms. Only ever extends: a short pause can't cut a long one.
local ms = tonumber(ARGV[1])
if redis.call('PTTL', KEYS[1]) < ms then redis.call('SET', KEYS[1], '1', 'PX', ms) end
return 1
"""


class RedisStore:
    def __init__(self, url: str, *, prefix: str = "relay:", timeout_s: float = 0.5) -> None:
        # Short timeouts: admission is on the request path. A slow Redis must turn into
        # "unavailable" (and the degraded mode) quickly, not into slow requests.
        self._redis: Any = aioredis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=timeout_s,
            socket_timeout=timeout_s,
        )
        self._prefix = prefix
        self._take = self._redis.register_script(TAKE)
        self._refund = self._redis.register_script(REFUND)
        self._reserve = self._redis.register_script(RESERVE)
        self._settle = self._redis.register_script(SETTLE)
        self._state = self._redis.register_script(STATE)
        self._pause = self._redis.register_script(PAUSE)

    async def start(self) -> None:
        return None  # the pool connects lazily, so startup never needs Redis

    async def close(self) -> None:
        await self._redis.aclose()

    def _k(self, key: str) -> str:
        return self._prefix + key

    def _budget_keys(self, budget: str) -> list[str]:
        tag = f"{self._prefix}budget:{{{budget}}}"  # {tenant:day}: one hash slot
        return [f"{tag}:committed", f"{tag}:expiries", f"{tag}:amounts", f"{tag}:settled"]

    async def _run(self, script: Any, keys: list[str], args: list[Any]) -> Any:
        try:
            return await script(keys=keys, args=args)
        except (RedisError, OSError) as exc:
            raise StoreUnavailable(repr(exc)) from exc

    async def take(self, key: str, *, capacity: int, refill_per_s: float, amount: int) -> float:
        return float(await self._run(self._take, [self._k(key)], [capacity, refill_per_s, amount]))

    async def refund(self, key: str, *, capacity: int, amount: int) -> None:
        await self._run(self._refund, [self._k(key)], [capacity, amount])

    async def reserve(
        self, budget: str, res_id: str, *, amount: int, limit: int, ttl_s: float
    ) -> bool:
        keys = self._budget_keys(budget)[:3]
        return bool(
            await self._run(self._reserve, keys, [res_id, amount, limit, round(ttl_s * 1000)])
        )

    async def settle(self, budget: str, res_id: str, *, actual: int) -> bool:
        return bool(await self._run(self._settle, self._budget_keys(budget), [res_id, actual]))

    async def budget_state(self, budget: str) -> BudgetState:
        committed, outstanding = await self._run(self._state, self._budget_keys(budget)[:3], [])
        return BudgetState(int(float(committed)), int(float(outstanding)))

    async def pause(self, key: str, ttl_s: float) -> None:
        await self._run(self._pause, [self._k(key)], [max(1, round(ttl_s * 1000))])

    async def paused_for(self, key: str) -> float:
        try:
            pttl = int(await self._redis.pttl(self._k(key)))
        except (RedisError, OSError) as exc:
            raise StoreUnavailable(repr(exc)) from exc
        return pttl / 1000 if pttl > 0 else 0.0

    async def ping(self) -> bool:
        try:
            return bool(await self._redis.ping())
        except (RedisError, OSError):
            return False
