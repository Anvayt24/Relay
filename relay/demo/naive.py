"""What teams write without a gateway: the baseline the demo and the chaos benchmark compare
Relay against.

Up to five tries on the primary with exponential backoff and **no jitter**, then one try on the
fallback. It is generic over how a try is made, so the benchmark (plain HTTP to simulators) and
the demo's chaos panel (in-process simulator) run the very same policy.
"""

import asyncio
from collections.abc import Awaitable, Callable

# 0.5, 1, 2, 4 s between five tries, no jitter: the textbook retry loop.
NAIVE_BACKOFF_S = (0.5, 1.0, 2.0, 4.0)


async def naive_retry[R](
    try_primary: Callable[[], Awaitable[tuple[bool, R]]],
    try_fallback: Callable[[], Awaitable[tuple[bool, R]]],
    *,
    backoff_s: tuple[float, ...] = NAIVE_BACKOFF_S,
) -> tuple[bool, R, str]:
    """(ok, the last try's result, "primary" or "fallback": which one produced it)."""
    for attempt in range(len(backoff_s) + 1):
        ok, result = await try_primary()
        if ok:
            return ok, result, "primary"
        if attempt < len(backoff_s):
            await asyncio.sleep(backoff_s[attempt])
    ok, result = await try_fallback()
    return ok, result, "fallback"
