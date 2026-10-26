"""Exponential backoff with full jitter.

``delay = U(0, min(cap, base * 2**attempt))``

Why *full* jitter, rather than a fixed delay or "exponential plus a little noise": when many
requests fail at the same moment (a provider blip), a fixed delay makes them all come back at
the same moment too, and the recovering provider is hit by a synchronized wave. Drawing the
whole delay uniformly spreads the retries evenly over the window. It is the variant AWS's
analysis ("Exponential Backoff and Jitter") found to do the least total work.
"""

import random


def full_jitter(attempt: int, *, base_s: float, cap_s: float, rng: random.Random) -> float:
    """``attempt`` counts from 0 (the first retry)."""
    return rng.uniform(0.0, min(cap_s, base_s * 2**attempt))
