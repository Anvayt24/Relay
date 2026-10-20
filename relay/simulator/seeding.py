"""Deterministic randomness: every random draw in the simulator comes from this module.

PRD §11.1: "each request's outcome is seeded by (seed, request index)". This module is what
makes that true, and the design has three deliberate choices:

1. **A fresh generator per (seed, index, model, purpose)**, instead of one shared generator.
   With a shared generator, request 7's draws depend on how many draws requests 0-6 made, so
   any change upstream (one more error, one longer reply) reshuffles everything after it.
   Keyed generators make each request's outcome a pure function of its keys.

2. **One substream per purpose** (``"latency"``, ``"error"``, ``"length"``, ``"quality"``...).
   Raising the error rate then changes *only* the error draws: latency and correctness stay
   identical request by request. This is the *common random numbers* technique. Comparing
   scenario A against scenario B on the same draws removes most of the noise from the
   difference, so fewer runs are needed to see a real effect.

3. **blake2b, not ``hash()``**. Python salts ``str`` hashes per process (``PYTHONHASHSEED``)
   to resist hash-flooding attacks, so ``hash("latency")`` differs between runs. A
   cryptographic hash of the keys' ``repr`` is stable everywhere: across processes, machines
   and Python versions.
"""

import hashlib
import random


def _hash64(seed: int, keys: tuple[str | int, ...]) -> int:
    # repr() of the whole tuple, not a join of the parts: ("ab", "c") and ("a", "bc"), or 1
    # and "1", must not collide.
    material = repr((seed, *keys)).encode()
    return int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big")


def rng_for(seed: int, *keys: str | int) -> random.Random:
    """A generator for purposes that need *several* draws (a lognormal and a jitter, a reply's
    words). Seeding a Mersenne Twister costs ~25 µs, so single draws use ``uniform_for``."""
    return random.Random(_hash64(seed, keys))


def uniform_for(seed: int, *keys: str | int) -> float:
    """One uniform draw in [0, 1), straight from the hash (~4 µs, no generator).

    A good hash's output bits are already uniformly distributed, so for a single draw a full
    PRNG adds nothing but cost. This matters for the evidence simulations (Phase 12), which
    draw correctness and judge verdicts millions of times. The top 53 bits are used because
    a float's mantissa holds exactly 53, which is what ``random.random()`` does too.
    """
    return (_hash64(seed, keys) >> 11) / (1 << 53)
