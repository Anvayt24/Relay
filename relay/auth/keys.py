"""API keys: generation and hashing (PRD FR-3).

A key looks like ``rk_<43 url-safe chars>``: 32 random bytes, 256 bits, from ``secrets``
(the OS's cryptographic RNG; ``random`` is predictable and must never make keys).

**Only a hash is stored: HMAC-SHA256(pepper, key).**
- *Why hash at all:* a leaked database must not leak working keys.
- *Why not bcrypt/argon2:* those are deliberately slow (~100 ms) to resist guessing *low-
  entropy* passwords. A 256-bit random key cannot be guessed, so slowness would only add
  latency to every request.
- *Why HMAC with a pepper, not plain SHA-256:* the pepper is a server secret kept outside the
  database (``RELAY_KEY_PEPPER``). Someone with only a stolen table cannot even test a guess
  without it.
- *Why this makes lookup safe:* we look the hash up in an index, so no per-character string
  comparison of secrets happens, and there is nothing for a timing attack to measure.

The prefix (``rk_`` plus 8 characters) is stored in clear, so a human can tell keys apart in a
list ("revoke rk_3fA9x...") without the key itself.
"""

import hashlib
import hmac
import secrets

KEY_PREFIX = "rk_"
DISPLAY_LEN = 11  # "rk_" + 8 chars


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_key(key: str, pepper: str) -> bytes:
    return hmac.new(pepper.encode(), key.encode(), hashlib.sha256).digest()


def display_prefix(key: str) -> str:
    return key[:DISPLAY_LEN]


def looks_like_key(value: str) -> bool:
    # A cheap shape check before any hashing or lookup: garbage never reaches the database.
    return value.startswith(KEY_PREFIX) and 20 <= len(value) <= 100
