"""The first-run API key, and the check that a changed pepper is noticed.

On the first start against an empty ``api_keys`` table, Relay mints one key for the bootstrap
tenant and prints it once, so a new operator can send a request without a separate setup
command. Only its HMAC is stored. A restart finds a key and makes none.

Everything happens in one transaction under an advisory lock, so replicas starting together
mint exactly one.

The same look at the table backs the second rule. API keys are hashed with a pepper; if the
pepper changes while keys exist (the data volume holding a generated pepper was lost, or a
different value was set), every existing key silently stops matching. The database therefore
remembers a fingerprint of the pepper its keys were made with, and Relay refuses to start when
the pepper in use does not match it, saying why.
"""

import hashlib
import hmac

import asyncpg

from relay.auth.keys import display_prefix, generate_key, hash_key

# Any fixed 64-bit number works; it only has to be the same in every process.
_LOCK_ID = 727_002
_CHECK_KEY = "pepper_check"


class LostPepperError(RuntimeError):
    pass


def pepper_fingerprint(pepper: str) -> str:
    """An HMAC of a fixed string under the pepper: enough to tell two peppers apart, and
    useless for recovering one."""
    return hmac.new(pepper.encode(), b"relay-pepper-check", hashlib.sha256).hexdigest()


async def ensure_first_key(
    dsn: str, *, pepper: str, tenant: str, pepper_is_new: bool, mint: bool
) -> str | None:
    """The new key, or None if none was needed. Raises LostPepperError, or asyncpg errors when
    the database is unreachable (the caller decides whether that stops startup)."""
    fingerprint = pepper_fingerprint(pepper)
    conn = await asyncpg.connect(dsn, timeout=5)
    try:
        async with conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock($1)", _LOCK_ID)
            existing = await conn.fetchval("SELECT count(*) FROM api_keys")
            recorded = await conn.fetchval(
                "SELECT value FROM instance_settings WHERE key = $1", _CHECK_KEY
            )
            if recorded is not None and recorded != fingerprint and existing:
                raise LostPepperError(
                    f"{existing} API key(s) exist, made with a different pepper than the one "
                    "in use now, so none of them would match. If the data volume holding the "
                    "original pepper (RELAY_DATA_DIR) was lost or replaced, restore it. If you "
                    "set RELAY_KEY_PEPPER, set it back to the original value."
                )
            if recorded is None and existing and pepper_is_new:
                # Keys from before the fingerprint existed, and a pepper made just now: the
                # volume with the real one is gone.
                raise LostPepperError(
                    f"{existing} API key(s) already exist, but this start generated a new key "
                    "pepper, so none of them would match. The data volume holding the "
                    "original pepper (RELAY_DATA_DIR) was lost or replaced. Restore it, or set "
                    "RELAY_KEY_PEPPER to the original value."
                )
            if recorded != fingerprint:
                await conn.execute(
                    "INSERT INTO instance_settings (key, value) VALUES ($1, $2) "
                    "ON CONFLICT (key) DO UPDATE SET value = $2, updated_at = now()",
                    _CHECK_KEY,
                    fingerprint,
                )
            if existing or not mint:
                return None
            key = generate_key()
            await conn.execute(
                "INSERT INTO api_keys (key_hash, key_prefix, tenant) VALUES ($1, $2, $3)",
                hash_key(key, pepper),
                display_prefix(key),
                tenant,
            )
            return key
    finally:
        await conn.close()


def first_key_notice(key: str, tenant: str, base_url: str = "http://localhost:8000") -> str:
    return (
        "\n"
        "================================================================\n"
        " Relay is running. Your first API key (shown only once):\n"
        f"\n   {key}\n\n"
        f" It belongs to tenant {tenant!r}. Try it:\n"
        f"   curl {base_url}/v1/chat/completions \\\n"
        f"     -H 'Authorization: Bearer {key}' -H 'Content-Type: application/json' \\\n"
        '     -d \'{"model":"relay/cheap","messages":[{"role":"user","content":"hi"}]}\'\n'
        "================================================================\n"
    )
