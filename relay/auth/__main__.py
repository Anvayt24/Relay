"""API-key admin CLI.

    python -m relay.auth create-key --tenant dev     # prints the new key ONCE
    python -m relay.auth list
    python -m relay.auth revoke --prefix rk_3fA9xQ2

Needs RELAY_DATABASE_URL and RELAY_KEY_PEPPER (the same pepper as the running gateway, or the
key will never match). The key is printed once and never stored: lose it, make a new one.
"""

import argparse
import asyncio
import sys

import asyncpg

from relay.auth.keys import display_prefix, generate_key, hash_key
from relay.config.loader import load_config
from relay.settings import Settings


async def _run(args: argparse.Namespace, settings: Settings) -> int:
    if not settings.database_url or settings.key_pepper is None:
        print("RELAY_DATABASE_URL and RELAY_KEY_PEPPER must be set", file=sys.stderr)
        return 2
    conn = await asyncpg.connect(settings.database_url)
    try:
        if args.command == "create-key":
            config = load_config(settings.config_path)
            if args.tenant not in config.tenants:
                print(
                    f"unknown tenant {args.tenant!r}; known: {sorted(config.tenants)}",
                    file=sys.stderr,
                )
                return 2
            key = generate_key()
            await conn.execute(
                "INSERT INTO api_keys (key_hash, key_prefix, tenant) VALUES ($1, $2, $3)",
                hash_key(key, settings.key_pepper.get_secret_value()),
                display_prefix(key),
                args.tenant,
            )
            print(key)
            print(
                f"created for tenant {args.tenant!r}; store it now, it is not shown again",
                file=sys.stderr,
            )
        elif args.command == "list":
            rows = await conn.fetch(
                "SELECT key_prefix, tenant, status, created_at FROM api_keys ORDER BY id"
            )
            for r in rows:
                created = f"{r['created_at']:%Y-%m-%d %H:%M}"
                print(f"{r['key_prefix']}...  {r['tenant']:<12} {r['status']:<8} {created}")
        elif args.command == "revoke":
            result = await conn.execute(
                "UPDATE api_keys SET status = 'revoked' "
                "WHERE key_prefix = $1 AND status = 'active'",
                args.prefix,
            )
            print(result)  # "UPDATE <n>"
        return 0
    finally:
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m relay.auth")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create-key")
    create.add_argument("--tenant", required=True)
    sub.add_parser("list")
    revoke = sub.add_parser("revoke")
    revoke.add_argument("--prefix", required=True)
    raise SystemExit(asyncio.run(_run(parser.parse_args(), Settings())))


if __name__ == "__main__":
    main()
