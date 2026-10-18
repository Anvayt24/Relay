"""A deliberately small schema migrator: plain SQL files applied in order (ADR 0003).

    python -m relay.telemetry.migrate     # or: make migrate

Rules:
- Files are ``relay/telemetry/migrations/NNNN_description.sql``, applied in ``NNNN`` order.
- Each applied version is recorded in ``schema_migrations``, so a re-run applies only new
  files.
- Each file runs in its own transaction, together with its ``schema_migrations`` row. Either
  both happen or neither does, so a failed migration leaves no half-applied schema behind.
- **Never edit an applied file.** Add a new one. The database only remembers *that* a version
  ran, not what it said.

The files live inside the package (read with ``importlib.resources``), so they ship in the
wheel and the Docker image with no extra COPY step.
"""

import asyncio
import re
import sys
from dataclasses import dataclass
from importlib import resources

import asyncpg

from relay.settings import Settings

_FILENAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")

# Any fixed 64-bit number works; it only has to be the same in every process. Two replicas
# starting together both run migrations: the advisory lock makes the second wait until the
# first is done, after which it finds nothing left to apply.
_ADVISORY_LOCK_ID = 727_001


@dataclass(frozen=True)
class Migration:
    version: str
    name: str
    sql: str


def discover() -> list[Migration]:
    folder = resources.files("relay.telemetry") / "migrations"
    found: dict[str, Migration] = {}
    for entry in folder.iterdir():
        match = _FILENAME.match(entry.name)
        if match is None:
            continue
        version = match.group(1)
        if version in found:
            raise ValueError(f"duplicate migration version {version}: {entry.name}")
        found[version] = Migration(version, entry.name, entry.read_text(encoding="utf-8"))
    return [found[v] for v in sorted(found)]


async def migrate(conn: asyncpg.Connection) -> list[str]:
    """Apply pending migrations; return the names of the ones applied."""
    applied_now: list[str] = []
    await conn.execute("SELECT pg_advisory_lock($1)", _ADVISORY_LOCK_ID)
    try:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version    text PRIMARY KEY,
                applied_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        done = {row["version"] for row in await conn.fetch("SELECT version FROM schema_migrations")}
        for migration in discover():
            if migration.version in done:
                continue
            async with conn.transaction():
                # execute() without arguments uses Postgres's simple query protocol, which
                # accepts several statements in one string: a whole migration file.
                await conn.execute(migration.sql)
                await conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES ($1)", migration.version
                )
            applied_now.append(migration.name)
    finally:
        await conn.execute("SELECT pg_advisory_unlock($1)", _ADVISORY_LOCK_ID)
    return applied_now


async def _main() -> int:
    dsn = Settings().database_url
    if not dsn:
        print("RELAY_DATABASE_URL is not set", file=sys.stderr)
        return 2
    conn = await asyncpg.connect(dsn)
    try:
        applied = await migrate(conn)
    finally:
        await conn.close()
    print(f"applied: {', '.join(applied)}" if applied else "schema is up to date")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
