"""Integration tests against a real Postgres: `make up`, then `make test-integration`.

They use RELAY_DATABASE_URL, and work in a throwaway schema so they never touch real data.
In CI they run against a Postgres service container (the `integration` job).
"""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import asyncpg
import pytest

from relay.auth.keys import display_prefix, generate_key, hash_key
from relay.auth.store import PostgresKeyStore
from relay.settings import Settings
from relay.telemetry.log_writer import LogRow
from relay.telemetry.migrate import migrate
from relay.telemetry.postgres import PostgresSink

pytestmark = pytest.mark.integration


@pytest.fixture
async def schema_dsn() -> AsyncIterator[tuple[asyncpg.Connection, str]]:
    dsn = Settings().database_url
    if not dsn:
        pytest.fail("RELAY_DATABASE_URL must be set for integration tests")
    schema = f"test_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(dsn)
    await admin.execute(f"CREATE SCHEMA {schema}")
    # search_path on the connection makes every unqualified name (requests,
    # schema_migrations) resolve inside the throwaway schema.
    conn = await asyncpg.connect(dsn, server_settings={"search_path": schema})
    try:
        yield conn, schema
    finally:
        await conn.close()
        await admin.execute(f"DROP SCHEMA {schema} CASCADE")
        await admin.close()


async def test_migrate_is_idempotent(schema_dsn: tuple[asyncpg.Connection, str]) -> None:
    conn, _ = schema_dsn
    applied = await migrate(conn)
    assert applied[0] == "0001_requests.sql"
    assert applied == sorted(applied)  # in filename order
    assert await migrate(conn) == []  # second run: nothing left to apply
    assert await conn.fetchval("SELECT count(*) FROM schema_migrations") == len(applied)


async def test_postgres_sink_round_trip(schema_dsn: tuple[asyncpg.Connection, str]) -> None:
    conn, schema = schema_dsn
    await migrate(conn)
    dsn = Settings().database_url
    assert dsn
    sink = PostgresSink(f"{dsn}?options=-csearch_path%3D{schema}")
    await sink.start()
    try:
        rows = [
            LogRow(
                request_id="dup",  # the same request_id twice: allowed (surrogate key)
                ts=datetime.now(UTC),
                tenant="t",
                route="default",
                priority="interactive",
                cohort=None,
                stream=True,
                provider="mock",
                model="mock/strong",
                tier="strong",
                status=status,
                error_class=None,
                tokens_in=3,
                tokens_out=5,
                latency_ms=12,
                ttft_ms=4,
                billed_cost_nano_usd=0,
                list_cost_nano_usd=1400,
                counterfactual_strong_nano_usd=14_000,
                price_table_version="v1",
            )
            for status in ("success", "failed_midstream")
        ]
        await sink.write(rows)
    finally:
        await sink.close()

    stored = await conn.fetch(
        "SELECT request_id, status, ttft_ms, list_cost_nano_usd FROM requests ORDER BY id"
    )
    assert [tuple(r) for r in stored] == [
        ("dup", "success", 4, 1400),
        ("dup", "failed_midstream", 4, 1400),
    ]


async def test_status_check_constraint(schema_dsn: tuple[asyncpg.Connection, str]) -> None:
    conn, _ = schema_dsn
    await migrate(conn)
    with pytest.raises(asyncpg.CheckViolationError):
        await conn.execute(
            "INSERT INTO requests (request_id, ts, tenant, route, priority, stream, status,"
            " latency_ms) VALUES ('x', now(), 't', 'r', 'interactive', false, 'bogus', 1)"
        )


async def test_api_key_lookup(schema_dsn: tuple[asyncpg.Connection, str]) -> None:
    conn, schema = schema_dsn
    await migrate(conn)
    key, revoked = generate_key(), generate_key()
    for k, status in ((key, "active"), (revoked, "revoked")):
        await conn.execute(
            "INSERT INTO api_keys (key_hash, key_prefix, tenant, status) VALUES ($1, $2, $3, $4)",
            hash_key(k, "pep"),
            display_prefix(k),
            "dev",
            status,
        )
    dsn = Settings().database_url
    store = PostgresKeyStore(f"{dsn}?options=-csearch_path%3D{schema}")
    await store.start()
    try:
        assert await store.lookup(hash_key(key, "pep")) == "dev"
        assert await store.lookup(hash_key(revoked, "pep")) is None  # revoked = gone
        assert await store.lookup(hash_key(key, "other-pepper")) is None
    finally:
        await store.close()
