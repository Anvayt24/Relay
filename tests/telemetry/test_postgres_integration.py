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
    # `public` stays on the path after the throwaway schema: the pgvector extension's `vector`
    # type lives there, and migration 0007 uses it.
    conn = await asyncpg.connect(dsn, server_settings={"search_path": f"{schema},public"})
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


async def test_attempts_are_written_with_their_request(
    schema_dsn: tuple[asyncpg.Connection, str],
) -> None:
    from relay.telemetry.log_writer import AttemptRow

    conn, schema = schema_dsn
    await migrate(conn)
    dsn = Settings().database_url
    assert dsn
    sink = PostgresSink(f"{dsn}?options=-csearch_path%3D{schema}")
    await sink.start()
    now = datetime.now(UTC)
    try:
        await sink.write(
            [
                LogRow(
                    request_id="r1",
                    ts=now,
                    tenant="t",
                    route="live",
                    priority="interactive",
                    cohort=None,
                    stream=False,
                    provider="backup",
                    model="b/model",
                    tier="cheap",
                    status="success",
                    error_class=None,
                    tokens_in=1,
                    tokens_out=1,
                    latency_ms=5,
                    ttft_ms=None,
                    fallback_hops=1,
                    degraded=True,
                    policy_version="static-v0",
                    attempts=(
                        AttemptRow("r1", 0, 0, "primary", "p/model", "server_error", None, 9, now),
                        AttemptRow("r1", 0, 1, "primary", "p/model", None, "breaker_open", 0, now),
                        AttemptRow("r1", 1, 0, "backup", "b/model", None, None, 4, now),
                    ),
                )
            ]
        )
    finally:
        await sink.close()

    request = await conn.fetchrow("SELECT fallback_hops, degraded, policy_version FROM requests")
    assert tuple(request or ()) == (1, True, "static-v0")
    attempts = await conn.fetch(
        "SELECT hop, retry, provider, error_class, skipped_reason FROM attempts ORDER BY id"
    )
    assert [tuple(a) for a in attempts] == [
        (0, 0, "primary", "server_error", None),
        (0, 1, "primary", None, "breaker_open"),
        (1, 0, "backup", None, None),
    ]


async def test_router_decision_columns_round_trip(
    schema_dsn: tuple[asyncpg.Connection, str],
) -> None:
    conn, schema = schema_dsn
    await migrate(conn)
    dsn = Settings().database_url
    assert dsn
    sink = PostgresSink(f"{dsn}?options=-csearch_path%3D{schema}%2Cpublic")
    await sink.start()
    try:
        await sink.write(
            [
                LogRow(
                    request_id=rid,
                    ts=datetime.now(UTC),
                    tenant="t",
                    route="default",
                    priority="interactive",
                    cohort=None,
                    stream=False,
                    provider="mock",
                    model="mock/cheap",
                    tier="cheap",
                    status="success",
                    error_class=None,
                    tokens_in=1,
                    tokens_out=1,
                    latency_ms=3,
                    ttft_ms=None,
                    policy_version=version,
                    router_reason=reason,
                    router_score=score,
                )
                for rid, version, reason, score in (
                    ("r1", "router-sim-v1", "router", 0.91),
                    ("r2", "static-v0", "alias", None),
                    ("r3", "router-unavailable", "router_fallback:no_artifact", None),
                )
            ]
        )
    finally:
        await sink.close()
    rows = await conn.fetch(
        "SELECT request_id, policy_version, router_reason, router_score FROM requests ORDER BY id"
    )
    assert [(r[0], r[1], r[2]) for r in rows] == [
        ("r1", "router-sim-v1", "router"),
        ("r2", "static-v0", "alias"),
        ("r3", "router-unavailable", "router_fallback:no_artifact"),
    ]
    assert rows[0]["router_score"] == pytest.approx(0.91, abs=1e-6)
    assert rows[1]["router_score"] is None
