"""The first-run API key and the lost-pepper check against a real Postgres:
`make up`, then `make test-integration`. A throwaway schema per test."""

import asyncio
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import asyncpg
import httpx
import pytest

from relay.api.app import create_app
from relay.settings import Settings
from relay.telemetry.migrate import migrate

from ..conftest import TEST_CONFIG_PATH  # type: ignore[import-not-found]

pytestmark = pytest.mark.integration

CHAT = "/v1/chat/completions"
BODY = {"model": "relay/cheap", "messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture
async def schema() -> AsyncIterator[tuple[asyncpg.Connection, str]]:
    dsn = Settings().database_url
    if not dsn:
        pytest.fail("RELAY_DATABASE_URL must be set for integration tests")
    name = f"test_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(dsn)
    await admin.execute(f"CREATE SCHEMA {name}")
    conn = await asyncpg.connect(dsn, server_settings={"search_path": f"{name},public"})
    await migrate(conn)
    try:
        yield conn, f"{dsn}?options=-csearch_path%3D{name}%2Cpublic"
    finally:
        await conn.close()
        await admin.execute(f"DROP SCHEMA {name} CASCADE")
        await admin.close()


def _app(dsn: str, data_dir: Path, **values: Any) -> Any:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        env="test",
        config_path=TEST_CONFIG_PATH,
        database_url=dsn,
        data_dir=str(data_dir),
        embedder="hash",
        **values,
    )
    return create_app(settings)


def _key_from(output: str) -> str:
    return next(w for w in output.split() if w.startswith("rk_"))


async def test_the_first_start_creates_one_key_that_works_and_a_restart_makes_none(
    schema: tuple[asyncpg.Connection, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    conn, dsn = schema
    app = _app(dsn, tmp_path, bootstrap_tenant="test")
    async with app.router.lifespan_context(app):
        shown = capsys.readouterr().out
        key = _key_from(shown)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://relay.test") as c:
            ok = await c.post(CHAT, json=BODY, headers={"Authorization": f"Bearer {key}"})
            no = await c.post(CHAT, json=BODY, headers={"Authorization": "Bearer rk_wrong"})
    assert (ok.status_code, no.status_code) == (200, 401)
    assert await conn.fetchval("SELECT count(*) FROM api_keys") == 1
    assert "only once" in shown  # the operator is told it will not be shown again

    again = _app(dsn, tmp_path, bootstrap_tenant="test")  # the same volume: the same pepper
    async with again.router.lifespan_context(again):
        assert "rk_" not in capsys.readouterr().out
    assert await conn.fetchval("SELECT count(*) FROM api_keys") == 1


async def test_replicas_starting_together_mint_exactly_one_key(
    schema: tuple[asyncpg.Connection, str], tmp_path: Path
) -> None:
    conn, dsn = schema
    apps = [_app(dsn, tmp_path, bootstrap_tenant="test") for _ in range(4)]

    async def start(app: Any) -> None:
        async with app.router.lifespan_context(app):
            await asyncio.sleep(0.05)

    await asyncio.gather(*(start(a) for a in apps))
    assert await conn.fetchval("SELECT count(*) FROM api_keys") == 1


async def test_a_new_pepper_beside_existing_keys_refuses_to_start(
    schema: tuple[asyncpg.Connection, str], tmp_path: Path
) -> None:
    # The volume was lost but the database was not: every existing key would silently stop
    # matching. Starting anyway would look healthy and reject every client.
    conn, dsn = schema
    await conn.execute(
        "INSERT INTO api_keys (key_hash, key_prefix, tenant) VALUES ($1, 'rk_old', 'test')",
        b"x" * 32,
    )
    app = _app(dsn, tmp_path)
    with pytest.raises(RuntimeError, match="pepper"):
        async with app.router.lifespan_context(app):
            pass


async def test_the_first_run_key_can_be_switched_off(
    schema: tuple[asyncpg.Connection, str], tmp_path: Path
) -> None:
    conn, dsn = schema
    app = _app(dsn, tmp_path, bootstrap_key=False)
    async with app.router.lifespan_context(app):
        pass
    assert await conn.fetchval("SELECT count(*) FROM api_keys") == 0


async def test_a_different_pepper_beside_keys_made_with_another_is_refused(
    schema: tuple[asyncpg.Connection, str], tmp_path: Path
) -> None:
    # The database remembers which pepper its keys were made with. Starting with another one,
    # even an explicit environment value, would reject every existing key without a word.
    conn, dsn = schema
    first = _app(dsn, tmp_path, bootstrap_tenant="test")
    async with first.router.lifespan_context(first):
        pass
    assert await conn.fetchval("SELECT count(*) FROM api_keys") == 1
    other = _app(dsn, tmp_path / "elsewhere", key_pepper="a-different-pepper")
    with pytest.raises(RuntimeError, match="different pepper"):
        async with other.router.lifespan_context(other):
            pass


async def test_with_no_keys_yet_a_changed_pepper_is_simply_recorded(
    schema: tuple[asyncpg.Connection, str], tmp_path: Path
) -> None:
    # Nothing was made with the old pepper, so there is nothing to invalidate.
    conn, dsn = schema
    for pepper in ("first", "second"):
        app = _app(dsn, tmp_path / pepper, key_pepper=pepper, bootstrap_key=False)
        async with app.router.lifespan_context(app):
            pass
    assert await conn.fetchval("SELECT count(*) FROM instance_settings") == 1
