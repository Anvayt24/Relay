"""Postgres sink for request-log rows.

**Relay must start and serve with Postgres down** (PRD §8, degradation matrix). So the pool is
created with ``min_size=0``: creating it opens no connection, and a database that's down at
boot costs nothing until the first write. That write fails, the LogWriter backs off, and rows
are eventually dropped and counted. Requests never notice.

Writes use ``COPY`` (asyncpg's ``copy_records_to_table``), Postgres's bulk-load protocol. One
``COPY`` streams the whole batch in a single round trip, skipping per-row statement overhead.
For batches of hundreds of rows, that is typically an order of magnitude faster than
``INSERT``, which is the reason the writer batches at all.

A batch is two COPYs, ``requests`` then ``attempts``, in **one transaction**: a request row
is never visible without its attempt rows, and a retried batch never leaves orphans behind.
"""

from collections.abc import Sequence
from dataclasses import astuple, fields

import asyncpg

from relay.telemetry.log_writer import AttemptRow, LogRow

# `attempts` is a nested tuple on LogRow, written to its own table.
COLUMNS = [f.name for f in fields(LogRow) if f.name != "attempts"]
ATTEMPT_COLUMNS = [f.name for f in fields(AttemptRow)]


class PostgresSink:
    def __init__(self, dsn: str, *, command_timeout_s: float = 5.0) -> None:
        self._dsn = dsn
        self._command_timeout_s = command_timeout_s
        self._pool: asyncpg.Pool | None = None

    async def start(self) -> None:
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=0,  # no connection at startup; see the module docstring
            max_size=2,  # one writer task needs one connection; one spare
            command_timeout=self._command_timeout_s,
        )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()

    async def write(self, rows: Sequence[LogRow]) -> None:
        if self._pool is None:
            raise RuntimeError("PostgresSink.write() called before start()")
        requests = [tuple(getattr(row, c) for c in COLUMNS) for row in rows]
        attempts = [astuple(a) for row in rows for a in row.attempts]
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.copy_records_to_table("requests", records=requests, columns=COLUMNS)
            if attempts:
                await conn.copy_records_to_table(
                    "attempts", records=attempts, columns=ATTEMPT_COLUMNS
                )
