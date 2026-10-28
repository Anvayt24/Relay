"""Where bundles and the pointer live (PRD §9.4 ``policy_bundles``; ADR 0024).

- ``policy_bundles``: one immutable row per version (the bundle as JSON, and its hash);
- ``policy_pointer``: a single row: what is active, what was, and the running candidate;
- ``policy_events``: an audit row for every change: who, what, why.

``change`` is the only way to move the pointer: it reads the pointer, applies one transition
from ``control.py`` and writes the pointer and its event **in one transaction**, with the row
locked (``FOR UPDATE``), so two operators (or an operator and the evidence worker) can't
interleave and lose an update.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import asyncpg

from relay.policy.bundle import PolicyBundle
from relay.policy.control import ControlError, Pointer, Transition


@dataclass(frozen=True)
class BundleInfo:
    version: str
    content_sha256: str
    created_at: datetime
    created_by: str
    note: str


@dataclass(frozen=True)
class PolicyEvent:
    ts: datetime
    action: str
    actor: str
    before: Pointer
    after: Pointer
    reason: str = ""


class PolicyStore(Protocol):
    async def start(self) -> None: ...
    async def close(self) -> None: ...
    async def create(self, bundle: PolicyBundle, *, created_by: str) -> None: ...
    async def get(self, version: str) -> PolicyBundle | None: ...
    async def bundles(self) -> list[BundleInfo]: ...
    async def pointer(self) -> Pointer: ...
    async def change(
        self, action: str, transition: Transition, *, actor: str, reason: str = ""
    ) -> Pointer: ...
    async def events(self, limit: int = 20) -> list[PolicyEvent]: ...


def _check_versions(after: Pointer, known: Callable[[str], bool]) -> None:
    for v in (after.active_version, after.candidate_version):
        if v is not None and not known(v):
            raise ControlError(f"no bundle {v!r}; create it first (python -m relay.policy create)")


class MemoryPolicyStore:
    """Tests, and a single process without a database."""

    def __init__(self) -> None:
        self._saved: dict[str, tuple[PolicyBundle, BundleInfo]] = {}
        self._pointer = Pointer()
        self.history: list[PolicyEvent] = []
        self.fail = False  # tests: simulate a database outage

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    def _up(self) -> None:
        if self.fail:
            raise ConnectionError("policy store unavailable")

    async def create(self, bundle: PolicyBundle, *, created_by: str) -> None:
        self._up()
        if bundle.version in self._saved:
            raise ControlError(f"bundle {bundle.version} already exists (bundles are immutable)")
        info = BundleInfo(
            bundle.version, bundle.content_sha256, bundle.created_at, created_by, bundle.note
        )
        self._saved[bundle.version] = (bundle, info)

    async def get(self, version: str) -> PolicyBundle | None:
        self._up()
        entry = self._saved.get(version)
        return entry[0] if entry else None

    async def bundles(self) -> list[BundleInfo]:
        self._up()
        return sorted((info for _, info in self._saved.values()), key=lambda i: i.created_at)

    async def pointer(self) -> Pointer:
        self._up()
        return self._pointer

    async def change(
        self, action: str, transition: Transition, *, actor: str, reason: str = ""
    ) -> Pointer:
        self._up()
        before = self._pointer
        after = transition(before)
        _check_versions(after, lambda v: v in self._saved)
        self._pointer = after
        self.history.append(PolicyEvent(datetime.now(UTC), action, actor, before, after, reason))
        return after

    async def events(self, limit: int = 20) -> list[PolicyEvent]:
        self._up()
        return list(reversed(self.history))[:limit]


_POINTER_COLUMNS = (
    "active_version, previous_version, candidate_version, candidate_mode, canary_pct, updated_at"
)


def _pointer(row: Any) -> Pointer:
    if row is None:
        return Pointer()
    return Pointer(
        active_version=row["active_version"],
        previous_version=row["previous_version"],
        candidate_version=row["candidate_version"],
        candidate_mode=row["candidate_mode"],
        canary_pct=row["canary_pct"],
        updated_at=row["updated_at"],
    )


def _pointer_json(p: Pointer) -> dict[str, Any]:
    return {
        "active_version": p.active_version,
        "previous_version": p.previous_version,
        "candidate_version": p.candidate_version,
        "candidate_mode": p.candidate_mode,
        "canary_pct": p.canary_pct,
    }


class PostgresPolicyStore:
    def __init__(self, dsn: str, *, command_timeout_s: float = 5.0) -> None:
        self._dsn = dsn
        self._command_timeout_s = command_timeout_s
        self._pool: asyncpg.Pool | None = None

    async def start(self) -> None:
        # min_size=0: a database that is down at startup costs nothing until the first poll.
        self._pool = await asyncpg.create_pool(
            self._dsn, min_size=0, max_size=2, command_timeout=self._command_timeout_s
        )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()

    def _need(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgresPolicyStore used before start()")
        return self._pool

    async def create(self, bundle: PolicyBundle, *, created_by: str) -> None:
        async with self._need().acquire() as conn:
            try:
                await conn.execute(
                    "INSERT INTO policy_bundles (version, content, content_sha256, created_at, "
                    "created_by, note) VALUES ($1, $2::jsonb, $3, $4, $5, $6)",
                    bundle.version,
                    bundle.to_json(),
                    bundle.content_sha256,
                    bundle.created_at,
                    created_by,
                    bundle.note,
                )
            except asyncpg.UniqueViolationError:
                raise ControlError(
                    f"bundle {bundle.version} already exists (bundles are immutable)"
                ) from None

    async def get(self, version: str) -> PolicyBundle | None:
        async with self._need().acquire() as conn:
            row = await conn.fetchrow(
                "SELECT content, content_sha256 FROM policy_bundles WHERE version = $1", version
            )
        if row is None:
            return None
        bundle = PolicyBundle.from_json(row["content"])
        if bundle.content_sha256 != row["content_sha256"]:
            raise ControlError(f"bundle {version}: stored content does not match its hash")
        return bundle

    async def bundles(self) -> list[BundleInfo]:
        async with self._need().acquire() as conn:
            rows = await conn.fetch(
                "SELECT version, content_sha256, created_at, created_by, "
                "coalesce(note, '') AS note FROM policy_bundles ORDER BY created_at"
            )
        return [BundleInfo(**dict(r)) for r in rows]

    async def pointer(self) -> Pointer:
        async with self._need().acquire() as conn:
            row = await conn.fetchrow(f"SELECT {_POINTER_COLUMNS} FROM policy_pointer")
        return _pointer(row)

    async def change(
        self, action: str, transition: Transition, *, actor: str, reason: str = ""
    ) -> Pointer:
        async with self._need().acquire() as conn, conn.transaction():
            row = await conn.fetchrow(f"SELECT {_POINTER_COLUMNS} FROM policy_pointer FOR UPDATE")
            before = _pointer(row)
            after = transition(before)
            known = {
                r["version"]
                for r in await conn.fetch(
                    "SELECT version FROM policy_bundles WHERE version = ANY($1::text[])",
                    [v for v in (after.active_version, after.candidate_version) if v],
                )
            }
            _check_versions(after, lambda v: v in known)
            await conn.execute(
                "INSERT INTO policy_pointer (id, active_version, previous_version, "
                "candidate_version, candidate_mode, canary_pct, updated_at) "
                "VALUES (true, $1, $2, $3, $4, $5, $6) ON CONFLICT (id) DO UPDATE SET "
                "active_version = EXCLUDED.active_version, "
                "previous_version = EXCLUDED.previous_version, "
                "candidate_version = EXCLUDED.candidate_version, "
                "candidate_mode = EXCLUDED.candidate_mode, canary_pct = EXCLUDED.canary_pct, "
                "updated_at = EXCLUDED.updated_at",
                after.active_version,
                after.previous_version,
                after.candidate_version,
                after.candidate_mode,
                after.canary_pct,
                after.updated_at or datetime.now(UTC),
            )
            await conn.execute(
                "INSERT INTO policy_events (ts, action, actor, before, after, reason) "
                "VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6)",
                datetime.now(UTC),
                action,
                actor,
                json.dumps(_pointer_json(before)),
                json.dumps(_pointer_json(after)),
                reason,
            )
        return after

    async def events(self, limit: int = 20) -> list[PolicyEvent]:
        async with self._need().acquire() as conn:
            rows = await conn.fetch(
                "SELECT ts, action, actor, before, after, coalesce(reason, '') AS reason "
                "FROM policy_events ORDER BY id DESC LIMIT $1",
                limit,
            )
        return [
            PolicyEvent(
                r["ts"],
                r["action"],
                r["actor"],
                Pointer(**json.loads(r["before"])),
                Pointer(**json.loads(r["after"])),
                r["reason"],
            )
            for r in rows
        ]


def describe(p: Pointer) -> dict[str, Any]:
    return {**_pointer_json(p), "updated_at": p.updated_at.isoformat() if p.updated_at else None}


def bundle_infos(infos: Sequence[BundleInfo]) -> list[dict[str, Any]]:
    return [
        {
            "version": i.version,
            "sha256": i.content_sha256[:12],
            "created_at": i.created_at.isoformat(),
            "created_by": i.created_by,
            "note": i.note,
        }
        for i in infos
    ]
