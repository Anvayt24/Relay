"""Migrations 0009 (evidence engine) and 0010 (policy bundles) against a real Postgres:
`make up`, then `make test-integration`. A throwaway schema per test, as in
test_postgres_integration.py."""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import asyncpg
import pytest

from relay.evidence.cards import CardSpec, calibration_by_judge, compute_cards
from relay.evidence.pairs import EvalPair, read_jsonl
from relay.evidence.store import HumanLabel, PostgresEvidenceStore
from relay.policy import control
from relay.policy.bundle import BundleError, PolicyBundle, RoutePolicy
from relay.policy.control import ControlError
from relay.policy.store import PostgresPolicyStore
from relay.settings import Settings
from relay.telemetry.log_writer import LogRow
from relay.telemetry.migrate import migrate
from relay.telemetry.postgres import PostgresSink

pytestmark = pytest.mark.integration

PILOT_PAIRS = Path(__file__).resolve().parents[2] / "evals" / "judge" / "pilot_pairs.jsonl"


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


def shadow_pair(i: int, label: str, ts: datetime) -> EvalPair:
    return EvalPair(
        pair_id=f"s{i}",
        request_id=f"r{i}",
        route="default",
        prompt_hash="h",
        cohort="lang=latin,len=short,code=false,tools=false",
        comparison="cheap_vs_strong",
        candidate_model="mock/cheap",
        reference_model="mock/strong",
        judge_label=label,
        judge_model="judge",
        judge_prompt_version="v1",
        position_swapped=False,
        text_retained=i % 2 == 0,
        ts=ts,
        source="shadow",
        policy_version="2026-10-28.1",
        prompt_text="hi" if i % 2 == 0 else None,
        cost_nano_usd=1000,
    )


async def test_evidence_store_round_trip(schema: tuple[asyncpg.Connection, str]) -> None:
    conn, dsn = schema
    store = PostgresEvidenceStore(dsn)
    await store.start()
    try:
        t0 = datetime.now(UTC) - timedelta(hours=1)
        await store.write_pairs([shadow_pair(i, "tie" if i % 5 else "loss", t0) for i in range(60)])
        labels = [
            HumanLabel(f"s{i}", "tie" if i % 5 else "loss", "a1", False, t0) for i in range(55)
        ] + [HumanLabel("s0", "tie", "a1", True, t0), HumanLabel("ghost", "tie", "a1", False, t0)]
        assert await store.import_labels(labels) == 56  # the ghost pair is skipped
        assert await store.import_labels(labels) == 0  # re-import adds nothing
        pairs = await store.load_pairs()
        assert len(pairs) == 60
        assert pairs[0].human_label == "loss"  # the first label, not the blind re-label

        cards = compute_cards(pairs, CardSpec(), calibration=calibration_by_judge(pairs), now=t0)
        await store.save_cards(cards)
        await store.save_cards(cards)  # an upsert, not a duplicate
        loaded = await store.load_cards()
        assert {c.key for c in loaded} == {c.key for c in cards}
        assert await conn.fetchval("SELECT count(*) FROM route_cards") == len(cards)
        assert (
            await conn.fetchval("SELECT count(*) FROM eval_pairs WHERE prompt_text IS NOT NULL")
            == 30
        )
    finally:
        await store.close()


async def test_policy_store_round_trip(schema: tuple[asyncpg.Connection, str]) -> None:
    conn, dsn = schema
    store = PostgresPolicyStore(dsn)
    await store.start()
    try:
        a = PolicyBundle(version="2026-10-28.1", routes={"default": RoutePolicy()})
        b = PolicyBundle(
            version="2026-10-28.2", routes={"default": RoutePolicy(max_fallback_hops=1)}
        )
        await store.create(a, created_by="t")
        await store.create(b, created_by="t")
        with pytest.raises(ControlError):
            await store.create(a, created_by="t")  # immutable
        assert (await store.get(a.version)).content_sha256 == a.content_sha256  # type: ignore[union-attr]
        assert [i.version for i in await store.bundles()] == [a.version, b.version]

        await store.change("activate", lambda p: control.activate(p, a.version), actor="op")
        await store.change("canary", lambda p: control.canary(p, b.version, 10), actor="op")
        p = await store.pointer()
        assert (p.active_version, p.candidate_version, p.canary_pct) == (a.version, b.version, 10)
        with pytest.raises(ControlError):
            await store.change(
                "activate", lambda p: control.activate(p, "2099-01-01.1"), actor="op"
            )
        await store.change("rollback", control.rollback, actor="evidence-worker", reason="unsafe")
        assert (await store.pointer()).candidate_version is None
        events = await store.events()
        assert [e.action for e in events] == ["rollback", "canary", "activate"]
        assert events[0].actor == "evidence-worker" and events[0].reason == "unsafe"
        # Tampering with a stored bundle is caught on read.
        await conn.execute(
            "UPDATE policy_bundles SET content = jsonb_set(content, '{note}', '\"x\"') "
            "WHERE version = $1",
            a.version,
        )
        assert await store.get(a.version) is not None  # the note isn't policy: hash unchanged
        await conn.execute(
            "UPDATE policy_bundles SET content_sha256 = 'bad' WHERE version = $1", a.version
        )
        # A BundleError, so the poller counts it as invalid_bundle, not as the store being down.
        with pytest.raises(BundleError, match="does not match"):
            await store.get(a.version)
    finally:
        await store.close()


async def test_canary_and_shadow_columns_round_trip(
    schema: tuple[asyncpg.Connection, str],
) -> None:
    conn, dsn = schema
    sink = PostgresSink(dsn)
    await sink.start()
    try:
        await sink.write(
            [
                LogRow(
                    request_id="r1", ts=datetime.now(UTC), tenant="t", route="default",
                    priority="interactive", cohort=None, stream=False, provider="mock",
                    model="mock/cheap", tier="cheap", status="success", error_class=None,
                    tokens_in=1, tokens_out=1, latency_ms=3, ttft_ms=None,
                    policy_version="2026-10-28.1", canary_arm="control",
                    shadow_policy_version="2026-10-28.2", shadow_tier="strong", shadow_score=0.4,
                )
            ]
        )  # fmt: skip
    finally:
        await sink.close()
    row = await conn.fetchrow(
        "SELECT canary_arm, shadow_policy_version, shadow_tier, shadow_score FROM requests"
    )
    assert (row[0], row[1], row[2]) == ("control", "2026-10-28.2", "strong")
    assert row[3] == pytest.approx(0.4, abs=1e-6)


async def test_shadow_pairs_export_for_labeling_and_labels_import_back(
    schema: tuple[asyncpg.Connection, str],
) -> None:
    # Found in review: the labeling tool only read JSONL files, and nothing exported the
    # production shadow pairs, so a route's rectifier could never reach 50 labels.
    from dataclasses import replace

    _, dsn = schema
    store = PostgresEvidenceStore(dsn)
    await store.start()
    try:
        t0 = datetime.now(UTC) - timedelta(hours=1)
        rows = [
            replace(shadow_pair(i, "tie", t0), candidate_text="a", reference_text="b")
            for i in range(10)
        ]
        await store.write_pairs(rows)
        await store.import_labels([HumanLabel("s0", "tie", "a1", False, t0)])
        exported = await store.labelable_pairs("default", 100)
        # Only even pairs kept their text, and s0 is already labeled.
        assert sorted(p.pair_id for p in exported) == ["s2", "s4", "s6", "s8"]
        assert all(p.prompt_text == "hi" and p.candidate_text == "a" for p in exported)
        assert exported[0] == next(r for r in rows if r.pair_id == exported[0].pair_id)
        assert await store.labelable_pairs("other-route", 100) == []
        labels = [HumanLabel(p.pair_id, "loss", "a1", False, t0) for p in exported]
        assert await store.import_labels(labels) == 4  # the labels land on the shadow pairs
        assert await store.labelable_pairs("default", 100) == []
    finally:
        await store.close()


async def test_expired_text_is_purged_in_postgres(schema: tuple[asyncpg.Connection, str]) -> None:
    from dataclasses import replace

    conn, dsn = schema
    store = PostgresEvidenceStore(dsn)
    await store.start()
    try:
        t0 = datetime.now(UTC)
        old = replace(shadow_pair(0, "tie", t0), text_expires_at=t0 - timedelta(seconds=1))
        new = replace(shadow_pair(2, "tie", t0), text_expires_at=t0 + timedelta(days=1))
        await store.write_pairs([old, new])
        assert await store.purge_expired_text(t0) == 1
        rows = await conn.fetch("SELECT pair_id, prompt_text FROM eval_pairs ORDER BY pair_id")
        assert [(r["pair_id"], r["prompt_text"]) for r in rows] == [("s0", None), ("s2", "hi")]
    finally:
        await store.close()


async def test_offline_pairs_import_idempotently(schema: tuple[asyncpg.Connection, str]) -> None:
    # The pilot set, as committed (old JSONL field names), through the store's import: the
    # one column list, and a re-run adds nothing.
    conn, dsn = schema
    pilot = read_jsonl(PILOT_PAIRS)
    store = PostgresEvidenceStore(dsn)
    await store.start()
    try:
        assert await store.import_pairs(pilot) == len(pilot)
        assert await store.import_pairs(pilot) == 0
        row = await conn.fetchrow(
            "SELECT route, comparison, source, prompt_text FROM eval_pairs WHERE pair_id = $1",
            pilot[0].pair_id,
        )
        assert row["route"] is None and row["source"] == pilot[0].source
        assert row["comparison"] == pilot[0].comparison
        assert row["prompt_text"] == pilot[0].prompt_text
    finally:
        await store.close()
