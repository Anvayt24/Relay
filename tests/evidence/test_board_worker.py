"""The card board (cards -> edge states on every replica) and the evidence worker (pairs ->
cards -> actions), including the end-to-end path: an edge's card turns unsafe, and a strict
tenant stops being served through it."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
import yaml
from fastapi import FastAPI

from relay.api.app import create_app
from relay.config.schema import RelayConfig
from relay.evidence import worker
from relay.evidence.board import CardBoard, fallback_comparison
from relay.evidence.cards import ALL, RouteCard
from relay.evidence.label import human_labels
from relay.evidence.store import (
    HumanLabel,
    MemoryEvidenceStore,
    ShadowPair,
    card_from_dict,
)
from relay.reliability.edges import SeededEdgeStates
from relay.schemas.errors import ErrorClass
from relay.simulator.plan import MockBehavior

from ..conftest import test_settings as relay_settings  # type: ignore[import-not-found]
from ..reliability.conftest import ScriptedProvider  # type: ignore[import-not-found]
from .test_cards import calibration_set, make_pairs

NOW = datetime(2026, 10, 28, 12, tzinfo=UTC)
FALLBACK_YAML = "tests/config/fallback.yaml"


def card(state: str, comparison: str = "fallback:p/model->b/model", **kw: Any) -> RouteCard:
    values: dict[str, Any] = {
        "route": "safe",
        "cohort": ALL,
        "comparison": comparison,
        "estimate": 0.8,
        "ci": (0.7, 0.85),
        "floor": 0.9,
        "n_judge": 1000,
        "n_human": 100,
        "judge": "j@v1",
        "judge_kappa": 0.7,
        "state": state,
        "recommended_action": "disable_edge",
        "look": 2,
        "max_looks": 4,
        "policy_version": "p",
        "updated_at": NOW,
        **kw,
    }
    return RouteCard(**values)


with open(FALLBACK_YAML, encoding="utf-8") as _f:
    PLAIN_FALLBACK = RelayConfig.model_validate(yaml.safe_load(_f))


def fallback_config(**evidence: Any) -> RelayConfig:
    with open(FALLBACK_YAML, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["routes"]["safe"]["evidence"] = {"judge": "b/model", "look_every": 100, **evidence}
    return RelayConfig.model_validate(raw)


# ------------------------------------------------------------------ the board
async def test_card_states_map_to_edge_states_and_seeds_fill_the_gaps() -> None:
    cfg = fallback_config()
    store = MemoryEvidenceStore()
    board = CardBoard(store, seeded=SeededEdgeStates(cfg))
    assert board.state("safe", "p/model", "b/model") == "safe"  # the YAML seed, no card yet
    for state, edge in [
        ("safe", "safe"),
        ("unsafe", "unsafe"),
        ("undecided", "undecided"),
        ("uncalibrated", "undecided"),  # no automatic change without a calibrated judge
        ("stale", "undecided"),
    ]:
        await store.save_cards([card(state)])
        await board.refresh()
        assert board.state("safe", "p/model", "b/model") == edge
    assert board.card_state("safe", fallback_comparison("p/model", "b/model")) == "stale"
    assert board.card_state("safe", "cheap_vs_strong") is None


async def test_a_failed_refresh_keeps_the_last_cards() -> None:
    class Flaky(MemoryEvidenceStore):
        down = False

        async def load_cards(self) -> list[RouteCard]:
            if self.down:
                raise ConnectionError("postgres down")
            return await super().load_cards()

    store = Flaky()
    board = CardBoard(store, seeded=SeededEdgeStates(fallback_config()))
    await store.save_cards([card("unsafe")])
    assert await board.refresh()
    store.down = True
    assert not await board.refresh()
    assert board.state("safe", "p/model", "b/model") == "unsafe"
    assert board.loaded


async def test_the_board_polls_in_the_background() -> None:
    store = MemoryEvidenceStore()
    board = CardBoard(store, seeded=SeededEdgeStates(fallback_config()), poll_s=0.01)
    await store.save_cards([card("unsafe")])
    board.start()
    try:
        for _ in range(100):
            if board.cards():
                break
            await asyncio.sleep(0.01)
    finally:
        await board.stop()
    assert len(board.cards()) == 1


# ------------------------------------------------------------------ end to end
@asynccontextmanager
async def serving(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://relay.test") as c:
            yield c


async def test_an_unsafe_card_takes_the_edge_away_from_strict_tenants() -> None:
    store = MemoryEvidenceStore()
    primary = ScriptedProvider(then=MockBehavior(error=ErrorClass.SERVER_ERROR))
    backup = ScriptedProvider()
    app = create_app(
        relay_settings(auth_disabled_tenant="strict", first_token_timeout_s=1.0),
        providers={
            "primary": primary,
            "backup": backup,
            "big": ScriptedProvider(),
            "capped": ScriptedProvider(),
        },
        config=fallback_config(),
        evidence_store=store,
    )
    body = {"model": "relay/auto", "messages": [{"role": "user", "content": "hi"}]}
    async with serving(app) as c:
        # Seeded safe, no card yet: the primary fails and the backup serves.
        ok = await c.post("/v1/chat/completions", json=body)
        assert ok.status_code == 200 and ok.headers["x-relay-model"] == "b/model"
        # The worker writes an unsafe card; the next poll reaches this replica.
        await store.save_cards([card("unsafe")])
        await app.state.card_board.refresh()
        refused = await c.post("/v1/chat/completions", json=body)
    # Under strict an unsafe edge is not a candidate at all, so no fallback existed and the
    # client gets the primary's own error (Phase 7's rule, tests/reliability/test_dispatch.py).
    assert refused.status_code == 502
    assert backup.calls == 1  # only the first request's failover


# ------------------------------------------------------------------ the worker
def shadow_rows(pairs: list[Any]) -> list[ShadowPair]:
    return [
        ShadowPair(
            pair_id=p.pair_id,
            request_id=f"req-{p.pair_id}",
            route="safe",
            prompt_hash="h",
            cohort=p.cohort,
            comparison=p.comparison,
            candidate_model="b/model",
            reference_model="p/model",
            judge_label=p.judge_label,
            judge_model=p.judge_model,
            judge_prompt_version=p.judge_prompt_version,
            position_swapped=False,
            text_retained=False,
            ts=p.ts,
            policy_version="p",
        )
        for p in pairs
    ]


async def _store_with(w: float) -> MemoryEvidenceStore:
    store = MemoryEvidenceStore()
    pairs = make_pairs(
        n_judge=400, n_human=150, w=w, comparison="fallback:p/model->b/model", route="safe"
    )
    store.pairs = shadow_rows(pairs)
    # The judge's calibration set (no route): it gates the cards, it isn't one of them.
    store.extra = [replace(p, route="") for p in calibration_set_pairs()]
    labels = [
        HumanLabel(p.pair_id, p.human_label, "a1", False, NOW - timedelta(days=1))
        for p in pairs
        if p.human_label is not None
    ]
    assert await store.import_labels(labels) == len(labels)
    return store


def calibration_set_pairs() -> list[Any]:
    return make_pairs(n_judge=0, n_human=400, w=0.7, seed=99, comparison="calibration")


async def test_the_worker_turns_pairs_into_cards_and_reports_actions() -> None:
    store = await _store_with(0.75)
    seen: list[RouteCard] = []

    async def on_action(c: RouteCard) -> None:
        seen.append(c)

    cards = await worker.run_once(store, fallback_config(), now=NOW, on_action=on_action)
    [all_card] = [c for c in cards if c.cohort == ALL]
    assert all_card.state == "unsafe"
    assert all_card.recommended_action == "disable_edge"
    assert [c.recommended_action for c in seen] == ["disable_edge"]
    assert store.cards[all_card.key].state == "unsafe"
    # A second run at the same look changes nothing and reports nothing new.
    seen.clear()
    await worker.run_once(store, fallback_config(), now=NOW, on_action=on_action)
    assert seen == []


async def test_the_worker_ignores_routes_without_evidence() -> None:
    store = await _store_with(0.97)
    assert await worker.run_once(store, PLAIN_FALLBACK, now=NOW) == []


def test_card_spec_follows_the_route_config() -> None:
    ev = fallback_config(max_looks=3, alpha=0.1, interval="bootstrap").routes["safe"].evidence
    assert ev is not None
    spec = worker.card_spec(ev)
    assert (spec.looks.every, spec.looks.max_looks, spec.looks.alpha) == (100, 3, 0.1)
    assert spec.interval == "bootstrap"


def test_worker_cli_needs_a_database(monkeypatch: pytest.MonkeyPatch) -> None:
    # Never the developer's .env: a database URL there must not reach a unit test.
    monkeypatch.setattr(worker, "Settings", lambda: relay_settings())
    with pytest.raises(SystemExit) as exit_:
        worker.main(["run", "--once"])
    assert exit_.value.code == 2


# ------------------------------------------------------------------ store details
async def test_relabels_are_not_the_gold_label() -> None:
    store = MemoryEvidenceStore()
    pairs = make_pairs(n_judge=0, n_human=1, w=1.0)
    store.pairs = shadow_rows([replace(p, human_label=None, human_ts=None) for p in pairs])
    pid = pairs[0].pair_id
    await store.import_labels(
        [
            HumanLabel(pid, "loss", "a1", True, NOW),  # the blind re-check, earlier
            HumanLabel(pid, "tie", "a1", False, NOW + timedelta(minutes=5)),
            HumanLabel("unknown", "tie", "a1", False, NOW),  # no such pair: skipped
        ]
    )
    [loaded] = await store.load_pairs()
    assert loaded.human_label == "tie"


def test_cards_round_trip_through_json() -> None:
    c = card("safe", recommended_action="none", samples_needed=None)
    assert card_from_dict(c.as_dict()) == c


def test_label_rows_from_the_labeling_tool() -> None:
    rows = [
        {"pair_id": "p1", "label": "loss", "annotator": "a", "is_relabel": False,
         "shown_swapped": True, "ts": "2026-10-28T10:00:00+00:00"},
    ]  # fmt: skip
    [label] = human_labels(rows)
    assert label == HumanLabel("p1", "loss", "a", False, datetime(2026, 10, 28, 10, tzinfo=UTC))


def test_calibration_helper_is_calibrated() -> None:
    assert all(c.calibrated for c in calibration_set().values())
