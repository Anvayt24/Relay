"""Where evidence lives: judged pairs, human labels and route cards (PRD §9.4).

One protocol, two implementations, like the other stores:
- ``MemoryEvidenceStore``: tests, and a gateway with no database (pairs then live only as long
  as the process, which is fine for a demo and useless for decisions);
- ``PostgresEvidenceStore``: the ``eval_pairs``, ``human_labels`` and ``route_cards`` tables.

The gateway only **writes pairs** (from the shadow stage's background tasks) and **reads
cards** (the card board, every few seconds). The evidence worker reads pairs and labels and
writes cards. Neither ever runs on a request's path.

The pool is created with ``min_size=0`` (as for the log sink): a database that is down at
startup costs nothing until the first use, and every caller treats a failure as "no evidence
right now", never as a request failure.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Protocol

import asyncpg

from relay.evidence.cards import JudgedPair, RouteCard


@dataclass(frozen=True)
class ShadowPair:
    """One ``eval_pairs`` row written by the shadow stage."""

    pair_id: str
    request_id: str
    route: str
    prompt_hash: str
    cohort: str
    comparison: str
    candidate_model: str
    reference_model: str
    judge_label: str | None
    judge_model: str
    judge_prompt_version: str
    position_swapped: bool
    text_retained: bool
    ts: datetime
    source: str = "shadow"
    policy_version: str | None = None
    prompt_text: str | None = None  # only for `full`-logging tenants
    candidate_text: str | None = None
    reference_text: str | None = None
    cost_nano_usd: int = 0

    def judged(self) -> JudgedPair:
        return JudgedPair(
            pair_id=self.pair_id,
            route=self.route,
            comparison=self.comparison,
            cohort=self.cohort,
            judge_label=self.judge_label,
            judge_model=self.judge_model,
            judge_prompt_version=self.judge_prompt_version,
            ts=self.ts,
            policy_version=self.policy_version,
        )


@dataclass(frozen=True)
class HumanLabel:
    pair_id: str
    label: str  # win | tie | loss
    annotator: str
    is_relabel: bool
    ts: datetime


class EvidenceStore(Protocol):
    async def write_pairs(self, rows: Sequence[ShadowPair]) -> None: ...
    async def load_pairs(self) -> list[JudgedPair]: ...
    async def import_labels(self, labels: Sequence[HumanLabel]) -> int: ...
    async def load_cards(self) -> list[RouteCard]: ...
    async def save_cards(self, cards: Sequence[RouteCard]) -> None: ...
    async def start(self) -> None: ...
    async def close(self) -> None: ...


class MemoryEvidenceStore:
    def __init__(self) -> None:
        self.pairs: list[ShadowPair] = []
        self.extra: list[JudgedPair] = []  # pairs from elsewhere (calibration sets), for tests
        self.labels: list[HumanLabel] = []
        self.cards: dict[tuple[str, str, str], RouteCard] = {}

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def write_pairs(self, rows: Sequence[ShadowPair]) -> None:
        self.pairs.extend(rows)

    async def load_pairs(self) -> list[JudgedPair]:
        first: dict[str, HumanLabel] = {}
        for lab in sorted(self.labels, key=lambda lab: lab.ts):
            if not lab.is_relabel:
                first.setdefault(lab.pair_id, lab)
        out = []
        for p in [*(row.judged() for row in self.pairs), *self.extra]:
            human = first.get(p.pair_id)
            if human is not None and p.human_label is None:
                p = JudgedPair(**{**asdict(p), "human_label": human.label, "human_ts": human.ts})
            out.append(p)
        return out

    async def import_labels(self, labels: Sequence[HumanLabel]) -> int:
        known = {p.pair_id for p in self.pairs} | {p.pair_id for p in self.extra}
        new = [lab for lab in labels if lab.pair_id in known]
        self.labels.extend(new)
        return len(new)

    async def load_cards(self) -> list[RouteCard]:
        return list(self.cards.values())

    async def save_cards(self, cards: Sequence[RouteCard]) -> None:
        for card in cards:
            self.cards[card.key] = card


_PAIR_COLUMNS = [
    "pair_id",
    "request_id",
    "route",
    "prompt_hash",
    "cohort",
    "comparison",
    "candidate_model",
    "reference_model",
    "judge_label",
    "judge_model",
    "judge_prompt_version",
    "position_swapped",
    "text_retained",
    "ts",
    "source",
    "policy_version",
    "prompt_text",
    "candidate_text",
    "reference_text",
    "cost_nano_usd",
]

# The first non-relabel human label of each pair, joined on. Pairs without a route (pilot and
# public calibration sets) are loaded too: they calibrate the judge (cards.calibration_by_judge).
_LOAD_PAIRS = """
SELECT p.pair_id, coalesce(p.route, '') AS route, p.comparison, p.cohort, p.judge_label,
       p.judge_model, p.judge_prompt_version, p.ts, p.policy_version,
       h.label AS human_label, h.ts AS human_ts
FROM eval_pairs p
LEFT JOIN LATERAL (
    SELECT label, ts FROM human_labels
    WHERE pair_id = p.pair_id AND NOT is_relabel
    ORDER BY ts LIMIT 1
) h ON true
ORDER BY p.ts
"""

_UPSERT_CARD = """
INSERT INTO route_cards (route, comparison, cohort, state, estimate, ci_low, ci_high, floor,
                         n_judge, n_human, look, recommended_action, policy_version, card,
                         updated_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb, $15)
ON CONFLICT (route, comparison, cohort) DO UPDATE SET
    state = EXCLUDED.state, estimate = EXCLUDED.estimate, ci_low = EXCLUDED.ci_low,
    ci_high = EXCLUDED.ci_high, floor = EXCLUDED.floor, n_judge = EXCLUDED.n_judge,
    n_human = EXCLUDED.n_human, look = EXCLUDED.look,
    recommended_action = EXCLUDED.recommended_action, policy_version = EXCLUDED.policy_version,
    card = EXCLUDED.card, updated_at = EXCLUDED.updated_at
"""


class PostgresEvidenceStore:
    def __init__(self, dsn: str, *, command_timeout_s: float = 10.0) -> None:
        self._dsn = dsn
        self._command_timeout_s = command_timeout_s
        self._pool: asyncpg.Pool | None = None

    async def start(self) -> None:
        self._pool = await asyncpg.create_pool(
            self._dsn, min_size=0, max_size=2, command_timeout=self._command_timeout_s
        )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()

    def _need_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgresEvidenceStore used before start()")
        return self._pool

    async def write_pairs(self, rows: Sequence[ShadowPair]) -> None:
        records = [tuple(getattr(r, c) for c in _PAIR_COLUMNS) for r in rows]
        async with self._need_pool().acquire() as conn:
            await conn.copy_records_to_table("eval_pairs", records=records, columns=_PAIR_COLUMNS)

    async def load_pairs(self) -> list[JudgedPair]:
        async with self._need_pool().acquire() as conn:
            rows = await conn.fetch(_LOAD_PAIRS)
        return [JudgedPair(**dict(r)) for r in rows]

    async def import_labels(self, labels: Sequence[HumanLabel]) -> int:
        # Labels for pairs that aren't in the database are skipped (the foreign key would
        # reject them anyway); re-importing the same file adds nothing twice.
        async with self._need_pool().acquire() as conn, conn.transaction():
            known = {
                r["pair_id"]
                for r in await conn.fetch(
                    "SELECT pair_id FROM eval_pairs WHERE pair_id = ANY($1::text[])",
                    [lab.pair_id for lab in labels],
                )
            }
            existing = {
                (r["pair_id"], r["annotator"], r["is_relabel"])
                for r in await conn.fetch(
                    "SELECT pair_id, annotator, is_relabel FROM human_labels "
                    "WHERE pair_id = ANY($1::text[])",
                    list(known),
                )
            }
            new = [
                lab
                for lab in labels
                if lab.pair_id in known
                and (lab.pair_id, lab.annotator, lab.is_relabel) not in existing
            ]
            await conn.executemany(
                "INSERT INTO human_labels (pair_id, label, annotator, is_relabel, ts) "
                "VALUES ($1, $2, $3, $4, $5)",
                [(lab.pair_id, lab.label, lab.annotator, lab.is_relabel, lab.ts) for lab in new],
            )
        return len(new)

    async def load_cards(self) -> list[RouteCard]:
        async with self._need_pool().acquire() as conn:
            rows = await conn.fetch("SELECT card FROM route_cards")
        return [card_from_dict(json.loads(r["card"])) for r in rows]

    async def save_cards(self, cards: Sequence[RouteCard]) -> None:
        async with self._need_pool().acquire() as conn, conn.transaction():
            await conn.executemany(
                _UPSERT_CARD,
                [
                    (
                        c.route,
                        c.comparison,
                        c.cohort,
                        c.state,
                        c.estimate,
                        c.ci[0] if c.ci else None,
                        c.ci[1] if c.ci else None,
                        c.floor,
                        c.n_judge,
                        c.n_human,
                        c.look,
                        c.recommended_action,
                        c.policy_version,
                        json.dumps(c.as_dict()),
                        c.updated_at,
                    )
                    for c in cards
                ],
            )


def card_from_dict(data: dict[str, Any]) -> RouteCard:
    data = dict(data)
    data["ci"] = tuple(data["ci"]) if data.get("ci") else None
    data["updated_at"] = datetime.fromisoformat(data["updated_at"])
    return RouteCard(**data)
