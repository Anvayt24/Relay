"""The evidence worker: judged pairs in, route cards and actions out (PRD FR-10; ADR 0019).

    python -m relay.evidence.worker run [--once] [--interval 30]   # needs RELAY_DATABASE_URL
    python -m relay.evidence.worker show [--route R] [--json]

A separate process, never the gateway: card computation reads every pair of a route and runs
bootstraps, which has no place next to request handling. The gateway only reads the finished
cards (``board.py``).

Each run:
1. loads every pair with its first human label (pairs without a route, such as the pilot's
   and public calibration sets, calibrate the judge);
2. computes the judge's M7 numbers per judge model and prompt version;
3. computes cards for every route with an ``evidence:`` section, at pre-registered looks,
   given the previous cards (a decided state changes only at a new look);
4. saves them, and reports the **actions** whose card just changed (§10.7):
   ``disable_edge`` takes effect by itself (the card board turns the edge unsafe on every
   replica); ``rollback`` is handed to ``on_action`` (Phase 13 rolls the canary back);
   promotions are only ever recommended (ADR 0022).
"""

import argparse
import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from prometheus_client import Counter

from relay.config.loader import load_config
from relay.config.schema import EvidenceSpec, RelayConfig
from relay.evidence.cards import (
    ALL,
    CardSpec,
    RouteCard,
    calibration_by_judge,
    compute_cards,
    render_markdown,
)
from relay.evidence.looks import LookSchedule
from relay.evidence.store import EvidenceStore, PostgresEvidenceStore
from relay.policy.auto import rollback_unsafe_canary
from relay.policy.store import PostgresPolicyStore
from relay.settings import Settings

logger = logging.getLogger(__name__)

ACTIONS = Counter(
    "relay_evidence_actions_total",
    "Card actions taken or recommended when a card changed (§10.7).",
    ["route", "action"],
)
# Actions the worker reports; "none" and "collect_more" are the quiet states.
LOUD = frozenset({"disable_edge", "rollback", "raise_threshold", "promote", "do_not_promote"})

type ActionHook = Callable[[RouteCard], Awaitable[None]]


def card_spec(ev: EvidenceSpec) -> CardSpec:
    return CardSpec(
        floor=ev.floor,
        looks=LookSchedule(every=ev.look_every, max_looks=ev.max_looks, alpha=ev.alpha),
        min_cohort_n=ev.min_cohort_n,
        rectifier_max_age=timedelta(days=ev.rectifier_max_age_days),
        interval=ev.interval,
    )


async def run_once(
    store: EvidenceStore,
    config: RelayConfig,
    *,
    now: datetime | None = None,
    on_action: ActionHook | None = None,
) -> list[RouteCard]:
    now = now or datetime.now(UTC)
    pairs = await store.load_pairs()
    calibration = calibration_by_judge(pairs)
    previous = {c.key: c for c in await store.load_cards()}
    cards: list[RouteCard] = []
    for name, route in sorted(config.routes.items()):
        if route.evidence is None:
            continue
        mine = [p for p in pairs if p.route == name]
        cards += compute_cards(
            mine, card_spec(route.evidence), calibration=calibration, now=now, previous=previous
        )
    await store.save_cards(cards)

    for card in cards:
        prev = previous.get(card.key)
        changed = prev is None or (prev.state, prev.look) != (card.state, card.look)
        if card.cohort != ALL or not changed or card.recommended_action not in LOUD:
            continue
        ACTIONS.labels(card.route, card.recommended_action).inc()
        logger.warning(
            "route %s %s: %s (w=%s, ci=%s) -> %s",
            card.route,
            card.comparison,
            card.state,
            None if card.estimate is None else round(card.estimate, 4),
            card.ci,
            card.recommended_action,
        )
        if on_action is not None:
            await on_action(card)
    return cards


async def run_forever(
    store: EvidenceStore,
    config: RelayConfig,
    *,
    interval_s: float,
    on_action: ActionHook | None = None,
) -> None:
    while True:
        try:
            await run_once(store, config, on_action=on_action)
        except Exception:  # a bad run (database down) must not end the worker
            logger.exception("evidence run failed; retrying next interval")
        await asyncio.sleep(interval_s)


async def _main(args: argparse.Namespace, settings: Settings) -> int:
    if not settings.database_url:
        print("RELAY_DATABASE_URL is not set: the evidence worker reads Postgres")
        return 2
    config = load_config(args.config or settings.config_path)
    store = PostgresEvidenceStore(settings.database_url)
    policies = PostgresPolicyStore(settings.database_url)
    await store.start()
    await policies.start()
    # The one automatic policy action: an unsafe canary is rolled back (§10.7, ADR 0024).
    on_action = rollback_unsafe_canary(policies)
    try:
        if args.command == "show":
            cards = [c for c in await store.load_cards() if not args.route or c.route == args.route]
            cards.sort(key=lambda c: (c.route, c.comparison, c.cohort != ALL, c.cohort))
            if args.json:
                print(json.dumps([c.as_dict() for c in cards], indent=2))
            else:
                print(render_markdown(cards))
            return 0
        if args.once:
            cards = await run_once(store, config, on_action=on_action)
            print(render_markdown(cards))
            return 0
        await run_forever(store, config, interval_s=args.interval, on_action=on_action)
        return 0
    finally:
        await policies.close()
        await store.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m relay.evidence.worker", description=__doc__)
    parser.add_argument("--config", help="config file (default: RELAY_CONFIG_PATH or built-in)")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="compute cards (repeatedly, or once)")
    run.add_argument("--once", action="store_true")
    run.add_argument("--interval", type=float, default=30.0, help="seconds between runs")
    show = sub.add_parser("show", help="print the current cards")
    show.add_argument("--route")
    show.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(asyncio.run(_main(args, Settings())))


if __name__ == "__main__":
    main()
