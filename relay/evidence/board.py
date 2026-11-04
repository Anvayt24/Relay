"""The gateway's view of the route cards: polled, cached, never on the request path.

The worker writes cards to Postgres; every replica reads them here, every ``poll_s`` seconds,
into a dict. Two things read that dict, both in microseconds:

- the **dispatcher**, through the ``EdgeStateSource`` protocol: a gated fallback edge takes
  its state from its card (FR-5's edge-usability table). It replaces ``SeededEdgeStates``,
  which stays as the fallback for an edge that has no card yet (cold start, ADR 0010);
- the **shadow stage**, for its sampling rate: higher while a comparison is undecided.

**Card state → edge state:**

    safe → safe     unsafe → unsafe     undecided | uncalibrated | stale → undecided

The last row is §10.7's "no automatic changes" when the judge is uncalibrated or the
rectifier stale: the edge falls back to the cautious state, which ``strict`` tenants never
use, instead of trusting a number nobody can vouch for.

**Postgres down:** the last cards loaded stay in force, and a counter says so (the PRD's
"replicas keep the last loaded" rule, applied to cards).

**A card the worker stopped updating stops counting.** The worker re-stamps every card on every
run. A card older than ``max_age_s`` means the worker is down, and its state (a ``safe`` edge,
say) may no longer be true: the edge falls back to ``undecided`` and the sampling rate to the
undecided one, and ``relay_card_oldest_age_seconds`` shows it (found in review: a dead worker's
cards stayed in force indefinitely).
"""

import asyncio
import contextlib
import logging
from collections.abc import Callable
from datetime import UTC, datetime

from prometheus_client import Counter, Gauge

from relay.config.schema import EdgeState
from relay.evidence.cards import ALL, RouteCard, State
from relay.evidence.store import EvidenceStore
from relay.reliability.edges import EdgeStateSource

logger = logging.getLogger(__name__)

POLL_ERRORS = Counter(
    "relay_card_poll_errors_total", "Route-card refreshes that failed (last cards kept)."
)
OLDEST_AGE = Gauge(
    "relay_card_oldest_age_seconds", "Age of the oldest route card at the last refresh."
)

_EDGE: dict[State, EdgeState] = {
    "safe": "safe",
    "unsafe": "unsafe",
    "undecided": "undecided",
    "uncalibrated": "undecided",
    "stale": "undecided",
}


def fallback_comparison(source: str, target: str) -> str:
    return f"fallback:{source}->{target}"


class CardBoard:
    def __init__(
        self,
        store: EvidenceStore,
        *,
        seeded: EdgeStateSource,
        poll_s: float = 5.0,
        max_age_s: float = 600.0,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._store = store
        self._seeded = seeded
        self._poll_s = poll_s
        self._max_age_s = max_age_s
        self._now = now
        self._cards: dict[tuple[str, str, str], RouteCard] = {}
        self._task: asyncio.Task[None] | None = None
        self.loaded = False  # at least one refresh succeeded
        # The first refresh has finished, whatever its outcome: /readyz waits for it (a down
        # database must not keep the replica out of service; it then uses the seed states).
        self.first_attempt_done = False

    # ---------------------------------------------------------------- reads (hot path)
    def _stale(self, card: RouteCard) -> bool:
        # The worker stopped re-stamping it: don't trust its state (see the docstring).
        return (self._now() - card.updated_at).total_seconds() > self._max_age_s

    def state(self, route: str, source: str, target: str) -> EdgeState:
        card = self._cards.get((route, fallback_comparison(source, target), ALL))
        if card is None:
            return self._seeded.state(route, source, target)
        if self._stale(card):
            return "undecided"  # cautious, rather than back to its seed state
        return _EDGE[card.state]

    def card_state(self, route: str, comparison: str) -> State | None:
        card = self._cards.get((route, comparison, ALL))
        return card.state if card is not None and not self._stale(card) else None

    def cards(self) -> list[RouteCard]:
        return list(self._cards.values())

    # ---------------------------------------------------------------- refresh
    async def refresh(self) -> bool:
        try:
            cards = await self._store.load_cards()
        except Exception as e:  # a down database must never reach a request
            POLL_ERRORS.inc()
            logger.warning("route cards not refreshed (keeping %d): %r", len(self._cards), e)
            return False
        self._cards = {c.key: c for c in cards}
        self.loaded = True
        if cards:
            oldest = min(c.updated_at for c in cards)
            OLDEST_AGE.set((self._now() - oldest).total_seconds())
        return True

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="relay-card-board")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _run(self) -> None:
        while True:
            await self.refresh()
            self.first_attempt_done = True
            await asyncio.sleep(self._poll_s)
