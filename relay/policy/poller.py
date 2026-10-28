"""Each replica follows the pointer: poll, validate, build, swap (PRD FR-11; ADR 0024).

Every ``poll_s`` seconds (default 5, the PRD's config-propagation interval) the poller reads
the pointer. When it changed, it fetches the bundles it names, builds them (validation
included) and swaps the runtime's snapshot. A rollback is therefore live on every replica
within one poll interval.

**Last-known-good.** If Postgres is unreachable, a bundle is missing, or a bundle fails
validation, nothing changes: the replica keeps serving what it served, and a counter says so.
It retries at the next poll. It never falls back to the YAML policy mid-flight, because that
would silently change policy in the middle of an outage.
"""

import asyncio
import contextlib
import logging

from prometheus_client import Counter

from relay.policy.bundle import BundleError, PolicyBundle
from relay.policy.control import Pointer
from relay.policy.runtime import PolicyRuntime
from relay.policy.store import PolicyStore

logger = logging.getLogger(__name__)

POLL_ERRORS = Counter(
    "relay_policy_poll_errors_total",
    "Policy refreshes that failed; the replica kept its current policy.",
    ["reason"],  # store_unavailable | invalid_bundle
)
SWAPS = Counter("relay_policy_swaps_total", "Policy snapshots swapped in on this replica.")


class PolicyPoller:
    def __init__(self, store: PolicyStore, runtime: PolicyRuntime, *, poll_s: float = 5.0) -> None:
        self._store = store
        self._runtime = runtime
        self._poll_s = poll_s
        self._applied: Pointer | None = None  # the pointer the current snapshot came from
        self._task: asyncio.Task[None] | None = None
        # The first poll has finished (applied, or failed and kept the YAML policy). /readyz
        # answers 503 until then, so a replica that starts during a canary doesn't take traffic
        # before it knows the canary exists; startup itself never waits on the database.
        self.first_attempt_done = False

    @property
    def applied(self) -> Pointer | None:
        return self._applied

    async def refresh(self) -> bool:
        """Apply the current pointer if it changed. True when this replica is up to date."""
        try:
            pointer = await self._store.pointer()
            if pointer == self._applied:
                return True
            active = await self._fetch(pointer.active_version)
            candidate = await self._fetch(pointer.candidate_version)
        except BundleError as e:
            POLL_ERRORS.labels("invalid_bundle").inc()
            logger.error("policy not applied (keeping the current one): %s", e)
            return False
        except Exception as e:  # a down database must never reach a request
            POLL_ERRORS.labels("store_unavailable").inc()
            logger.warning("policy store unavailable (keeping the current policy): %r", e)
            return False
        try:
            self._runtime.apply(
                active,
                candidate,
                mode=pointer.candidate_mode or "active",
                canary_pct=pointer.canary_pct,
            )
        except BundleError as e:
            POLL_ERRORS.labels("invalid_bundle").inc()
            logger.error("policy not applied (keeping the current one): %s", e)
            return False
        self._applied = pointer
        SWAPS.inc()
        return True

    async def _fetch(self, version: str | None) -> PolicyBundle | None:
        if version is None:
            return None
        bundle = await self._store.get(version)
        if bundle is None:
            raise BundleError(f"the pointer names bundle {version!r}, which does not exist")
        return bundle

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="relay-policy-poller")

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
