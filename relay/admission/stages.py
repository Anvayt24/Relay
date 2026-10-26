"""The two admission stages (PRD FR-4: "admission is split into two steps").

    Log → [Admission] → Route → [Reservation] → Provider

**Why two steps:** the price depends on the model, and the model isn't known until the router
has run.
- *Before* routing, admission can only count requests: tenant RPM, plus a concurrency slot
  (waiting in the priority queue if needed).
- *After* routing, reservation knows the model, so it can hold tokens (TPM) and money (USD)
  for the worst case, and settle them against real usage when the request ends.

Both stages do their cleanup in ``finalize``, which runs for every way a request can end:
success, error, a client that disconnects, a stream that breaks. That is what guarantees
"every reservation is reconciled or released" (FR-4 acceptance) and that no slot leaks.
"""

import logging
import math
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

from prometheus_client import Counter

from relay.admission import tokens
from relay.admission.queue import AdmissionQueue, QueueFull, QueueTimeout, Slot
from relay.admission.reservation import Reservation
from relay.admission.store import CapacityStore, StoreUnavailable
from relay.config.pricing import NANO_PER_USD, Pricing
from relay.config.schema import RelayConfig
from relay.pipeline.context import RequestContext
from relay.pipeline.stage import Continue, Outcome, Reject, StageResult
from relay.schemas.chat import Usage
from relay.schemas.errors import ErrorClass, RelayError

logger = logging.getLogger(__name__)

REJECTED = Counter(
    "relay_admission_rejected_total", "Requests refused by admission or reservation.", ["reason"]
)
UNDERESTIMATE = Counter(
    "relay_reservation_underestimate_total",
    "Requests whose real usage exceeded the reserved estimate (spend may overshoot by this).",
    ["provider"],
)
SOFT_OVER_BUDGET = Counter(
    "relay_budget_soft_exceeded_total",
    "Requests served over budget for soft-budget tenants.",
    ["tenant"],
)


def _reject(
    reason: str,
    status: int,
    error_class: ErrorClass,
    message: str,
    retry_after_s: float | None = None,
) -> Reject:
    REJECTED.labels(reason=reason).inc()
    return Reject(RelayError(status, error_class, message, retry_after_s=retry_after_s))


class AdmissionStage:
    name = "admission"

    def __init__(
        self, store: CapacityStore, queue: AdmissionQueue, *, replica_count: int = 1
    ) -> None:
        self._store = store
        self._queue = queue
        self._replicas = max(1, replica_count)

    def _slot_limit(self, ctx: RequestContext) -> int:
        # Concurrency is per replica: each gets its share of the tenant's limit (ADR 0007).
        return max(1, math.ceil(ctx.tenant.limits.concurrency / self._replicas))

    async def process(self, ctx: RequestContext) -> StageResult:
        limits = ctx.tenant.limits
        wait = await self._store.take(
            f"rpm:{ctx.tenant.id}", capacity=limits.rpm, refill_per_s=limits.rpm / 60, amount=1
        )
        if wait > 0:
            return _reject(
                "rpm",
                429,
                ErrorClass.TENANT_RATE_LIMITED,
                f"rate limit of {limits.rpm} requests per minute reached",
                retry_after_s=wait,
            )
        try:
            slot = await self._queue.acquire(
                ctx.tenant.id, limit=self._slot_limit(ctx), priority=ctx.priority
            )
        except QueueFull as e:
            return _reject("queue_full", 429, ErrorClass.QUEUE_FULL, str(e), retry_after_s=1)
        except QueueTimeout as e:
            return _reject("queue_timeout", 429, ErrorClass.QUEUE_TIMEOUT, str(e), retry_after_s=1)
        ctx.admission_slot = slot
        ctx.queue_wait_s = slot.waited_s
        return Continue()

    async def finalize(self, ctx: RequestContext, outcome: Outcome) -> None:
        slot: Slot | None = ctx.admission_slot
        if slot is not None:
            ctx.admission_slot = None
            self._queue.release(slot, limit=self._slot_limit(ctx))


def seconds_to_utc_midnight(now: datetime) -> float:
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return (tomorrow - now).total_seconds()


# Which models a request might be served by: the dispatcher's candidate plan (primary first,
# then fallbacks). Without a planner, only the routed model.
type ModelPlanner = Callable[[RequestContext], Sequence[str]]


class ReservationStage:
    name = "reservation"

    def __init__(
        self, store: CapacityStore, config: RelayConfig, *, planner: ModelPlanner | None = None
    ) -> None:
        self._store = store
        self._config = config
        self._planner = planner

    def _worst_case(self, ctx: RequestContext, prompt: int) -> tuple[int, int]:
        """(tokens, billed nano-USD) to hold: the maximum over every model this request might
        be served by, each at its own output cap and price (FR-4: "size the USD reservation
        for the most expensive model reachable through the route's fallback graph"). The
        settle at the end charges only what the serving model really cost."""
        assert ctx.decision is not None
        names = self._planner(ctx) if self._planner else [ctx.decision.model]
        tokens_needed, nano = 0, 0
        for name in names:
            model = self._config.models[name]
            needed = prompt + tokens.output_cap(ctx.request, model)
            pricing = Pricing(
                model=model, strong=None, price_table_version=self._config.price_table_version
            )
            cost = pricing.cost(
                Usage(prompt_tokens=prompt, completion_tokens=needed - prompt, total_tokens=needed)
            )
            tokens_needed = max(tokens_needed, needed)
            nano = max(nano, cost.billed_nano)
        return tokens_needed, nano

    async def process(self, ctx: RequestContext) -> StageResult:
        decision = ctx.decision
        if decision is None:
            raise RuntimeError("ReservationStage needs a route decision; RouteStage goes first")
        tenant = ctx.tenant
        prompt = tokens.prompt_tokens(ctx.request)
        needed, nano = self._worst_case(ctx, prompt)

        # 1. Tokens per minute. A request bigger than the whole bucket can never fit: that's a
        #    client error (lower max_tokens), not a "wait and retry".
        if needed > tenant.limits.tpm:
            return Reject(
                RelayError(
                    400,
                    ErrorClass.INVALID_REQUEST,
                    f"this request needs about {needed} tokens, above the tenant limit of "
                    f"{tenant.limits.tpm} per minute; lower max_tokens",
                    param="max_tokens",
                )
            )
        tpm_key = f"tpm:{tenant.id}"
        wait = await self._store.take(
            tpm_key, capacity=tenant.limits.tpm, refill_per_s=tenant.limits.tpm / 60, amount=needed
        )
        if wait > 0:
            return _reject(
                "tpm",
                429,
                ErrorClass.TENANT_RATE_LIMITED,
                f"token rate limit of {tenant.limits.tpm} per minute reached",
                retry_after_s=wait,
            )

        # 2. Money: the worst case (estimated prompt + the full output cap) at the billed
        #    price of the most expensive model the request might reach.
        now = datetime.now(UTC)
        budget = f"{tenant.id}:{now:%Y%m%d}"  # a new key each UTC day: the daily reset
        res_id = uuid.uuid4().hex
        held = False
        if nano > 0:
            limit = int(tenant.budget_usd_daily * NANO_PER_USD)
            # Held until the request's deadline (plus a margin), then it expires by itself.
            ttl = max(1.0, ctx.deadline - time.monotonic()) + 60
            try:
                held = await self._store.reserve(
                    budget, res_id, amount=nano, limit=limit, ttl_s=ttl
                )
            except StoreUnavailable:
                if tenant.budget_mode == "hard":
                    await self._store.refund(tpm_key, capacity=tenant.limits.tpm, amount=needed)
                    return _reject(
                        "budget_unavailable",
                        503,
                        ErrorClass.BUDGET_UNAVAILABLE,
                        "budget enforcement is temporarily unavailable; retry shortly",
                        retry_after_s=5,
                    )
                logger.warning(
                    "budget store down: serving soft-budget tenant %r unchecked", tenant.id
                )
            else:
                if not held:
                    if tenant.budget_mode == "hard":
                        await self._store.refund(tpm_key, capacity=tenant.limits.tpm, amount=needed)
                        return _reject(
                            "budget",
                            429,
                            ErrorClass.BUDGET_EXCEEDED,
                            f"daily budget of ${tenant.budget_usd_daily} reached; "
                            "resets at 00:00 UTC",
                            retry_after_s=seconds_to_utc_midnight(now),
                        )
                    SOFT_OVER_BUDGET.labels(tenant=tenant.id).inc()

        ctx.reservation = Reservation(
            res_id=res_id,
            budget=budget,
            tpm_key=tpm_key,
            tokens=needed,
            prompt_estimate=prompt,
            nano=nano if held else 0,
            held=held,
        )
        return Continue()

    async def finalize(self, ctx: RequestContext, outcome: Outcome) -> None:
        r: Reservation | None = ctx.reservation
        if r is None:
            return
        ctx.reservation = None  # settle exactly once, whatever happens below

        # What did this request really use?
        # - usage reported: exact (cost too, computed when the usage arrived);
        # - none, but content was forwarded (client cancel, mid-stream failure): providers bill
        #   tokens generated before the abort, so estimate from what was forwarded;
        # - nothing forwarded (failed before content): nothing billed.
        usage = ctx.usage
        if usage is None and ctx.forwarded_tokens > 0:
            usage = Usage(
                prompt_tokens=r.prompt_estimate,
                completion_tokens=ctx.forwarded_tokens,
                total_tokens=r.prompt_estimate + ctx.forwarded_tokens,
            )
        actual_tokens = usage.total_tokens if usage else 0
        if ctx.cost is not None:
            actual_nano = ctx.cost.billed_nano
        elif usage is not None and ctx.pricing is not None:
            actual_nano = ctx.pricing.cost(usage).billed_nano
        else:
            actual_nano = 0

        if actual_tokens > r.tokens or (r.held and actual_nano > r.nano):
            served = ctx.served
            provider = served.provider if served else "none"
            UNDERESTIMATE.labels(provider=provider).inc()

        tenant = ctx.tenant
        unused = r.tokens - actual_tokens
        if unused > 0:
            await self._store.refund(r.tpm_key, capacity=tenant.limits.tpm, amount=unused)
        if r.held or actual_nano > 0:
            try:
                await self._store.settle(r.budget, r.res_id, actual=actual_nano)
            except StoreUnavailable:
                # The reservation expires by itself; this spend goes unrecorded (bounded).
                logger.warning("could not settle reservation %s: budget store down", r.res_id)
