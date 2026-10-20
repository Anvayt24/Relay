"""Per-tenant concurrency slots with a weighted-fair priority queue (PRD FR-4).

Each tenant gets ``concurrency`` slots on this replica. A request takes a slot for its whole
life (streams included) and gives it back in finalize. When the slots are full, it waits:

- **Which waiter goes next: stride scheduling** (a simple, exact form of weighted fair
  queueing). Each class has a *pass* value; the waiting class with the lowest pass is served,
  and its pass grows by ``1 / weight``. With weights interactive 4 : batch 1, a backlog is served
  I I I I B I I I I B ... Interactive gets 4/5 of slots under contention, and batch is
  **guaranteed** 1/5: it can never starve, however much interactive traffic arrives.
- **A class that was idle doesn't bank credit:** when it starts waiting again, its pass is
  raised to the current virtual time. Otherwise a class idle for an hour would arrive with an
  hour of "unused share" and monopolize the queue.
- **Bounded waits:** each class has a max wait (2 s interactive, 30 s batch) and a max depth;
  past either, 429. A queue that can grow forever turns an overload into a slow-motion outage.

**Per replica on purpose** (ADR 0007): a shared, cross-replica queue would need polling or
pub/sub to wake waiters. Limits are divided by ``replica_count``, so the fleet total matches
the configured limit.

**The two races, and why they're handled:**
1. The timeout fires just as the slot is granted. Then the waiter *has* the slot and must
   take it, or the slot leaks.
2. The client disconnects (``CancelledError``) while queued, and the slot was granted in the
   same instant. Then the slot must be passed on before the cancellation propagates.
"""

import asyncio
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from prometheus_client import Gauge, Histogram

from relay.config.tenants import Priority

WEIGHTS: dict[Priority, int] = {"interactive": 4, "batch": 1}

QUEUE_DEPTH = Gauge("relay_queue_depth", "Requests waiting for a concurrency slot.", ["priority"])
QUEUE_WAIT = Histogram(
    "relay_queue_wait_seconds",
    "Time spent waiting for a concurrency slot (admitted requests).",
    ["priority"],
    buckets=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)


class QueueFull(Exception):
    pass


class QueueTimeout(Exception):
    pass


class StridePicker:
    """Pure scheduling logic: given which classes are waiting, which goes next. No I/O and no
    asyncio, so property tests can run thousands of scenarios in milliseconds."""

    def __init__(self, weights: dict[Priority, int] = WEIGHTS) -> None:
        self.weights = weights
        self.passes: dict[Priority, float] = dict.fromkeys(weights, 0.0)
        self.virtual_time = 0.0

    def activate(self, cls: Priority) -> None:
        # Called when a class goes from "no waiters" to "has waiters": no banked credit.
        self.passes[cls] = max(self.passes[cls], self.virtual_time)

    def pick(self, waiting: set[Priority]) -> Priority:
        # Lowest pass wins; ties go to the higher weight (interactive), then by name.
        cls = min(waiting, key=lambda c: (self.passes[c], -self.weights[c], c))
        self.virtual_time = self.passes[cls]
        self.passes[cls] += 1 / self.weights[cls]
        return cls


@dataclass
class _TenantQueue:
    active: int = 0
    waiting: dict[Priority, deque[asyncio.Future[None]]] = field(
        default_factory=lambda: {"interactive": deque(), "batch": deque()}
    )
    picker: StridePicker = field(default_factory=StridePicker)


@dataclass(frozen=True)
class Slot:
    tenant: str
    waited_s: float


class AdmissionQueue:
    def __init__(
        self,
        *,
        max_wait_s: dict[Priority, float],
        max_depth: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_wait = max_wait_s
        self._max_depth = max_depth
        self._clock = clock
        self._tenants: dict[str, _TenantQueue] = {}

    def active(self, tenant: str) -> int:
        return self._tenants[tenant].active if tenant in self._tenants else 0

    def waiting(self, tenant: str) -> int:
        q = self._tenants.get(tenant)
        return sum(len(d) for d in q.waiting.values()) if q else 0

    async def acquire(self, tenant: str, *, limit: int, priority: Priority) -> Slot:
        q = self._tenants.setdefault(tenant, _TenantQueue())
        started = self._clock()
        if q.active < limit and not any(q.waiting.values()):
            q.active += 1  # fast path: a free slot and nobody ahead
            return Slot(tenant, 0.0)

        line = q.waiting[priority]
        if len(line) >= self._max_depth:
            raise QueueFull(f"{priority} queue for tenant {tenant!r} is full")
        if not line:
            q.picker.activate(priority)
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        line.append(future)
        QUEUE_DEPTH.labels(priority=priority).inc()
        try:
            async with asyncio.timeout(self._max_wait[priority]):
                await asyncio.shield(future)
        except TimeoutError:
            if not future.done():
                line.remove(future)
                future.cancel()
                raise QueueTimeout(
                    f"no {priority} slot for tenant {tenant!r} within {self._max_wait[priority]}s"
                ) from None
            # Race 1: granted at the deadline. The slot is ours; take it.
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                self._release(q, limit)  # race 2: granted as we were cancelled; pass it on
            elif future in line:
                line.remove(future)
                future.cancel()
            raise
        finally:
            QUEUE_DEPTH.labels(priority=priority).dec()
        waited = self._clock() - started
        QUEUE_WAIT.labels(priority=priority).observe(waited)
        return Slot(tenant, waited)

    def release(self, slot: Slot, *, limit: int) -> None:
        self._release(self._tenants[slot.tenant], limit)

    def _release(self, q: _TenantQueue, limit: int) -> None:
        q.active -= 1
        # Hand freed slots to waiters, in stride order. Skip futures that gave up (timed out
        # or cancelled) but are still at the head of a line.
        while q.active < limit:
            waiting = {c for c, line in q.waiting.items() if line}
            if not waiting:
                return
            line = q.waiting[q.picker.pick(waiting)]
            future = line.popleft()
            if future.done():
                continue
            future.set_result(None)
            q.active += 1
