"""RequestContext: the one mutable object that travels through every stage (PRD §9.3).

Stages communicate only through it. For example, RouteStage writes ``decision`` and
ProviderStage reads it. That makes the data flow between stages explicit and testable: build
a context, run one stage, assert on the context.

Fields arrive in the phase that first needs them; ``cache`` (PRD §9.3) comes in Phase 10.

``started_at``, ``received_at``, ``ttft_s``, ``usage``, ``pricing`` and ``cost`` are not in
§9.3. They are what the log row (Phase 2), the cost (Phase 5) and the reservation reconcile
(Phase 6) need, and those run in finalize(), after the response is gone.
"""

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from relay.admission.queue import Slot
from relay.admission.reservation import Reservation
from relay.cache.result import CacheResult
from relay.config.pricing import Cost, Pricing
from relay.config.tenants import Priority, Tenant
from relay.reliability.breaker import Permit
from relay.routing.base import RouteDecision
from relay.schemas.chat import ChatRequest, Usage
from relay.schemas.errors import ErrorClass


@dataclass
class ProviderAttempt:
    """One candidate the dispatcher considered: a call to a provider, or a skip.

    The list on the context is the request's full history: retries, fallback hops, and the
    candidates skipped without a call (paused, breaker open, ...). It becomes the
    ``attempts`` table (PRD §9.4).
    """

    provider: str
    model: str  # Relay name
    latency_s: float  # for a stream: time to the first content chunk
    error_class: ErrorClass | None  # None = success (or a skip: see skipped_reason)
    hop: int = 0  # 0 = the primary; n = the n-th fallback model attempted
    retry: int = 0  # 0 = first try of this model; 1 = its retry
    # Set when the candidate was not called at all, and why (e.g. "paused", "breaker_open").
    skipped_reason: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error_class is None and self.skipped_reason is None


@dataclass
class RequestContext:
    request_id: str
    tenant: Tenant
    request: ChatRequest
    route: str
    priority: Priority
    cohort: str | None  # X-Relay-Cohort after the tenant's allowlist: a tag, "other", or None
    policy_version: str
    # Monotonic clock, not wall clock: time.time() can jump (NTP, DST), and a deadline that
    # moves backwards never fires. Bounds every retry and fallback hop (ADR 0008).
    deadline: float
    stream: bool
    decision: RouteDecision | None = None
    attempts: list[ProviderAttempt] = field(default_factory=list)
    # Two clocks for two jobs. Monotonic for measuring durations: it never jumps. Wall-clock
    # UTC for the timestamp stored in the log row: it means something to humans and joins
    # across machines.
    started_at: float = field(default_factory=time.monotonic)
    received_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    ttft_s: float | None = None  # streams only: when the first content chunk was sent
    usage: Usage | None = None
    # Set by ProviderStage once the model is known; whoever learns the usage (ProviderStage
    # for JSON, the stream body for SSE) turns it into `cost`.
    pricing: Pricing | None = None
    cost: Cost | None = None
    # Phase 6 admission. Each is cleared by the finalize that releases it, so nothing is
    # released twice.
    admission_slot: Slot | None = None
    queue_wait_s: float = 0.0
    reservation: Reservation | None = None
    # Content tokens sent to the client so far (streams; same estimator as reservations).
    # When a stream ends without a usage report (client gone, upstream broke), this is what
    # the reservation is settled against: providers bill tokens generated before an abort.
    forwarded_tokens: int = 0
    # Phase 7. True when a fallback edge without evidence (undecided/unsafe) served the
    # request: X-Relay-Degraded (ADR 0010).
    degraded: bool = False
    # A stream's breaker verdict is only known when the stream ends, so the permit for the
    # serving attempt waits here for ProviderStage.finalize (ADR 0009).
    breaker_permit: Permit | None = None
    # Phase 10. The client asked not to cache (X-Relay-Cache: no-store).
    cache_no_store: bool = False
    cache: CacheResult | None = None
    # The prompt embedding, computed once for the cache and reused by the router (Phase 11).
    embedding: list[float] | None = None
    # What the model said, collected as it is produced, so the cache can store it when the
    # request ends: the JSON reply, or the chunks a stream forwarded.
    reply_parts: list[str] = field(default_factory=list)
    finish_reason: str | None = None

    @property
    def served(self) -> ProviderAttempt | None:
        """The attempt that produced the response, if any did."""
        for attempt in reversed(self.attempts):
            if attempt.succeeded:
                return attempt
        return None

    @property
    def served_by(self) -> tuple[str, str] | None:
        """(provider, model) that answered: an upstream attempt, or the cache."""
        if self.cache is not None and self.cache.hit:
            return ("relay-cache", self.cache.served_model or "")
        attempt = self.served
        return (attempt.provider, attempt.model) if attempt else None

    @property
    def fallback_hops(self) -> int:
        return max((a.hop for a in self.attempts if a.skipped_reason is None), default=0)

    def record_usage(self, usage: Usage) -> None:
        self.usage = usage
        if self.pricing is not None:
            self.cost = self.pricing.cost(usage)
