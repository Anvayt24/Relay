"""RequestContext: the one mutable object that travels through every stage (PRD §9.3).

Stages communicate only through it. For example, RouteStage writes ``decision`` and
ProviderStage reads it. That makes the data flow between stages explicit and testable: build
a context, run one stage, assert on the context.

Fields arrive in the phase that first needs them; the rest of PRD §9.3 is still to come:
- ``reservation``: Phase 6;
- ``degraded``: Phase 7;
- ``cache``: Phase 10.

``started_at``, ``received_at``, ``ttft_s`` and ``usage`` are not in §9.3. They are what the
log row (Phase 2) and later the reservation reconcile (Phase 6) need, and both run in
finalize(), after the response is gone.
"""

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from relay.config.tenants import Tenant
from relay.routing.base import RouteDecision
from relay.schemas.chat import ChatRequest, Usage
from relay.schemas.errors import ErrorClass

Priority = Literal["interactive", "batch"]


@dataclass
class ProviderAttempt:
    """One call to one provider. Phase 7 adds retries and fallback hops, so there can be
    several per request; the list is the request's full history."""

    provider: str
    model: str  # Relay name
    latency_s: float  # for a stream: time to the first content chunk
    error_class: ErrorClass | None  # None = success


@dataclass
class RequestContext:
    request_id: str
    tenant: Tenant
    request: ChatRequest
    route: str
    priority: Priority
    cohort: str | None  # raw X-Relay-Cohort; the allowlist/cardinality cap arrives in Phase 5
    policy_version: str
    # Monotonic clock, not wall clock: time.time() can jump (NTP, DST), and a deadline that
    # moves backwards never fires. Set now, enforced from Phase 7 (retries/fallback).
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
