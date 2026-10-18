"""The router interface (PRD §9.3, FR-7).

A router answers one question: *which model should serve this request?* It does not call the
model. Keeping the decision separate from the call means a decision can be logged, evaluated
offline and A/B-tested (Phases 11-13) without touching provider code.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from relay.config.models import Tier
from relay.schemas.chat import ChatRequest

if TYPE_CHECKING:
    # Only for type hints: RequestContext imports RouteDecision from this module, so a runtime
    # import here would be circular.
    from relay.pipeline.context import RequestContext


@dataclass(frozen=True)
class RouteDecision:
    tier: Tier
    model: str  # a Relay model name, never an upstream ID
    score: float | None  # router confidence; None for rule-based decisions
    reason: str
    policy_version: str


class Router(Protocol):
    async def decide(self, req: ChatRequest, ctx: "RequestContext") -> RouteDecision: ...
