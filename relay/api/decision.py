"""What Relay decided and why, as response metadata.

One function builds the fields, so a JSON response's headers and a stream's final chunk can
never disagree about what they call things. A field Relay does not know is left out, never
sent as an empty or "None" value.
"""

import time

from relay.config.pricing import usd
from relay.pipeline.context import RequestContext

# The response header carrying each field.
DECISION_HEADERS = {
    "router_score": "X-Relay-Router-Score",
    "router_reason": "X-Relay-Router-Reason",
    "strong_cost_usd": "X-Relay-Strong-Cost-USD",
    "fallback_hops": "X-Relay-Fallback-Hops",
    "latency_ms": "X-Relay-Latency-Ms",
}


def decision_fields(ctx: RequestContext) -> dict[str, str]:
    fields: dict[str, str] = {}
    if ctx.decision is not None and ctx.decision.score is not None:
        # The trained router's estimate that the cheap model's answer is good enough.
        fields["router_score"] = f"{ctx.decision.score:.4f}"
    if ctx.decision is not None and ctx.decision.reason.startswith("router"):
        # "router" when it decided, "router_fallback:<cause>" when it could not and the strong
        # model answered instead (so an absent score is explained, not a mystery).
        fields["router_reason"] = ctx.decision.reason
    if ctx.cost is not None and ctx.cost.counterfactual_strong_nano is not None:
        # The same usage at the route's strong model's list price: "cost saved" is this minus
        # what was spent (list price, so a free tier does not read as $0 saved).
        fields["strong_cost_usd"] = usd(ctx.cost.counterfactual_strong_nano)
    fields["fallback_hops"] = str(ctx.fallback_hops)
    fields["latency_ms"] = str(round((time.monotonic() - ctx.started_at) * 1000))
    return fields
