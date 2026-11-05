"""Request-level Prometheus metrics (PRD FR-8; ADR 0025).

The stage-level metrics live next to the code they count (cache, breakers, queue, retries,
fallbacks, router, shadow, policy, dropped logs). What FR-8 also needs is a view of the
*request*: how many, how they ended, how long they took, the tokens, the money. Those are
recorded **once per request, in** ``LogStage.finalize``, from the same finished context that
becomes the log row: no new code inside any stage, and nothing on the request's path (the
response has already been sent when finalize runs).

**Labels are bounded** (a test enforces an allowlist): tenant, route, provider, model, status,
error class, and small fixed enumerations. Never request IDs, cohorts, sessions or keys: each
distinct value is a new time series, and enough of them take Prometheus down.
"""

from typing import TYPE_CHECKING

from prometheus_client import Counter, Histogram

if TYPE_CHECKING:
    from relay.pipeline.context import RequestContext
    from relay.pipeline.stage import Outcome

# LLM latencies span three orders of magnitude: a cache hit in milliseconds, a long reasoning
# answer in a minute or two.
LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    20.0,
    30.0,
    60.0,
    120.0,
)

REQUESTS = Counter(
    "relay_requests_total",
    "Client requests by how they ended (status) and why (error_class; 'none' on success).",
    ["tenant", "route", "status", "error_class"],
)
DURATION = Histogram(
    "relay_request_duration_seconds",
    "End-to-end request time, including a stream's body until it ended.",
    ["route", "status", "stream"],
    buckets=LATENCY_BUCKETS,
)
TTFT = Histogram(
    "relay_ttft_seconds",
    "Streams: time from arrival to the first content chunk sent to the client.",
    ["route", "model"],
    buckets=LATENCY_BUCKETS,
)
TOKENS = Counter(
    "relay_tokens_total",
    "Tokens served, by direction (in = prompt, out = completion).",
    ["tenant", "provider", "model", "direction"],
)
COST = Counter(
    "relay_cost_nano_usd_total",
    "Cost in nano-USD: billed (what is charged), list (list price), counterfactual_strong "
    "(what the route's strong model would have cost at list price). Cost saved vs always-"
    "strong = counterfactual_strong - list.",
    ["tenant", "provider", "model", "kind"],
)
# Each upstream call is observed where it ends, by the dispatcher (relay_provider_attempt_seconds
# in relay/reliability/dispatch.py): that also covers judge and shadow calls, which have no
# request of their own to be recorded with here.


def record_request(ctx: "RequestContext", outcome: "Outcome", duration_s: float) -> None:
    error = outcome.error_class.value if outcome.error_class else "none"
    REQUESTS.labels(ctx.tenant.id, ctx.route, outcome.status, error).inc()
    DURATION.labels(ctx.route, outcome.status, str(ctx.stream).lower()).observe(duration_s)

    served = ctx.served_by
    provider, model = served if served is not None else ("none", "none")
    if ctx.ttft_s is not None:
        TTFT.labels(ctx.route, model).observe(ctx.ttft_s)
    if ctx.usage is not None and served is not None:
        TOKENS.labels(ctx.tenant.id, provider, model, "in").inc(ctx.usage.prompt_tokens)
        TOKENS.labels(ctx.tenant.id, provider, model, "out").inc(ctx.usage.completion_tokens)
    if ctx.cost is not None and served is not None:
        COST.labels(ctx.tenant.id, provider, model, "billed").inc(ctx.cost.billed_nano)
        COST.labels(ctx.tenant.id, provider, model, "list").inc(ctx.cost.list_nano)
        if ctx.cost.counterfactual_strong_nano is not None:
            COST.labels(ctx.tenant.id, provider, model, "counterfactual_strong").inc(
                ctx.cost.counterfactual_strong_nano
            )
