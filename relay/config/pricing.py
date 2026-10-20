"""Request cost (PRD FR-9): integer **nano-USD**, computed from provider-reported usage.

Why integers: money that is summed millions of times must not drift. In floats,
``0.1 + 0.2 != 0.3``; summed over a day of requests, a budget check can be off in either
direction. Integers add exactly. Nano (1e-9 USD) is fine enough that one token of the cheapest
model is still a whole number, and a 64-bit integer still holds billions of dollars.

Prices are USD per 1M tokens, so per token in nano-USD: ``price * 1e9 / 1e6 = price * 1000``.
Each request's total is rounded *up* once, at the end: rounding each component would compound
the error, and rounding down would let tiny requests be free.

Three costs per request:
- ``list``: usage times list prices. Comparable across free and paid tiers ("cost saved").
- ``billed``: what is actually charged: list, or 0 for a ``billing: free`` model.
- ``counterfactual_strong``: the same usage priced on the route's strong model, an estimate
  of "what always-strong would have cost". The strong model's real output length is unknown,
  so this assumes the same length (ratio 1.0, FR-9) and is labelled an estimate wherever shown.
"""

import math
from dataclasses import dataclass
from decimal import Decimal

from relay.config.models import ModelSpec
from relay.schemas.chat import Usage

NANO_PER_USD = 10**9


@dataclass(frozen=True)
class Cost:
    list_nano: int
    billed_nano: int
    counterfactual_strong_nano: int | None


def list_cost_nano(model: ModelSpec, usage: Usage) -> int:
    cached = usage.prompt_tokens_details.cached_tokens if usage.prompt_tokens_details else 0
    written = usage.cache_write_tokens
    uncached = max(0, usage.prompt_tokens - cached - written)
    price_cached = model.price_cached_in if model.price_cached_in is not None else model.price_in
    price_write = model.price_cache_write if model.price_cache_write is not None else model.price_in
    usd_times_million = (
        uncached * model.price_in
        + cached * price_cached
        + written * price_write
        + usage.completion_tokens * model.price_out
    )
    return math.ceil(usd_times_million * 1000)


@dataclass(frozen=True)
class Pricing:
    """Everything needed to price one request, captured when the model is chosen.

    The stream's usage arrives long after ``ProviderStage`` returned, so the response body
    prices it with this object instead of looking the model up again.
    """

    model: ModelSpec
    strong: ModelSpec | None
    price_table_version: str

    def cost(self, usage: Usage) -> Cost:
        list_nano = list_cost_nano(self.model, usage)
        return Cost(
            list_nano=list_nano,
            billed_nano=0 if self.model.billing == "free" else list_nano,
            counterfactual_strong_nano=list_cost_nano(self.strong, usage) if self.strong else None,
        )


def usd(nano: int) -> str:
    """Exact decimal text for headers and JSON: Decimal division, no binary rounding."""
    return str(Decimal(nano) / NANO_PER_USD)
