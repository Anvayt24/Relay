"""Providers and models, as declared in the YAML config (PRD §9.5).

Relay names and upstream IDs are kept apart on purpose:
- clients only ever see Relay names;
- adapters only ever see ``provider_model``.

That lets an upstream rename (common with free tiers) be a one-line config change instead of a
breaking change for every client.

Prices are **list** prices in USD per 1M tokens, even for models used on a free tier (PRD
FR-9): "cost saved" must compare list prices, or every saving on a free tier reads as $0.
``billing`` says what is actually charged. ``Decimal``, not ``float``: 0.075 must stay exactly
0.075, because these numbers end up in money.
"""

from collections.abc import Mapping
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Tier = Literal["cheap", "strong"]
LimitScope = Literal["model", "account"]

# Every config model: frozen (config never changes at runtime) and extra="forbid" (a typo
# such as `prise_in` is an error at startup, not a silently ignored field).
STRICT = ConfigDict(frozen=True, extra="forbid")


class ProviderSpec(BaseModel):
    model_config = STRICT

    # "openai": speaks OpenAI's chat-completions format (Groq, OpenRouter, Gemini's
    # compatibility endpoint). "anthropic": Anthropic's Messages API. "mock": in-process.
    kind: Literal["openai", "anthropic", "mock"]
    base_url: str = ""
    # Where this provider enforces rate limits (FR-4): per model, or per account across all
    # models. Phase 6 pauses exactly this scope after a 429.
    limit_scope: LimitScope = "model"
    # The env var holding the key. Never the key itself: config may be committed, keys never.
    api_key_env: str = ""
    # The request field for the output cap. Groq documents max_completion_tokens and treats
    # max_tokens as deprecated; the others take max_tokens.
    max_tokens_param: str = "max_tokens"
    # Optional provider-wide limits Relay reserves against (Phase 6), to stay under the
    # provider's own 429s instead of discovering them.
    rpm: int | None = Field(default=None, gt=0)
    tpm: int | None = Field(default=None, gt=0)


class ModelSpec(BaseModel):
    model_config = STRICT

    provider: str
    provider_model: str
    tier: Tier
    price_in: Decimal = Field(ge=0)  # USD per 1M uncached input tokens
    price_out: Decimal = Field(ge=0)  # USD per 1M output tokens
    price_cached_in: Decimal | None = Field(default=None, ge=0)  # cache hits; default price_in
    price_cache_write: Decimal | None = Field(default=None, ge=0)  # default price_in
    # "list": billed at the prices above. "free": a free tier, billed $0 (list cost is still
    # computed and logged).
    billing: Literal["list", "free"] = "list"
    # The output cap sent upstream when the client sets none. Anthropic requires one, and
    # Phase 6 reserves budget against it, so every request gets one.
    default_max_tokens: int = Field(default=1024, gt=0)


def available_models(
    models: Mapping[str, ModelSpec], providers: Mapping[str, object]
) -> dict[str, ModelSpec]:
    """The models whose provider has a registered adapter (i.e. whose key is set)."""
    return {name: spec for name, spec in models.items() if spec.provider in providers}
