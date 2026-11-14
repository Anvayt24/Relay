"""Tenants: who is calling, and what they may do.

A tenant is resolved from the API key (``relay/auth``) before the pipeline runs, because it
decides the route, the default priority and the cohort allowlist, all of which the
``RequestContext`` needs from the start.

Who reads which field: ``limits`` and the budget are read by admission; ``fallback_policy``,
``fallback_on_explicit_model`` and ``deadlines`` by reliability; ``evidence_sampling`` by the
evidence engine.
"""

from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from relay.config.models import STRICT

Priority = Literal["interactive", "batch"]


class TenantLimits(BaseModel):
    model_config = STRICT

    rpm: int = Field(gt=0)
    tpm: int = Field(gt=0)
    concurrency: int = Field(default=8, gt=0)
    # Requests per minute from one client address. Off unless set: behind a proxy
    # every client looks like one address until RELAY_TRUSTED_PROXIES says whom to believe.
    per_ip_rpm: int | None = Field(default=None, gt=0)


class RequestLimits(BaseModel):
    """What one request may ask for. Each limit is off unless set."""

    model_config = STRICT

    # More messages than this is a 400 (param "messages").
    max_messages: int | None = Field(default=None, gt=0)
    # The most output a request may ask for. A larger max_tokens is a 400 (param
    # "max_tokens"), and a request that sets none gets exactly this many: the cap is also the
    # default, whatever the model's own default is.
    max_tokens: int | None = Field(default=None, gt=0)


class TenantDeadlines(BaseModel):
    """End-to-end deadlines. Fixed when the request arrives; they bound
    every retry and fallback hop before the first byte."""

    model_config = STRICT

    first_token_s: float = Field(default=15.0, gt=0)  # per attempt, to the first content chunk
    interactive_s: float = Field(default=60.0, gt=0)
    batch_s: float = Field(default=120.0, gt=0)

    def total_for(self, priority: Priority) -> float:
        return self.interactive_s if priority == "interactive" else self.batch_s


class TenantCache(BaseModel):
    """Response caching for this tenant. Off unless asked for: a cache stores
    model output, so a tenant opts in."""

    model_config = STRICT

    enabled: bool = False
    semantic: bool = False  # also match similar prompts (needs the embedding model + Postgres)
    ttl_s: int = Field(default=3600, gt=0)
    # OpenAI's default temperature is 1, so an unset one is not "deterministic" and is not
    # cached unless the tenant says its traffic is.
    allow_unset_temperature: bool = False
    max_prompt_tokens: int = Field(default=2000, gt=0)


class Tenant(BaseModel):
    model_config = STRICT

    id: str  # the key under `tenants:` in the YAML; filled in by the config loader
    route: str = "default"
    limits: TenantLimits
    request_limits: RequestLimits = Field(default_factory=RequestLimits)
    budget_usd_daily: Decimal = Field(ge=0)
    # hard: requests that would exceed the budget are refused. soft: served, and reported.
    budget_mode: Literal["hard", "soft"] = "hard"
    fallback_policy: Literal["strict", "allow_degraded"] = "strict"
    logging: Literal["metadata_only", "full"] = "metadata_only"
    retention_days: int = Field(default=30, gt=0)
    default_priority: Priority = "interactive"
    evidence_sampling: Literal["off", "on"] = "off"
    # Providers that may receive this tenant's prompts: primary, fallback, shadow and judge.
    allowed_providers: list[str]
    # Relay model names this tenant may call; "*" means any model of an allowed provider.
    allowed_models: list[str] | Literal["*"] = "*"
    # X-Relay-Cohort values kept as-is; anything else becomes "other" (bounded cardinality).
    cohort_tags: list[str] = Field(default_factory=list)
    fallback_on_explicit_model: bool = False
    deadlines: TenantDeadlines = Field(default_factory=TenantDeadlines)
    cache: TenantCache = Field(default_factory=TenantCache)

    @field_validator("evidence_sampling", mode="before")
    @classmethod
    def _yaml_reads_bare_on_and_off_as_booleans(cls, v: Any) -> Any:
        # `evidence_sampling: on` is how a person writes it, and YAML 1.1 turns a bare on/off
        # into True/False.
        return {True: "on", False: "off"}.get(v, v) if isinstance(v, bool) else v

    def allows(self, model: str, provider: str) -> bool:
        model_ok = self.allowed_models == "*" or model in self.allowed_models
        return model_ok and provider in self.allowed_providers

    def cohort(self, raw: str | None) -> str | None:
        if raw is None:
            return None
        return raw if raw in self.cohort_tags else "other"
