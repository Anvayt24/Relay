"""Tenants: who is calling, and what they may do (PRD FR-3).

A tenant is resolved from the API key (``relay/auth``) before the pipeline runs, because it
decides the route, the default priority and the cohort allowlist, all of which the
``RequestContext`` needs from the start.

Fields are read by the phase that needs them: ``limits`` and the budget by Phase 6
admission; ``fallback_policy``, ``fallback_on_explicit_model`` and ``deadlines`` by Phase 7
reliability; ``evidence_sampling`` by Phase 12.
"""

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field

from relay.config.models import STRICT

Priority = Literal["interactive", "batch"]


class TenantLimits(BaseModel):
    model_config = STRICT

    rpm: int = Field(gt=0)
    tpm: int = Field(gt=0)
    concurrency: int = Field(default=8, gt=0)


class TenantDeadlines(BaseModel):
    """End-to-end deadlines (PRD FR-5 defaults). Fixed when the request arrives; they bound
    every retry and fallback hop before the first byte (ADR 0008)."""

    model_config = STRICT

    first_token_s: float = Field(default=15.0, gt=0)  # per attempt, to the first content chunk
    interactive_s: float = Field(default=60.0, gt=0)
    batch_s: float = Field(default=120.0, gt=0)

    def total_for(self, priority: Priority) -> float:
        return self.interactive_s if priority == "interactive" else self.batch_s


class TenantCache(BaseModel):
    """Response caching for this tenant (PRD FR-6). Off unless asked for: a cache stores
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

    def allows(self, model: str, provider: str) -> bool:
        model_ok = self.allowed_models == "*" or model in self.allowed_models
        return model_ok and provider in self.allowed_providers

    def cohort(self, raw: str | None) -> str | None:
        if raw is None:
            return None
        return raw if raw in self.cohort_tags else "other"
