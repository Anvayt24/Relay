"""Tenants: who is calling, and what they may do (PRD FR-3).

A tenant is resolved from the API key (``relay/auth``) before the pipeline runs, because it
decides the route, the default priority and the cohort allowlist, all of which the
``RequestContext`` needs from the start.

Some fields are read by later phases: ``limits`` and the budget by Phase 6 admission,
``fallback_policy`` and ``fallback_on_explicit_model`` by Phase 7, ``evidence_sampling`` by
Phase 12. They are declared now so the config format is stable and validated from day one.
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

    def allows(self, model: str, provider: str) -> bool:
        model_ok = self.allowed_models == "*" or model in self.allowed_models
        return model_ok and provider in self.allowed_providers

    def cohort(self, raw: str | None) -> str | None:
        if raw is None:
            return None
        return raw if raw in self.cohort_tags else "other"
