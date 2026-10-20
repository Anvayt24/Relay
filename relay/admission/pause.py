"""The shared provider pause (PRD FR-4: "shared capacity state").

When a provider answers 429, every replica should stop calling that limit, not just the one
that got the 429. Otherwise N replicas each discover the limit separately, and the provider
sees N times the traffic it just refused. So a 429 writes a pause key, with a TTL, that all
replicas check before calling.

**Scope: exactly the limit that was hit, never wider** (FR-4).
- The limit's scope comes from the 429's signal, or else from the provider's config.
- ``model`` scope pauses only ``provider + model``: Groq limits each model separately, so a
  429 on one model must not idle the others.
- ``account`` scope pauses every model of the provider: OpenRouter's ``:free`` daily cap is
  shared across models, so trying another one would just get another 429.

**Duration:** the provider's ``Retry-After`` if it sent one (1 s otherwise; 1 h for a quota
with no reset given), then x U(1.0, 1.2). That **jitter** is the point: without it, every
replica resumes at the same instant, and a synchronized burst arrives exactly when the limit
reopens, the "thundering herd" that triggers the next 429 (FR-4: "resume uses jitter").
"""

import random

from prometheus_client import Counter

from relay.admission.store import CapacityStore
from relay.config.schema import RelayConfig
from relay.schemas.errors import ErrorClass, ProviderError

PAUSED = Counter(
    "relay_provider_paused_total", "Pauses written after an upstream 429.", ["provider", "scope"]
)

DEFAULT_RATE_LIMIT_PAUSE_S = 1.0
DEFAULT_QUOTA_PAUSE_S = 3600.0
JITTER = (1.0, 1.2)


def _account_key(provider: str) -> str:
    return f"pause:{provider}"


def _model_key(provider: str, upstream_model: str) -> str:
    return f"pause:{provider}:{upstream_model}"


class PauseState:
    def __init__(
        self, store: CapacityStore, config: RelayConfig, rng: random.Random | None = None
    ) -> None:
        self._store = store
        self._config = config
        self._rng = rng or random.Random()

    async def paused_for(self, provider: str, upstream_model: str) -> float:
        """Seconds until this model may be called again (0 = go). Checks both scopes, since
        either kind of limit blocks the call."""
        account = await self._store.paused_for(_account_key(provider))
        model = await self._store.paused_for(_model_key(provider, upstream_model))
        return max(account, model)

    async def record(self, provider: str, upstream_model: str, err: ProviderError) -> None:
        if err.error_class not in (ErrorClass.RATE_LIMITED, ErrorClass.QUOTA_EXHAUSTED):
            return  # only limits pause; 5xx are the circuit breaker's job (Phase 7)
        signal = err.signal
        scope = (signal.limit_scope if signal else None) or self._config.providers[
            provider
        ].limit_scope
        base = signal.retry_after_s if signal and signal.retry_after_s else None
        if base is None:
            base = (
                DEFAULT_QUOTA_PAUSE_S
                if err.error_class is ErrorClass.QUOTA_EXHAUSTED
                else DEFAULT_RATE_LIMIT_PAUSE_S
            )
        ttl = base * self._rng.uniform(*JITTER)
        key = _account_key(provider) if scope == "account" else _model_key(provider, upstream_model)
        await self._store.pause(key, ttl)
        PAUSED.labels(provider=provider, scope=scope).inc()
