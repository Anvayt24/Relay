"""Error taxonomy (PRD Appendix B) and the two exception types that carry it.

There are two kinds of error, raised at different layers:

- ``ProviderError``: raised by provider adapters. It describes what went wrong *upstream*, in
  Relay's vocabulary (``ErrorClass``), so nothing above the adapter ever parses a provider's
  error body. Later phases decide retries, breakers and fallback from ``error_class``.
- ``RelayError``: what the *client* will be told. It carries the HTTP status. The API layer
  turns it into the OpenAI error envelope, and it is the only error type it does that for.

``RelayError.from_provider`` is the single place where an upstream failure becomes a
client-facing status. That mapping is a product decision, not a pass-through: see the
comments in ``UPSTREAM_STATUS``.
"""

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Self


class ErrorClass(StrEnum):
    # Provider-side classes (PRD Appendix B). Retry/breaker rules arrive in Phase 7.
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    OVERLOADED = "overloaded"
    TIMEOUT = "timeout"
    SERVER_ERROR = "server_error"
    BAD_REQUEST = "bad_request"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    MODEL_NOT_FOUND = "model_not_found"
    AUTH = "auth"
    CONTENT_FILTERED = "content_filtered"
    STREAM_BROKEN = "stream_broken"

    # Gateway-side classes: Relay itself refuses or fails the request.
    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_PARAMETER = "unsupported_parameter"
    INTERNAL_ERROR = "internal_error"
    INVALID_API_KEY = "invalid_api_key"  # 401: OpenAI's own code for a bad key
    AUTH_UNAVAILABLE = "auth_unavailable"  # 503: the key store can't answer right now
    # Admission (Phase 6): Relay itself refuses, before or instead of calling a provider.
    TENANT_RATE_LIMITED = "tenant_rate_limited"  # 429: the tenant's RPM/TPM
    QUEUE_FULL = "queue_full"  # 429: too many requests already waiting
    QUEUE_TIMEOUT = "queue_timeout"  # 429: waited the class's max for a slot
    PROVIDER_PAUSED = "provider_paused"  # 429: the provider said "slow down"; all replicas wait
    BUDGET_EXCEEDED = "budget_exceeded"  # 429: the tenant's daily budget is spent
    BUDGET_UNAVAILABLE = "budget_unavailable"  # 503: can't check the budget (Redis down)


# What the client sees when a provider fails with a given class. The guiding question is:
# "whose fault is this, and can the client do anything about it?"
UPSTREAM_STATUS: dict[ErrorClass, int] = {
    # Passed through: the client hit a limit and should slow down.
    ErrorClass.RATE_LIMITED: 429,
    ErrorClass.QUOTA_EXHAUSTED: 429,
    # The provider is unhealthy. 503 = "try later", 504 = "upstream took too long",
    # 502 = "upstream gave a bad answer". None of these are the client's fault.
    ErrorClass.OVERLOADED: 503,
    ErrorClass.TIMEOUT: 504,
    ErrorClass.SERVER_ERROR: 502,
    ErrorClass.STREAM_BROKEN: 502,
    # The request itself is the problem; resending it unchanged cannot succeed.
    ErrorClass.BAD_REQUEST: 400,
    ErrorClass.CONTEXT_LENGTH_EXCEEDED: 400,
    ErrorClass.CONTENT_FILTERED: 400,
    # Relay's *own* setup is broken. A 401 here would tell the client its key is bad (it isn't:
    # Relay's key to the provider is), and a 404 would say the client asked for a model that
    # doesn't exist (it asked for one Relay advertises). Both are Relay's fault: 502.
    ErrorClass.AUTH: 502,
    ErrorClass.MODEL_NOT_FOUND: 502,
}


# Errors a client *should* retry, after Retry-After: Relay refused for load reasons and has not
# retried anything itself. Every other error says x-should-retry: false, because Relay is the
# single retry owner (FR-1) or because retrying cannot help (budget spent until midnight).
CLIENT_RETRYABLE: frozenset[ErrorClass] = frozenset(
    {
        ErrorClass.TENANT_RATE_LIMITED,
        ErrorClass.QUEUE_FULL,
        ErrorClass.QUEUE_TIMEOUT,
        ErrorClass.PROVIDER_PAUSED,
    }
)


@dataclass(frozen=True)
class ProviderSignal:
    """Rate-limit hints from a provider response. Phase 4 adapters fill this in from headers
    such as ``Retry-After``; Phase 6 admission reads it to pause the right scope."""

    retry_after_s: float | None = None
    # Providers enforce limits per model or per account/key; pausing the wrong scope either
    # keeps hammering a limit or idles healthy models (PRD FR-2, FR-4).
    limit_scope: Literal["model", "account"] | None = None
    # What the provider says is left in the current window (x-ratelimit-remaining-*), when it
    # says. Admission (Phase 6) can slow down *before* the 429 instead of after it.
    remaining_requests: int | None = None
    remaining_tokens: int | None = None


class ProviderError(Exception):
    def __init__(
        self, error_class: ErrorClass, message: str, signal: ProviderSignal | None = None
    ) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.message = message
        self.signal = signal


class RelayError(Exception):
    def __init__(
        self,
        status: int,
        error_class: ErrorClass,
        message: str,
        param: str | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.error_class = error_class
        self.message = message
        self.param = param
        self.retry_after_s = retry_after_s

    @classmethod
    def from_provider(cls, err: ProviderError) -> Self:
        retry_after = err.signal.retry_after_s if err.signal else None
        return cls(
            status=UPSTREAM_STATUS[err.error_class],
            error_class=err.error_class,
            message=f"upstream provider error: {err.message}",
            retry_after_s=retry_after,
        )

    def retry_after_header(self) -> str | None:
        # Retry-After takes whole seconds (RFC 9110). Round up: coming back early just
        # earns another 429.
        if self.retry_after_s is None:
            return None
        return str(math.ceil(self.retry_after_s))
