"""PRD Appendix B as data: what Relay does about each upstream error class.

The table is code-reviewed in one place instead of being scattered across ``if`` statements.
The dispatcher asks three questions about each failure:

1. Retry the same model once? Only for transient upstream faults (``overloaded``,
   ``server_error``). A 429 is never retried on the same limit: the pause state waits it
   out. A timeout isn't retried either: the same model is likely to be slow again, and the
   deadline is better spent on a fallback.
2. Fall back to another model? Yes for anything that says "this provider can't serve right
   now". No when the *request* is the problem (``bad_request``, ``content_filtered``): every
   model would refuse it, and for a moderation block, trying other models until one says yes
   would launder the refusal.
3. Count toward the circuit breaker? Only signs of an unhealthy upstream. 429s are excluded:
   the shared pause state handles them, and counting them too would mean two mechanisms
   reacting to one signal (FR-5).
"""

from relay.schemas.errors import ErrorClass

RETRY_ONCE: frozenset[ErrorClass] = frozenset({ErrorClass.OVERLOADED, ErrorClass.SERVER_ERROR})

FALLBACK_ON: frozenset[ErrorClass] = frozenset(
    {
        ErrorClass.RATE_LIMITED,
        ErrorClass.QUOTA_EXHAUSTED,
        ErrorClass.OVERLOADED,
        ErrorClass.SERVER_ERROR,
        ErrorClass.TIMEOUT,
        # The model was removed or renamed upstream (common on free tiers): mark it
        # unavailable, alert, and serve from the next candidate.
        ErrorClass.MODEL_NOT_FOUND,
        # Relay's own provider key is wrong. Not the client's fault, and another provider's
        # key is unaffected.
        ErrorClass.AUTH,
    }
)

# Falls back only along an edge to a model with a larger context window: any other model
# would reject the same prompt.
CONTEXT_FALLBACK: frozenset[ErrorClass] = frozenset({ErrorClass.CONTEXT_LENGTH_EXCEEDED})

BREAKER_FAILURES: frozenset[ErrorClass] = frozenset(
    {
        ErrorClass.OVERLOADED,
        ErrorClass.TIMEOUT,
        ErrorClass.SERVER_ERROR,
        ErrorClass.STREAM_BROKEN,
    }
)

# Neither a breaker success nor a failure: owned by the pause state (FR-4).
BREAKER_IGNORES: frozenset[ErrorClass] = frozenset(
    {ErrorClass.RATE_LIMITED, ErrorClass.QUOTA_EXHAUSTED}
)


def can_fall_back(error_class: ErrorClass) -> bool:
    return error_class in FALLBACK_ON or error_class in CONTEXT_FALLBACK
