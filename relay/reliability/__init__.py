"""Reliability (PRD FR-5): the single retry owner.

Everything that decides "call again, call something else, or give up" lives here, and only
here. Adapters never retry (ADR 0005); ``ProviderStage`` just hands each request to
``ReliableDispatcher``. Retries in two places would multiply: that is how a 429 becomes a
retry storm.

Reading order:
- ``policy``: Appendix B as data. Which error classes retry, fall back, or count toward the
  breaker.
- ``backoff``, ``retry_budget``: how long to wait before the one retry, and a cap on how
  many retries a provider may receive.
- ``breaker``: the per-(provider, model) circuit breaker (ADR 0009).
- ``edges``, ``candidates``: which fallback edges are usable, and in what order (ADRs 0008,
  0010, 0011).
- ``dispatch``: the loop that ties it all together, bounded by the request deadline.

All of it is pure apart from ``dispatch``: clocks and random generators are injected, so the
state machines are tested exactly.
"""
