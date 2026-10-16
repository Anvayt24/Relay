"""HTTP surface of the gateway (PRD FR-1).

Owns the FastAPI app factory, the operational endpoints (health, readiness, metrics),
and, from Phase 1, the OpenAI-compatible ``/v1`` routes, error envelopes, and SSE streaming.
It translates HTTP to and from the internal schema and hands every request to the pipeline;
it contains no routing, caching, or provider logic itself.
"""
