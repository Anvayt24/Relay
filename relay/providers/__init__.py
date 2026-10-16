"""Provider adapters (PRD FR-2). Mock in Phase 1; Groq, OpenRouter, Anthropic in Phase 4.

Each adapter translates the internal OpenAI-shaped schema to and from one provider's wire
format, maps errors to the taxonomy in PRD Appendix B, and surfaces rate-limit signals.
Adapters never retry: Relay owns retries in exactly one place (``relay.reliability``).
"""
