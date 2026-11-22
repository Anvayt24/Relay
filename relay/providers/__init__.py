"""Provider adapters.

Each adapter translates the internal OpenAI-shaped schema to and from one provider's wire
format, maps errors to Relay's error taxonomy (``ErrorClass``), and surfaces rate-limit signals.
Adapters never retry: Relay owns retries in exactly one place (``relay.reliability``).
"""
