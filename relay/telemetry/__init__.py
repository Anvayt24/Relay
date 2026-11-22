"""Observability: an async request-log writer, structured logs, and OpenTelemetry tracing.

Rule: logging is never on the hot path. Request rows go to a bounded in-memory queue and a
background task batches them to Postgres. If Postgres is slow or down, the oldest rows are
dropped and counted; the request itself is never delayed.
"""
