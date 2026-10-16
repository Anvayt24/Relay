"""Admission control and rate limiting (PRD FR-4). Phase 6.

RPM/concurrency limits and priority queues before the cache. After routing, token and USD
reservations sized for the worst-case fallback, reconciled with actual usage. Shared pause
state in Redis, scoped to each provider's real limit scope.
"""
