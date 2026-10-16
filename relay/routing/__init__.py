"""Cheap-vs-strong routing (PRD FR-7). Static rules in Phase 1; trained router in Phase 11.

``Router.decide(request, ctx) -> RouteDecision``. A missing or corrupt router artifact always
routes to the strong tier (quality-safe default).
"""
