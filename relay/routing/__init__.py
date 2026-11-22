"""Cheap-vs-strong routing: static rules, or a trained router.

``Router.decide(request, ctx) -> RouteDecision``. A missing or corrupt router artifact always
routes to the strong tier (quality-safe default).
"""
