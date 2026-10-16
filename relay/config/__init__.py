"""Business configuration (PRD FR-3, §9.5). In-code model registry in Phase 1; YAML in Phase 5.

Models (Relay name -> provider model ID), the price table, routes with fallback graphs,
and tenants. Validated at load time: acyclic fallback graphs, every model priced, aliases
resolvable. Process settings (env vars) live in ``relay.settings`` instead.
"""
