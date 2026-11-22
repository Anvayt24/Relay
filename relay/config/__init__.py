"""Business configuration, loaded from YAML.

Models (Relay name -> provider model ID), the price table, routes with fallback graphs,
and tenants. Validated at load time: acyclic fallback graphs, every model priced, aliases
resolvable. Process settings (env vars) live in ``relay.settings`` instead.
"""
