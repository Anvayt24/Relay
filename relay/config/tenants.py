"""Tenants. Phase 1 has no auth, so every request belongs to one anonymous tenant.

Phase 5 replaces this with API-key lookup. ``RequestContext`` already carries a ``Tenant``, so
the stages written now will not change shape when real tenants arrive.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Tenant:
    id: str
    route: str  # which route config this tenant's aliases resolve through (FR-3)


ANONYMOUS = Tenant(id="anonymous", route="default")
