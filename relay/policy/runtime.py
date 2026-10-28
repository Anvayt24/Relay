"""The policy a replica is serving, swapped atomically, read once per request (ADR 0023).

    PolicyRuntime ── snapshot() ──▶ PolicySnapshot(active, candidate, mode, canary_pct)
                                         │ one per request, immutable
                       canary arm? ──────┤
                                         ▼
                                ctx.applied: BuiltPolicy (effective config + its router)

A ``BuiltPolicy`` is a bundle made ready to serve: the YAML config with the bundle's routes
(validated) and the router built from the bundle's embedded artifacts. Built policies are
cached by version, so a rollback to a version this replica has served before is instant.

``snapshot()`` returns one object holding everything a request needs to know about policy.
Every stage reads from the request's snapshot, never from the runtime, so a swap that lands
while a request is in flight doesn't give that request two policies.

**No bundle** (no database, or none created yet): the active policy is the YAML one
(``version=None``) and everything behaves exactly as before policy bundles existed.
"""

import hashlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from relay.config.models import ModelSpec
from relay.config.schema import RelayConfig
from relay.policy.bundle import PolicyBundle, artifacts, effective_config
from relay.routing.base import Router

logger = logging.getLogger(__name__)

Mode = Literal["active", "shadow", "canary"]
Arm = Literal["control", "canary"]

# Builds a router for an effective config and its artifacts (the app passes one that wires
# the shared embedder; tests pass simple ones).
type RouterFactory = Callable[[RelayConfig, Mapping[str, object]], Router]


@dataclass(frozen=True)
class BuiltPolicy:
    version: str | None  # None = the YAML policy (no bundle)
    config: RelayConfig  # the effective config: YAML infrastructure + this policy's routes
    router: Router


@dataclass(frozen=True)
class PolicySnapshot:
    active: BuiltPolicy
    candidate: BuiltPolicy | None = None
    mode: Mode = "active"  # what the candidate is doing: shadow or canary (active: none)
    canary_pct: int = 0

    def arm(self, key: str) -> Arm | None:
        """The request's canary arm, or None when no canary is running."""
        if self.mode != "canary" or self.candidate is None:
            return None
        return canary_arm(self.candidate.version or "", key, self.canary_pct)

    def applied(self, arm: Arm | None) -> BuiltPolicy:
        if arm == "canary" and self.candidate is not None:
            return self.candidate
        return self.active

    def shadow(self) -> BuiltPolicy | None:
        return self.candidate if self.mode == "shadow" else None


def canary_arm(version: str, key: str, pct: int) -> Arm:
    """Deterministic: the same key always lands in the same arm for one canary (stickiness),
    and a new canary version reshuffles everyone (FR-11). blake2b, not Python's hash(), which
    is randomized per process and would put a session in different arms on different
    replicas."""
    digest = hashlib.blake2b(f"{version}|{key}".encode(), digest_size=8).digest()
    return "canary" if int.from_bytes(digest, "big") % 100 < pct else "control"


class PolicyRuntime:
    def __init__(
        self,
        base: RelayConfig,
        models: Mapping[str, ModelSpec],
        *,
        bootstrap: Router,
        router_factory: RouterFactory,
    ) -> None:
        self._base = base
        self._models = models
        self._router_factory = router_factory
        self._built: dict[str, BuiltPolicy] = {}
        self._yaml = BuiltPolicy(None, base, bootstrap)
        self._snapshot = PolicySnapshot(active=self._yaml)

    @property
    def base(self) -> RelayConfig:
        return self._base

    def snapshot(self) -> PolicySnapshot:
        return self._snapshot

    def build(self, bundle: PolicyBundle) -> BuiltPolicy:
        """Validate and build a bundle (cached by version). Raises ``BundleError``."""
        cached = self._built.get(bundle.version)
        if cached is not None:
            return cached
        config = effective_config(self._base, bundle)
        router = self._router_factory(config, artifacts(bundle))
        built = BuiltPolicy(bundle.version, config, router)
        self._built[bundle.version] = built
        return built

    def apply(
        self,
        active: PolicyBundle | None,
        candidate: PolicyBundle | None = None,
        *,
        mode: Mode = "active",
        canary_pct: int = 0,
    ) -> PolicySnapshot:
        """Build everything first, then swap in one assignment: a request sees either the old
        snapshot or the new one, never half of each. Raises ``BundleError`` and changes
        nothing if any bundle is invalid."""
        # No active bundle in the pointer: the YAML policy.
        new_active = self.build(active) if active is not None else self._yaml
        new_candidate = self.build(candidate) if candidate is not None else None
        if new_candidate is None:
            mode, canary_pct = "active", 0
        self._snapshot = PolicySnapshot(new_active, new_candidate, mode, canary_pct)
        logger.info(
            "policy: active %s, candidate %s (%s %s%%)",
            new_active.version or "yaml",
            new_candidate.version if new_candidate else None,
            mode,
            canary_pct,
        )
        return self._snapshot
