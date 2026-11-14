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

**A bundle can't ask for more than the replica built.** What a replica constructs at startup
(the embedder, the evidence engine) is decided from its YAML config. A bundle that adds a
trained router to a replica with no embedder, or an ``evidence:`` section to one with no
evidence engine, would pass validation and then silently degrade (every ``relay/auto`` to
strong; no shadow pairs, no cards). ``build`` refuses it instead, so the replica keeps its
last-known-good policy and the poll error says why (found in review).
"""

import hashlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from relay.config.schema import RelayConfig
from relay.policy.bundle import BundleError, PolicyBundle, artifacts, effective_config
from relay.policy.control import CandidateMode
from relay.routing.artifact import RouterArtifact
from relay.routing.base import Router

logger = logging.getLogger(__name__)

Arm = Literal["control", "canary"]

# Builds a router for an effective config and each routed route's artifact (None: missing or
# invalid, so that route's relay/auto goes to strong). The app's wires the shared embedder and
# also builds the YAML policy's router; tests pass simple ones.
type RouterFactory = Callable[[RelayConfig, Mapping[str, RouterArtifact | None]], Router]


@dataclass(frozen=True)
class BuiltPolicy:
    version: str | None  # None = the YAML policy (no bundle)
    config: RelayConfig  # the effective config: YAML infrastructure + this policy's routes
    router: Router


@dataclass(frozen=True)
class PolicySnapshot:
    active: BuiltPolicy
    candidate: BuiltPolicy | None = None
    mode: CandidateMode | None = None  # what the candidate is doing (None: no candidate)
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

    def describe(self) -> dict[str, Any]:
        """What this replica serves, as /readyz and the admin API both report it."""
        return {
            "active": self.active.version or "yaml",
            "candidate": self.candidate.version if self.candidate else None,
            "mode": self.mode,
            "canary_pct": self.canary_pct,
        }


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
        *,
        bootstrap: Router,
        router_factory: RouterFactory,
        can_route: bool,
        has_evidence: bool,
    ) -> None:
        """``can_route``: this replica can run a trained router (it built an embedder, or the
        embedder is deliberately off, which is the documented route-to-strong degradation).
        ``has_evidence``: it built the evidence engine (card board, shadow sampling)."""
        self._base = base
        self._router_factory = router_factory
        self._can_route = can_route
        self._has_evidence = has_evidence
        self._built: dict[str, BuiltPolicy] = {}
        self._yaml = BuiltPolicy(None, base, bootstrap)
        self._snapshot = PolicySnapshot(active=self._yaml)

    def snapshot(self) -> PolicySnapshot:
        return self._snapshot

    def build(self, bundle: PolicyBundle) -> BuiltPolicy:
        """Validate and build a bundle (cached by version). Raises ``BundleError``."""
        cached = self._built.get(bundle.version)
        if cached is not None:
            return cached
        config = effective_config(self._base, bundle)
        for name, route in config.routes.items():
            if route.router is not None and not self._can_route:
                raise BundleError(
                    f"bundle {bundle.version} route {name!r} has a trained router, but this "
                    "replica built no embedder (no route in its YAML config uses a router)"
                )
            if route.evidence is not None and not self._has_evidence:
                raise BundleError(
                    f"bundle {bundle.version} route {name!r} has an evidence section, but this "
                    "replica built no evidence engine (no route in its YAML config has one)"
                )
        router = self._router_factory(config, artifacts(bundle))
        built = BuiltPolicy(bundle.version, config, router)
        self._built[bundle.version] = built
        return built

    def apply(
        self,
        active: PolicyBundle | None,
        candidate: PolicyBundle | None = None,
        *,
        mode: CandidateMode | None = None,
        canary_pct: int = 0,
    ) -> PolicySnapshot:
        """Build everything first, then swap in one assignment: a request sees either the old
        snapshot or the new one, never half of each. Raises ``BundleError`` and changes
        nothing if any bundle is invalid."""
        # No active bundle in the pointer: the YAML policy.
        new_active = self.build(active) if active is not None else self._yaml
        new_candidate = self.build(candidate) if candidate is not None else None
        if new_candidate is None:
            mode, canary_pct = None, 0
        self._snapshot = PolicySnapshot(new_active, new_candidate, mode, canary_pct)
        logger.info(
            "policy: active %s, candidate %s (%s %s%%)",
            new_active.version or "yaml",
            new_candidate.version if new_candidate else None,
            mode,
            canary_pct,
        )
        return self._snapshot
