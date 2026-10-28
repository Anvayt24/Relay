"""Policy bundles: an immutable, versioned snapshot of every route's *policy* (PRD FR-11;
ADR 0023).

A bundle holds what decides how a request is served, per route:

- the **router**: its artifact (embedded, as the artifact's own JSON), its threshold, alias
  and embedding timeout;
- the **fallback graph** and how many hops a request may take;
- the **cache** parameters;
- the **evidence** spec.

It holds nothing else. Providers, models, prices and tenants stay in the YAML config: those
are infrastructure, and changing them is a deploy, not a policy change.

**Applying** a bundle never trusts it: ``effective_config`` copies the running config with the
bundle's routes and re-runs the whole config validation (unknown models, cyclic graphs, router
aliases, the judge, the band below the threshold), and every embedded artifact must match its
recorded hash. Anything wrong raises ``BundleError``, and the caller keeps what it has.
"""

import hashlib
import json
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from relay.config.schema import EvidenceSpec, FallbackEdge, RelayConfig, RouteCache, RouterSpec
from relay.routing.artifact import ArtifactError, RouterArtifact
from relay.routing.artifact import load as load_artifact
from relay.routing.artifact import loads as loads_artifact

FROZEN = ConfigDict(frozen=True, extra="forbid")
VERSION = re.compile(r"^\d{4}-\d{2}-\d{2}\.\d+$")


class BundleError(ValueError):
    """The bundle can't be applied: invalid, inconsistent with this deployment, or tampered."""


class BundleRouter(BaseModel):
    model_config = FROZEN

    artifact_json: str  # the artifact file's exact text: every replica gets identical bytes
    artifact_sha256: str  # RouterArtifact.sha256(): checked on every load
    threshold: float = Field(ge=0, le=1)
    alias: str = "relay/auto"
    embed_timeout_s: float = Field(default=0.5, gt=0)


class RoutePolicy(BaseModel):
    model_config = FROZEN

    router: BundleRouter | None = None
    fallback_graph: dict[str, list[FallbackEdge]] = Field(default_factory=dict)
    max_fallback_hops: int = Field(default=2, ge=0)
    cache: RouteCache = Field(default_factory=RouteCache)
    evidence: EvidenceSpec | None = None


class PolicyBundle(BaseModel):
    model_config = FROZEN

    version: str
    routes: dict[str, RoutePolicy]
    note: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def content(self) -> dict[str, Any]:
        """What the bundle *is*: the routes. Version, note and timestamp are labels."""
        return {name: r.model_dump(mode="json") for name, r in sorted(self.routes.items())}

    @property
    def content_sha256(self) -> str:
        blob = json.dumps(self.content(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, text: str) -> "PolicyBundle":
        return cls.model_validate_json(text)


def next_version(existing: list[str], today: date | None = None) -> str:
    """``YYYY-MM-DD.N``: the next free N for today (PRD §10.8's example "2026-10-20.3")."""
    day = (today or datetime.now(UTC).date()).isoformat()
    taken = [int(v.rsplit(".", 1)[1]) for v in existing if v.startswith(day + ".")]
    return f"{day}.{max(taken, default=0) + 1}"


def from_config(
    config: RelayConfig, config_dir: Path, *, version: str, note: str = ""
) -> PolicyBundle:
    """The policy in a YAML config, as a bundle. Router artifacts are read from disk (relative
    to the config file) and embedded; a missing or invalid artifact is an error here, unlike at
    gateway startup: a bundle must be complete before it can ever be activated."""
    if not VERSION.match(version):
        raise BundleError(f"version {version!r} is not YYYY-MM-DD.N")
    routes: dict[str, RoutePolicy] = {}
    for name, route in config.routes.items():
        router = None
        if route.router is not None:
            path = config_dir / route.router.artifact
            try:
                artifact = load_artifact(path)
            except ArtifactError as e:
                raise BundleError(f"route {name!r}: {e}") from e
            router = BundleRouter(
                artifact_json=path.read_text(encoding="utf-8"),
                artifact_sha256=artifact.sha256(),
                threshold=route.router.threshold,
                alias=route.router.alias,
                embed_timeout_s=route.router.embed_timeout_s,
            )
        routes[name] = RoutePolicy(
            router=router,
            fallback_graph=route.fallback_graph,
            max_fallback_hops=route.max_fallback_hops,
            cache=route.cache,
            evidence=route.evidence,
        )
    return PolicyBundle(version=version, routes=routes, note=note)


def artifacts(bundle: PolicyBundle) -> dict[str, RouterArtifact]:
    """Each routed route's artifact, parsed and checked against the bundle's recorded hash."""
    out = {}
    for name, route in bundle.routes.items():
        if route.router is None:
            continue
        try:
            artifact = loads_artifact(route.router.artifact_json)
        except ArtifactError as e:
            raise BundleError(f"bundle {bundle.version} route {name!r}: {e}") from e
        if artifact.sha256() != route.router.artifact_sha256:
            raise BundleError(f"bundle {bundle.version} route {name!r}: artifact hash mismatch")
        out[name] = artifact
    return out


def effective_config(base: RelayConfig, bundle: PolicyBundle) -> RelayConfig:
    """``base`` with each route's policy replaced by the bundle's, fully re-validated.

    A route the bundle doesn't mention keeps its YAML policy; a route the bundle names that
    this deployment doesn't have is an error (tenants and aliases live in YAML, so such a route
    could never be reached)."""
    unknown = set(bundle.routes) - set(base.routes)
    if unknown:
        raise BundleError(f"bundle {bundle.version} names unknown routes {sorted(unknown)}")
    raw = base.model_dump(mode="python")
    for name, policy in bundle.routes.items():
        route = raw["routes"][name]
        route["fallback_graph"] = policy.model_dump()["fallback_graph"]
        route["max_fallback_hops"] = policy.max_fallback_hops
        route["cache"] = policy.cache.model_dump()
        route["evidence"] = policy.evidence.model_dump() if policy.evidence else None
        route["router"] = (
            RouterSpec(
                artifact=f"bundle:{bundle.version}",  # embedded; not a file
                threshold=policy.router.threshold,
                alias=policy.router.alias,
                embed_timeout_s=policy.router.embed_timeout_s,
            ).model_dump()
            if policy.router
            else None
        )
    try:
        return RelayConfig.model_validate(raw)
    except ValueError as e:
        raise BundleError(f"bundle {bundle.version} is invalid here: {e}") from e
