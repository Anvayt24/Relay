"""Helpers for policy tests: bundles whose behavior is obvious.

Both bundles use the length-only router artifact from the router tests, so a short prompt
("hi") scores about 0.95. Only the threshold differs:
- ``V1`` (threshold 0.5): "hi" goes **cheap**;
- ``V2`` (threshold 0.99): "hi" goes **strong**.
Which bundle served a request is therefore visible in ``X-Relay-Model``.
"""

from typing import Any

from relay.config.schema import FallbackEdge
from relay.policy.bundle import BundleRouter, PolicyBundle, RoutePolicy

from ..routing.conftest import length_artifact  # type: ignore[import-not-found]

V1, V2 = "2026-10-28.1", "2026-10-28.2"


def bundle(
    version: str,
    threshold: float,
    *,
    graph: dict[str, list[FallbackEdge]] | None = None,
    hops: int = 2,
    **route: Any,
) -> PolicyBundle:
    art = length_artifact()
    return PolicyBundle(
        version=version,
        routes={
            "default": RoutePolicy(
                router=BundleRouter(
                    artifact_json=art.dumps(), artifact_sha256=art.sha256(), threshold=threshold
                ),
                fallback_graph=graph or {},
                max_fallback_hops=hops,
                **route,
            )
        },
    )


def v1() -> PolicyBundle:
    return bundle(V1, 0.5)


def v2() -> PolicyBundle:
    return bundle(V2, 0.99)
