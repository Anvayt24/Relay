"""Operational endpoints: liveness, readiness, and Prometheus metrics.

Liveness and readiness answer different questions, and orchestrators act on them differently:

- ``/healthz`` (liveness): "is this process alive?" If it fails, the orchestrator *restarts*
  the container. It must therefore never check dependencies: if Redis is down, restarting
  every gateway replica doesn't fix Redis, and the restarts just cause a second outage.
- ``/readyz`` (readiness): "should this replica receive traffic right now?" If it fails, the
  load balancer *stops routing to* the replica without killing it. Per PRD FR-1, it reports
  degraded dependencies in the body but returns 503 only when the gateway cannot serve at
  all (config not loaded, shutting down). Dependencies are added here as they are
  introduced (Postgres in Phase 2, Redis in Phase 6).
"""

from typing import Any, Literal

from fastapi import APIRouter, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from relay.admission.resilient import ResilientStore
from relay.settings import Settings
from relay.telemetry.log_writer import LogWriter

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request, response: Response) -> dict[str, Any]:
    dependencies = {"postgres": _postgres_status(request), "redis": _redis_status(request)}
    policy = _policy_status(request)
    if getattr(request.app.state, "draining", False):
        # Shutting down: take this replica out of the load balancer (PRD §8).
        response.status_code = 503
        return {"status": "draining", "dependencies": dependencies, "policy": policy}
    poller = getattr(request.app.state, "policy_poller", None)
    if poller is not None and not poller.first_attempt_done:
        # PRD FR-1: not ready until the policy is loaded. Only the first poll is waited for:
        # if the database is down it fails, and the replica serves the last-known (YAML) policy.
        response.status_code = 503
        return {"status": "loading_policy", "dependencies": dependencies, "policy": policy}
    # Otherwise always 200 while serving: a degraded dependency is reported, not failed on.
    return {"status": "ready", "dependencies": dependencies, "policy": policy}


def _policy_status(request: Request) -> dict[str, Any]:
    runtime = getattr(request.app.state, "policy_runtime", None)
    if runtime is None:
        return {"active": "yaml"}
    snap = runtime.snapshot()
    return {
        "active": snap.active.version or "yaml",
        "candidate": snap.candidate.version if snap.candidate else None,
        "mode": snap.mode,
        "canary_pct": snap.canary_pct,
    }


def _redis_status(request: Request) -> Literal["ok", "degraded", "disabled"]:
    store = request.app.state.capacity_store
    if not isinstance(store, ResilientStore):
        return "disabled"  # in-process admission state: no Redis configured
    # Passive, like Postgres: the last admission operation's outcome.
    return "ok" if store.healthy else "degraded"


def _postgres_status(request: Request) -> Literal["ok", "degraded", "disabled"]:
    settings: Settings = request.app.state.settings
    if not settings.database_url:
        return "disabled"
    writer: LogWriter = request.app.state.log_writer
    # Passive check: the result of the last log write. It costs nothing per probe, but an
    # idle replica reports the last known state. Phase 14 can add an active ping if needed.
    return "degraded" if writer.last_write_ok is False else "ok"


# Never expose /metrics publicly (PRD §8). For now compose publishes it for local dev;
# the deployed demo keeps it on the internal network only (Phase 15).
@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
