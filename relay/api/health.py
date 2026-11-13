"""Operational endpoints: liveness, readiness, and Prometheus metrics.

Liveness and readiness answer different questions, and orchestrators act on them differently:

- ``/healthz`` (liveness): "is this process alive?" If it fails, the orchestrator *restarts*
  the container. It must therefore never check dependencies: if Redis is down, restarting
  every gateway replica doesn't fix Redis, and the restarts just cause a second outage.
- ``/readyz`` (readiness): "should this replica receive traffic right now?" If it fails, the
  load balancer *stops routing to* the replica without killing it. It reports
  degraded dependencies in the body but returns 503 only when the gateway cannot serve at
  all (config not loaded, shutting down). Postgres and Redis are reported here.
"""

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, Response
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
    state = request.app.state
    policy = state.policy_runtime.snapshot().describe()
    if state.draining:
        # Shutting down: take this replica out of the load balancer.
        response.status_code = 503
        return {"status": "draining", "dependencies": dependencies, "policy": policy}
    board = state.card_board
    if board is not None and not board.loaded and not board.first_attempt_done:
        # Same rule as the policy: a replica that hasn't read the route cards yet would use
        # every gated edge's seed state, even one a card says is unsafe.
        response.status_code = 503
        return {"status": "loading_cards", "dependencies": dependencies, "policy": policy}
    poller = state.policy_poller
    if poller is not None and not poller.first_attempt_done:
        # Not ready until the policy is loaded. Only the first poll is waited for:
        # if the database is down it fails, and the replica serves the last-known (YAML) policy.
        response.status_code = 503
        return {"status": "loading_policy", "dependencies": dependencies, "policy": policy}
    # Otherwise always 200 while serving: a degraded dependency is reported, not failed on.
    return {"status": "ready", "dependencies": dependencies, "policy": policy}


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
    # idle replica reports the last known state. An active ping could be added if needed.
    return "degraded" if writer.last_write_ok is False else "ok"


# Never expose /metrics publicly. With RELAY_METRICS_PORT set it is served on that
# port only (started in the app's lifespan), and this route answers 404.
@router.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> Response:
    settings: Settings = request.app.state.settings
    if settings.metrics_port is not None:
        raise HTTPException(status_code=404, detail="Not Found")
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
