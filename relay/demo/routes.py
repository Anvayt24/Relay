"""The demo's HTTP surface, mounted under /v1/demo.

- ``GET  /v1/demo/info``          what is switched on, so the page shows only what works
- ``POST /v1/demo/chat?mode=``    keyless chat as one configured tenant; the real pipeline and
                                  the tenant's real limits (relay/api/routes.py:serve_chat)
- ``POST /v1/demo/chaos``         break the simulated primary (simulator only)
- ``GET  /v1/demo/chaos/stats``   Relay vs a naive client over the last 30 seconds
- ``GET  /v1/demo/evidence``      route cards, the judge's calibration, simulated results

Everything but ``info`` and ``evidence`` answers 404 unless it was configured, and the chat
endpoint can only ever act as the tenants named in settings: the caller picks a *mode*, never
a tenant.
"""

import hmac
import json
from functools import cache
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from pydantic import BaseModel

from relay.api.client_ip import resolve_client_ip
from relay.api.routes import get_pipeline, get_settings, serve_chat
from relay.config.tenants import Tenant
from relay.pipeline.runner import Pipeline
from relay.schemas.chat import ChatRequest, ChatResponse
from relay.schemas.errors import ErrorClass, RelayError
from relay.settings import Settings

router = APIRouter(prefix="/v1/demo")

DATA = Path(__file__).parent / "data"
CHAOS_ACTIONS_PER_MINUTE = 5
SECRET_HEADER = "X-Relay-Demo-Secret"
INTERNAL_CLIENT = "demo-internal"  # the panel's own traffic: not a visitor, never limited per IP


class ChaosAction(BaseModel):
    action: str
    duration_s: float = 30.0


def _tenant_for(request: Request, mode: str, secret: str | None) -> Tenant:
    settings: Settings = request.app.state.settings
    name: str | None
    if mode == "chaos":
        # Only the panel's own traffic generator knows this per-process secret.
        expected: str = request.app.state.demo_secret
        if request.app.state.chaos is None or not hmac.compare_digest(secret or "", expected):
            raise HTTPException(status_code=404, detail="Not Found")
        name = settings.demo_chaos_tenant
    else:
        name = settings.demo_tenant if mode == "real" else settings.demo_sim_tenant
    tenant: Tenant | None = request.app.state.config.tenants.get(name) if name else None
    if tenant is None:
        raise HTTPException(status_code=404, detail="Not Found")
    return tenant


@router.get("/info")
async def info(request: Request) -> dict[str, Any]:
    settings: Settings = request.app.state.settings
    config = request.app.state.config
    tenants = config.tenants

    def on(name: str | None) -> bool:
        return name is not None and name in tenants

    threshold = None
    if settings.demo_sim_tenant is not None and on(settings.demo_sim_tenant):
        route = config.routes[tenants[settings.demo_sim_tenant].route]
        threshold = route.router.threshold if route.router is not None else None
    return {
        "chat": {"real": on(settings.demo_tenant), "sim": on(settings.demo_sim_tenant)},
        "chaos": request.app.state.chaos is not None,
        "router_threshold": threshold,
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    request: Request,
    response: Response,
    mode: Literal["real", "sim", "chaos"],
    pipeline: Annotated[Pipeline, Depends(get_pipeline)],
    settings: Annotated[Settings, Depends(get_settings)],
    x_relay_demo_secret: Annotated[str | None, Header()] = None,
) -> ChatResponse | Response:
    tenant = _tenant_for(request, mode, x_relay_demo_secret)
    return await serve_chat(
        body,
        request,
        response,
        tenant,
        pipeline=pipeline,
        settings=settings,
        client_ip=INTERNAL_CLIENT if mode == "chaos" else None,
    )


# ------------------------------------------------------------------ chaos
@router.post("/chaos")
async def chaos(action: ChaosAction, request: Request) -> dict[str, Any]:
    panel = request.app.state.chaos
    if panel is None:
        raise HTTPException(status_code=404, detail="Not Found")
    # Visitors can break the simulator, so each address gets a handful of actions a minute.
    ip = resolve_client_ip(
        request.client.host if request.client else None,
        request.headers.get("x-forwarded-for"),
        request.app.state.trusted_proxies,
    )
    wait = await request.app.state.capacity_store.take(
        f"demochaos:{ip}",
        capacity=CHAOS_ACTIONS_PER_MINUTE,
        refill_per_s=CHAOS_ACTIONS_PER_MINUTE / 60,
        amount=1,
    )
    if wait > 0:
        raise RelayError(
            429,
            ErrorClass.CLIENT_RATE_LIMITED,
            f"at most {CHAOS_ACTIONS_PER_MINUTE} chaos actions per minute from one address",
            retry_after_s=wait,
        )
    try:
        panel.apply(action.action, action.duration_s)
    except ValueError as exc:
        raise RelayError(400, ErrorClass.INVALID_REQUEST, str(exc), param="action") from exc
    return {"ok": True, "action": action.action}


@router.get("/chaos/stats")
async def chaos_stats(request: Request) -> dict[str, Any]:
    panel = request.app.state.chaos
    if panel is None:
        raise HTTPException(status_code=404, detail="Not Found")
    stats: dict[str, Any] = panel.stats()
    return {**stats, "enabled": True}


# ------------------------------------------------------------------ evidence
@cache
def _packaged(name: str) -> Any:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


@router.get("/evidence")
async def evidence(request: Request) -> dict[str, Any]:
    board = request.app.state.card_board
    live = [card.as_dict() for card in board.cards()] if board is not None else []
    return {
        # Route cards from the real Groq check (29 shadow pairs): what a card looks like on real
        # traffic. `live` is what this gateway's own evidence engine holds right now, if any.
        "cards": _packaged("cards.json"),
        "live_cards": live,
        "policy": request.app.state.policy_runtime.snapshot().describe(),
        # The judge's agreement with human labels on public data, and the M7 gate it is held to.
        "calibration": _packaged("judge_calibration.json"),
        # Results from the simulator, labelled as such.
        "simulated": _packaged("evidence_sim.json"),
    }
