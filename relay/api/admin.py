"""The policy control plane over HTTP (PRD FR-11; ADR 0024).

    GET  /v1/admin/policy                 the pointer, what this replica serves, recent events
    POST /v1/admin/policy/shadow          {"version": "2026-10-28.2"}
    POST /v1/admin/policy/canary          {"version": "2026-10-28.2", "pct": 10}
    POST /v1/admin/policy/promote         {}
    POST /v1/admin/policy/rollback        {"reason": "..."}
    POST /v1/admin/policy/activate        {"version": "2026-10-28.1"}

The same operations as ``python -m relay.policy``, applied through the same store. Bundles are
created with the CLI only (it reads artifact files).

**Off unless configured.** Without ``RELAY_ADMIN_TOKEN`` every route here answers 404, as if
it didn't exist: an admin API should never be on by accident. With it, a request needs
``Authorization: Bearer <token>``; the comparison is constant-time (``hmac.compare_digest``),
so response timing can't be used to guess the token. A wrong token is 401.

After a change the replica that served the call refreshes at once; the others follow at
their next poll.
"""

import hmac
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from relay.policy import control
from relay.policy.control import ControlError
from relay.policy.poller import PolicyPoller
from relay.policy.runtime import PolicyRuntime
from relay.policy.store import PolicyStore, bundle_infos, describe
from relay.settings import Settings

router = APIRouter(prefix="/v1/admin/policy")


def require_admin(request: Request) -> str:
    settings: Settings = request.app.state.settings
    if settings.admin_token is None:
        raise HTTPException(status_code=404, detail="Not Found")
    sent = request.headers.get("authorization", "")
    expected = f"Bearer {settings.admin_token.get_secret_value()}"
    if not hmac.compare_digest(sent.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="invalid admin token")
    return "admin-api"


def _store(request: Request) -> PolicyStore:
    store: PolicyStore | None = request.app.state.policy_store
    if store is None:
        raise HTTPException(status_code=503, detail="no policy store (RELAY_DATABASE_URL unset)")
    return store


class Change(BaseModel):
    version: str | None = None
    pct: int | None = Field(default=None, ge=1, le=100)
    reason: str = ""


@router.get("")
async def status(request: Request, _: Annotated[str, Depends(require_admin)]) -> dict[str, Any]:
    store = _store(request)
    runtime: PolicyRuntime = request.app.state.policy_runtime
    snap = runtime.snapshot()
    return {
        "pointer": describe(await store.pointer()),
        "serving": {
            "active": snap.active.version or "yaml",
            "candidate": snap.candidate.version if snap.candidate else None,
            "mode": snap.mode,
            "canary_pct": snap.canary_pct,
        },
        "bundles": bundle_infos(await store.bundles()),
        "events": [
            {"ts": e.ts.isoformat(), "action": e.action, "actor": e.actor, "reason": e.reason,
             "after": describe(e.after)}
            for e in await store.events(10)
        ],
    }  # fmt: skip


Action = Literal["shadow", "canary", "promote", "rollback", "activate"]


@router.post("/{action}")
async def change(
    action: Action,
    body: Change,
    request: Request,
    actor: Annotated[str, Depends(require_admin)],
) -> dict[str, Any]:
    store = _store(request)
    try:
        after = await store.change(
            action, transition(action, body), actor=actor, reason=body.reason
        )
    except ControlError as e:
        raise HTTPException(status_code=409, detail=str(e)) from None
    poller: PolicyPoller | None = request.app.state.policy_poller
    if poller is not None:
        await poller.refresh()  # this replica at once; the others at their next poll
    return {"pointer": describe(after)}


def transition(action: str, body: Change) -> control.Transition:
    """The pointer transition for an action, with its arguments checked up front."""

    def need_version() -> str:
        if not body.version:
            raise HTTPException(status_code=422, detail=f"{action} needs a version")
        return body.version

    match action:
        case "shadow":
            v = need_version()
            return lambda p: control.shadow(p, v)
        case "canary":
            v = need_version()
            if body.pct is None:
                raise HTTPException(status_code=422, detail="canary needs pct (1-100)")
            pct = body.pct
            return lambda p: control.canary(p, v, pct)
        case "activate":
            v = need_version()
            return lambda p: control.activate(p, v)
        case "promote":
            return control.promote
        case _:
            return control.rollback
