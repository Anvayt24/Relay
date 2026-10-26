"""The OpenAI-compatible endpoints (PRD FR-1).

This layer only translates: HTTP in → ``RequestContext`` → pipeline → HTTP out. Routing,
provider calls and (later) admission and caching live in the pipeline, so they can be tested
without HTTP and reused by non-HTTP callers such as the evidence engine's shadow runner.
"""

import time
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, Request, Response
from prometheus_client import Counter
from pydantic import BaseModel

from relay.api.streaming import RelayStreamingResponse
from relay.config.models import ModelSpec
from relay.config.pricing import usd
from relay.config.schema import RelayConfig
from relay.config.tenants import Priority, Tenant
from relay.pipeline.context import RequestContext
from relay.pipeline.runner import Pipeline, StreamResult
from relay.routing.static import STATIC_POLICY_VERSION
from relay.schemas.chat import ChatRequest, ChatResponse
from relay.settings import Settings

router = APIRouter(prefix="/v1")

IGNORED_FIELDS = Counter(
    "relay_request_ignored_fields_total",
    "Unknown request fields that were ignored.",
    # Deliberately no "field" label. Field names come from clients, so a label would let
    # any client create unbounded time series and take down Prometheus (cardinality).
)


# Dependencies. FastAPI calls these per request and passes the results to the endpoint.
# They read from app.state, which the factory filled in, so there are no module globals
# and a test app can carry different objects.
def get_pipeline(request: Request) -> Pipeline:
    pipeline: Pipeline = request.app.state.pipeline
    return pipeline


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


async def get_tenant(request: Request) -> Tenant:
    """Authenticate the caller (FR-3). Runs before the body is even looked at: an
    unauthenticated caller gets 401, never a validation error that hints at the API."""
    fixed: Tenant | None = request.app.state.auth_disabled_tenant
    if fixed is not None:
        return fixed
    tenant: Tenant = await request.app.state.authenticator.authenticate(
        request.headers.get("authorization")
    )
    return tenant


# response_model is explicit because the endpoint returns either a ChatResponse (serialized by
# FastAPI) or a ready-made streaming Response (passed through untouched). FastAPI cannot
# derive a response model from that union.
@router.post("/chat/completions", response_model=ChatResponse)
async def chat_completions(
    body: ChatRequest,
    request: Request,
    response: Response,
    tenant: Annotated[Tenant, Depends(get_tenant)],
    pipeline: Annotated[Pipeline, Depends(get_pipeline)],
    settings: Annotated[Settings, Depends(get_settings)],
    # A bad value is a validation error → 400 with param "x-relay-priority".
    x_relay_priority: Annotated[Priority | None, Header()] = None,
    x_relay_cohort: Annotated[str | None, Header()] = None,
    x_relay_cache: Annotated[str | None, Header()] = None,
) -> ChatResponse | Response:
    if body.ignored_fields:
        IGNORED_FIELDS.inc(len(body.ignored_fields))

    priority = x_relay_priority or tenant.default_priority
    ctx = RequestContext(
        request_id=request.state.request_id,
        tenant=tenant,
        request=body,
        route=tenant.route,
        priority=priority,
        # Only tags on the tenant's allowlist are kept; anything else becomes "other", so
        # clients can't create unbounded cohorts in the evidence data (FR-1).
        cohort=tenant.cohort(x_relay_cohort),
        policy_version=STATIC_POLICY_VERSION,
        # Fixed now, per tenant and priority: every retry and fallback hop must fit inside it.
        deadline=time.monotonic() + tenant.deadlines.total_for(priority),
        stream=body.stream,
        cache_no_store=(x_relay_cache or "").lower() == "no-store",
    )

    # Any failure up to here, including a stream that broke before its first content chunk,
    # is raised as RelayError and becomes an ordinary JSON error response.
    result = await pipeline.run(ctx)

    served = ctx.served_by
    if served is None:
        raise RuntimeError("pipeline returned a response nobody served")
    # The provider and model that *answered*: after a fallback not the router's choice, and
    # on a cache hit "relay-cache" with the model that originally produced the answer (FR-1).
    relay_headers = {
        "X-Relay-Route": ctx.route,
        "X-Relay-Provider": served[0],
        "X-Relay-Model": served[1],
        "X-Relay-Cache": ctx.cache.status if ctx.cache else "bypass",
    }
    if ctx.degraded:
        # Only ever "true"; absent otherwise (ADR 0010).
        relay_headers["X-Relay-Degraded"] = "true"

    if isinstance(result, StreamResult):
        # From here the response owns the stream *and* the pipeline's finalize.
        return RelayStreamingResponse(
            result, ctx, idle_timeout_s=settings.stream_idle_timeout_s, headers=relay_headers
        )
    # Headers on the injected Response are merged into the one FastAPI builds from the
    # return value. JSON responses know their cost now; streams carry it in the final chunk.
    if ctx.cost is not None:
        relay_headers["X-Relay-Cost-USD"] = usd(ctx.cost.billed_nano)
    response.headers.update(relay_headers)
    return result


class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    # Relay names have no creation date. 0 keeps the field valid and the output deterministic.
    created: int = 0
    owned_by: str


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelObject]


@router.get("/models")
async def list_models(
    request: Request, tenant: Annotated[Tenant, Depends(get_tenant)]
) -> ModelList:
    # Only what *this* tenant can call (FR-1): its route's aliases, then explicit models, each
    # only if its provider is configured and the tenant may use it.
    config: RelayConfig = request.app.state.config
    available: dict[str, ModelSpec] = request.app.state.models

    def usable(name: str) -> bool:
        return name in available and tenant.allows(name, available[name].provider)

    aliases = config.routes[tenant.route].aliases
    data = [ModelObject(id=alias, owned_by="relay") for alias, t in aliases.items() if usable(t)]
    data += [
        ModelObject(id=name, owned_by=spec.provider)
        for name, spec in available.items()
        if usable(name)
    ]
    return ModelList(data=data)
