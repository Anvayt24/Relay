"""The OpenAI-compatible endpoints.

This layer only translates: HTTP in → ``RequestContext`` → pipeline → HTTP out. Routing,
provider calls and (later) admission and caching live in the pipeline, so they can be tested
without HTTP and reused by non-HTTP callers such as the evidence engine's shadow runner.
"""

import time
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, Request, Response
from prometheus_client import Counter
from pydantic import BaseModel

from relay.api.client_ip import resolve_client_ip
from relay.api.decision import DECISION_HEADERS, decision_fields
from relay.api.streaming import RelayStreamingResponse
from relay.config.models import ModelSpec
from relay.config.pricing import usd
from relay.config.schema import RelayConfig
from relay.config.tenants import Priority, Tenant
from relay.pipeline.context import RequestContext
from relay.pipeline.runner import Pipeline, StreamResult
from relay.policy.runtime import PolicyRuntime
from relay.routing.static import STATIC_POLICY_VERSION
from relay.schemas.chat import ChatRequest, ChatResponse
from relay.schemas.errors import RelayError
from relay.settings import Settings
from relay.telemetry.metrics import record_rejected_credentials

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
    """Authenticate the caller. Runs before the body is even looked at: an
    unauthenticated caller gets 401, never a validation error that hints at the API."""
    fixed: Tenant | None = request.app.state.auth_disabled_tenant
    if fixed is not None:
        return fixed
    try:
        tenant: Tenant = await request.app.state.authenticator.authenticate(
            request.headers.get("authorization")
        )
    except RelayError as exc:
        record_rejected_credentials(exc.error_class.value)
        raise
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
    # Canary stickiness: requests with the same session key land in the same arm.
    x_relay_session: Annotated[str | None, Header()] = None,
) -> ChatResponse | Response:
    return await serve_chat(
        body,
        request,
        response,
        tenant,
        pipeline=pipeline,
        settings=settings,
        priority_header=x_relay_priority,
        cohort_header=x_relay_cohort,
        cache_header=x_relay_cache,
        session_header=x_relay_session,
    )


async def serve_chat(
    body: ChatRequest,
    request: Request,
    response: Response,
    tenant: Tenant,
    *,
    pipeline: Pipeline,
    settings: Settings,
    priority_header: Priority | None = None,
    cohort_header: str | None = None,
    cache_header: str | None = None,
    session_header: str | None = None,
    client_ip: str | None = None,
) -> ChatResponse | Response:
    """Run one chat request as `tenant`. The public endpoint and the demo's keyless one
    (relay/demo) both call this, so the demo is the real pipeline with the real limits, and
    only the way the tenant is found differs."""
    if body.ignored_fields:
        IGNORED_FIELDS.inc(len(body.ignored_fields))

    priority = priority_header or tenant.default_priority
    # One policy snapshot per request: every stage reads route policy from it, so
    # a swap that lands mid-request can't mix two policies. The canary arm is decided here,
    # once, from the session key if the client sent one, else the request id.
    runtime: PolicyRuntime = request.app.state.policy_runtime
    snapshot = runtime.snapshot()
    arm = snapshot.arm(session_header or request.state.request_id)
    applied = snapshot.applied(arm)
    ctx = RequestContext(
        request_id=request.state.request_id,
        tenant=tenant,
        request=body,
        route=tenant.route,
        priority=priority,
        # Only tags on the tenant's allowlist are kept; anything else becomes "other", so
        # clients can't create unbounded cohorts in the evidence data.
        cohort=tenant.cohort(cohort_header),
        policy_version=applied.version or STATIC_POLICY_VERSION,
        # Fixed now, per tenant and priority: every retry and fallback hop must fit inside it.
        deadline=time.monotonic() + tenant.deadlines.total_for(priority),
        stream=body.stream,
        cache_no_store=(cache_header or "").lower() == "no-store",
        policy=snapshot,
        applied=applied,
        canary_arm=arm,
        span=getattr(request.state, "span", None),
        client_ip=client_ip
        or resolve_client_ip(
            request.client.host if request.client else None,
            request.headers.get("x-forwarded-for"),
            request.app.state.trusted_proxies,
        ),
    )

    # Any failure up to here, including a stream that broke before its first content chunk,
    # is raised as RelayError and becomes an ordinary JSON error response.
    result = await pipeline.run(ctx)

    served = ctx.served_by
    if served is None:
        raise RuntimeError("pipeline returned a response nobody served")
    # The provider and model that *answered*: after a fallback not the router's choice, and
    # on a cache hit "relay-cache" with the model that originally produced the answer.
    relay_headers = {
        "X-Relay-Route": ctx.route,
        "X-Relay-Provider": served[0],
        "X-Relay-Model": served[1],
        "X-Relay-Cache": ctx.cache.status if ctx.cache else "bypass",
    }
    if ctx.degraded:
        # Only ever "true"; absent otherwise.
        relay_headers["X-Relay-Degraded"] = "true"
    if applied.version is not None:
        relay_headers["X-Relay-Policy"] = applied.version  # the bundle that served it
    # What was decided and why. For a stream these are what is known when the first
    # chunk goes out; the rest arrives in the final chunk.
    for name, value in decision_fields(ctx).items():
        relay_headers[DECISION_HEADERS[name]] = value

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
    # Only what *this* tenant can call: its route's aliases, then explicit models, each
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
