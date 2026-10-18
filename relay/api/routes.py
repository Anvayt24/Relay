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

from relay.config.models import ModelSpec
from relay.config.tenants import ANONYMOUS
from relay.pipeline.context import Priority, RequestContext
from relay.pipeline.runner import Pipeline
from relay.routing.static import STATIC_POLICY_VERSION
from relay.schemas.chat import ChatRequest, ChatResponse
from relay.schemas.errors import ErrorClass, RelayError
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


@router.post("/chat/completions")
async def chat_completions(
    body: ChatRequest,
    request: Request,
    response: Response,
    pipeline: Annotated[Pipeline, Depends(get_pipeline)],
    settings: Annotated[Settings, Depends(get_settings)],
    # A bad value is a validation error → 400 with param "x-relay-priority".
    x_relay_priority: Annotated[Priority, Header()] = "interactive",
    x_relay_cohort: Annotated[str | None, Header()] = None,
) -> ChatResponse:
    if body.stream:
        raise RelayError(
            400, ErrorClass.UNSUPPORTED_PARAMETER, "stream=true is not supported yet", "stream"
        )
    if body.ignored_fields:
        IGNORED_FIELDS.inc(len(body.ignored_fields))

    tenant = ANONYMOUS  # Phase 5: resolved from the API key
    ctx = RequestContext(
        request_id=request.state.request_id,
        tenant=tenant,
        request=body,
        route=tenant.route,
        priority=x_relay_priority,
        cohort=x_relay_cohort,
        policy_version=STATIC_POLICY_VERSION,
        deadline=time.monotonic() + settings.request_timeout_s,
        stream=body.stream,
    )

    result = await pipeline.run(ctx)

    if ctx.decision is None or not ctx.attempts:
        raise RuntimeError("pipeline returned a response without a decision and an attempt")
    # Headers on the injected Response are merged into the one FastAPI builds from the
    # return value.
    response.headers["X-Relay-Route"] = ctx.route
    response.headers["X-Relay-Provider"] = ctx.attempts[-1].provider
    response.headers["X-Relay-Model"] = ctx.decision.model
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
async def list_models(request: Request) -> ModelList:
    # Phase 5 filters this per tenant (allowed_providers, FR-1).
    models: dict[str, ModelSpec] = request.app.state.models
    aliases: dict[str, str] = request.app.state.aliases
    data = [ModelObject(id=alias, owned_by="relay") for alias in aliases]
    data += [ModelObject(id=name, owned_by=spec.provider) for name, spec in models.items()]
    return ModelList(data=data)
