import time
from collections.abc import AsyncIterator, Callable

import httpx
import pytest
from fastapi import FastAPI

from relay.api.app import create_app
from relay.config.tenants import ANONYMOUS
from relay.pipeline.context import RequestContext
from relay.providers.mock import MockProvider
from relay.schemas.chat import ChatRequest, ChatResponse, Choice, ResponseMessage, Usage
from relay.settings import Settings

BASE_URL = "http://relay.test"


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    """An HTTP client wired straight into the ASGI app: no server, no ports, no network.

    ASGITransport calls the app in-process, so tests are fast and deterministic, yet they
    still exercise the real HTTP layer (routing, headers, status codes, serialization).
    """
    app = create_app(Settings(env="test"))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as c:
        yield c


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()


@pytest.fixture
def make_app() -> Callable[[MockProvider], FastAPI]:
    """Build an app around a given mock provider, so a test controls the provider's behavior
    and can inspect it afterwards (e.g. ``provider.calls``)."""

    def _make(provider: MockProvider) -> FastAPI:
        return create_app(Settings(env="test"), providers={"mock": provider})

    return _make


@pytest.fixture
def ctx() -> RequestContext:
    """A RequestContext for unit tests of stages and routers (no HTTP involved)."""
    request = ChatRequest.model_validate(
        {"model": "relay/auto", "messages": [{"role": "user", "content": "hi"}]}
    )
    return RequestContext(
        request_id="test-req",
        tenant=ANONYMOUS,
        request=request,
        route=ANONYMOUS.route,
        priority="interactive",
        cohort=None,
        policy_version="test",
        deadline=time.monotonic() + 60,
        stream=False,
    )


@pytest.fixture
def chat_response() -> ChatResponse:
    return ChatResponse(
        id="chatcmpl-test",
        created=0,
        model="mock/strong",
        choices=[Choice(message=ResponseMessage(content="ok"), finish_reason="stop")],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
