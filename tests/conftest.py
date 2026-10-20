import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from relay.api.app import create_app
from relay.config.tenants import ANONYMOUS
from relay.pipeline.context import RequestContext
from relay.providers.mock import MockProvider
from relay.schemas.chat import ChatRequest, ChatResponse, Choice, ResponseMessage, Usage
from relay.settings import Settings
from relay.telemetry.log_writer import LogSink

BASE_URL = "http://relay.test"


def test_settings(**overrides: Any) -> Settings:
    """Settings for tests: never read the developer's .env. A RELAY_DATABASE_URL set there
    must not turn unit tests into database tests."""
    values: dict[str, Any] = {"env": "test", "database_url": None, **overrides}
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    """An HTTP client wired straight into the ASGI app: no server, no ports, no network.

    ASGITransport calls the app in-process, so tests are fast and deterministic, yet they
    still exercise the real HTTP layer (routing, headers, status codes, serialization).
    """
    app = create_app(test_settings())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as c:
        yield c


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()


@pytest.fixture
def make_app() -> Callable[..., FastAPI]:
    """Build an app around a given mock provider, so a test controls the provider's behavior
    and can inspect it afterwards (e.g. ``provider.calls``). Keyword arguments override
    settings; ``sink`` replaces the log sink."""

    def _make(provider: MockProvider, *, sink: LogSink | None = None, **settings: Any) -> FastAPI:
        return create_app(test_settings(**settings), providers={"mock": provider}, log_sink=sink)

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
