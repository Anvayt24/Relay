import asyncio
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from relay.admission.pause import PauseState
from relay.admission.store import CapacityStore, MemoryStore
from relay.api.app import create_app
from relay.config.loader import load_config
from relay.config.schema import RelayConfig
from relay.pipeline.context import RequestContext
from relay.providers.base import ProviderAdapter
from relay.providers.mock import MockProvider
from relay.providers.registry import NO_KEYS
from relay.reliability.breaker import BreakerRegistry
from relay.reliability.dispatch import ReliableDispatcher
from relay.reliability.edges import SeededEdgeStates
from relay.reliability.retry_budget import RetryBudget
from relay.schemas.chat import ChatRequest, ChatResponse, Choice, ResponseMessage, Usage
from relay.settings import Settings
from relay.telemetry.log_writer import LogSink

BASE_URL = "http://relay.test"
TEST_CONFIG_PATH = str(Path(__file__).with_name("config") / "test.yaml")
TEST_CONFIG: RelayConfig = load_config(TEST_CONFIG_PATH)
TEST_PEPPER = "test-pepper"


def test_settings(**overrides: Any) -> Settings:
    """Settings for tests: never read the developer's .env. A RELAY_DATABASE_URL set there
    must not turn unit tests into database tests.

    Auth is *off* by default here, mapping every request to the `test` tenant, so tests about
    streaming or logging don't each carry key setup. Auth itself is tested explicitly, with
    `auth_disabled=False` (tests/auth/)."""
    values: dict[str, Any] = {
        "env": "test",
        "database_url": None,
        "config_path": TEST_CONFIG_PATH,
        "auth_disabled": True,
        "auth_disabled_tenant": "test",
        "key_pepper": TEST_PEPPER,
        **overrides,
    }
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    """An HTTP client wired straight into the ASGI app: no server, no ports, no network.

    ASGITransport calls the app in-process, so tests are fast and deterministic, yet they
    still exercise the real HTTP layer (routing, headers, status codes, serialization).
    """
    app = create_app(test_settings(), provider_keys=NO_KEYS)
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

    def _make(
        provider: MockProvider | None = None,
        *,
        providers: Mapping[str, ProviderAdapter] | None = None,
        sink: LogSink | None = None,
        config: RelayConfig | None = None,
        store: CapacityStore | None = None,
        **settings: Any,
    ) -> FastAPI:
        # One provider for the common case; `providers` for fallback tests with several.
        if providers is None:
            providers = {"mock": provider or MockProvider()}
        return create_app(
            test_settings(**settings),
            providers=providers,
            log_sink=sink,
            config=config,
            capacity_store=store,
        )

    return _make


@pytest.fixture
def ctx() -> RequestContext:
    """A RequestContext for unit tests of stages and routers (no HTTP involved)."""
    request = ChatRequest.model_validate(
        {"model": "relay/auto", "messages": [{"role": "user", "content": "hi"}]}
    )
    return RequestContext(
        request_id="test-req",
        tenant=TEST_CONFIG.tenants["test"],
        request=request,
        route="default",
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


async def _no_sleep(_: float) -> None:
    return None


def build_dispatcher(
    config: RelayConfig,
    providers: Mapping[str, ProviderAdapter],
    *,
    store: CapacityStore | None = None,
    clock: Callable[[], float] = time.monotonic,
    first_token_ceiling_s: float = 1.0,
    pause: bool = True,
    seed: int = 0,
) -> ReliableDispatcher:
    """A dispatcher wired like the app's, with test-friendly defaults: backoff sleeps return
    at once (the jitter is still drawn, from a seeded RNG), and every clock is injectable."""
    store = store or MemoryStore(clock=clock)
    rel = config.reliability
    return ReliableDispatcher(
        config,
        {name: spec for name, spec in config.models.items() if spec.provider in providers},
        providers,
        store=store,
        pause=PauseState(store, config, rng=random.Random(seed)) if pause else None,
        breakers=BreakerRegistry(rel, clock=clock),
        retry_budget=RetryBudget(
            ratio=rel.retry_budget_ratio,
            minimum=rel.retry_budget_min,
            window_s=rel.retry_budget_window_s,
            clock=clock,
        ),
        edge_states=SeededEdgeStates(config),
        first_token_ceiling_s=first_token_ceiling_s,
        rng=random.Random(seed),
        clock=clock,
        sleep=_no_sleep,
    )


class _Server(uvicorn.Server):
    """uvicorn with an Event that fires once it is listening (no polling for readiness)."""

    def __init__(self, config: uvicorn.Config) -> None:
        super().__init__(config)
        self.ready = asyncio.Event()

    async def startup(self, sockets: Any = None) -> None:
        await super().startup(sockets)
        self.ready.set()


@pytest.fixture
async def live_server() -> AsyncIterator[Callable[[Any], Awaitable[str]]]:
    """Serve an ASGI app on a real localhost socket; returns its base URL.

    Only for what an in-process transport cannot show: a connection that drops mid-body exists
    only on a real connection. Still hermetic: 127.0.0.1, port 0 (the OS picks a free one).
    """
    servers: list[tuple[_Server, asyncio.Task[None]]] = []

    async def start(app: Any) -> str:
        # log_level critical: dropped-connection tests make uvicorn log an expected traceback.
        server = _Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="critical"))
        task = asyncio.create_task(server.serve())
        await asyncio.wait_for(server.ready.wait(), timeout=10)
        servers.append((server, task))
        port = server.servers[0].sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}"

    yield start
    for server, task in servers:
        server.should_exit = True
        await task
