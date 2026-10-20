"""The whole path over real HTTP for the first time:

    OpenAI SDK ─HTTP─▶ Relay ─▶ real adapter ─HTTP─▶ provider simulator

Before Phase 4, Relay called the simulator in-process. Here Relay's ``mock`` provider is a
*real* HTTP adapter pointed at the simulator's wire format, so the adapter's request building,
SSE parsing and error mapping run inside the full pipeline, streaming included.
"""

from collections.abc import Callable
from typing import Any

import httpx
import httpx2
import openai
import pytest
from fastapi import FastAPI

from relay.api.app import create_app
from relay.providers.anthropic import AnthropicAdapter
from relay.providers.base import ProviderAdapter
from relay.providers.openai_compat import OpenAICompatAdapter
from relay.schemas.errors import ErrorClass
from relay.simulator.engine import Simulator
from relay.simulator.plan import MockBehavior
from relay.simulator.server import create_sim_app

from ..conftest import test_settings as relay_settings  # type: ignore[import-not-found]

MESSAGES: Any = [{"role": "user", "content": "hello there"}]


def _adapter(kind: str, sim: Simulator) -> ProviderAdapter:
    transport = httpx.ASGITransport(app=create_sim_app(sim), raise_app_exceptions=False)
    # Named "mock" so the registry's mock models route to it.
    if kind == "openai":
        return OpenAICompatAdapter(
            "mock", base_url="http://sim.test/v1", api_key="k", transport=transport
        )
    return AnthropicAdapter("mock", base_url="http://sim.test", api_key="k", transport=transport)


def _sdk(app: FastAPI) -> openai.AsyncOpenAI:
    http_client = openai.DefaultAsyncHttpxClient(transport=httpx2.ASGITransport(app=app))
    return openai.AsyncOpenAI(
        base_url="http://relay.test/v1", api_key="test-key", http_client=http_client
    )


@pytest.fixture(params=["openai", "anthropic"])
def relay_over(request: pytest.FixtureRequest) -> Callable[[Simulator], FastAPI]:
    def build(sim: Simulator) -> FastAPI:
        return create_app(relay_settings(), providers={"mock": _adapter(request.param, sim)})

    return build


async def test_stream_through_relay_and_a_real_adapter(
    relay_over: Callable[[Simulator], FastAPI],
) -> None:
    async with _sdk(relay_over(Simulator.scripted())) as sdk:
        stream = await sdk.chat.completions.create(
            model="relay/cheap",
            messages=MESSAGES,
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks = [c async for c in stream]
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == "mock: hello there"
    assert chunks[-1].usage is not None
    assert chunks[-1].model == "mock/cheap"  # Relay's name, never the upstream id


async def test_upstream_429_reaches_the_client_once(
    relay_over: Callable[[Simulator], FastAPI],
) -> None:
    sim = Simulator.scripted(MockBehavior(error=ErrorClass.RATE_LIMITED, retry_after_s=2))
    async with _sdk(relay_over(sim)) as sdk:
        with pytest.raises(openai.RateLimitError) as exc:
            await sdk.chat.completions.create(model="relay/cheap", messages=MESSAGES)
    assert exc.value.response.headers["retry-after"] == "2"
    # SDK retries left at default (2), yet one upstream call: Relay said x-should-retry: false,
    # and the adapter never retries on its own.
    assert sim.calls == 1


async def test_mid_stream_break_becomes_relays_error_event(
    relay_over: Callable[[Simulator], FastAPI],
) -> None:
    async with _sdk(relay_over(Simulator.scripted(MockBehavior(fail_after_chunks=1)))) as sdk:
        stream = await sdk.chat.completions.create(
            model="relay/cheap", messages=MESSAGES, stream=True
        )
        with pytest.raises(openai.APIError, match="stream broken"):
            async for _ in stream:
                pass
