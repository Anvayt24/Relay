"""The adapter contract suite (PRD FR-2 acceptance): every HTTP adapter, the same checks.

Each adapter talks to the provider simulator's matching wire format, over HTTP but in-process
(``httpx.ASGITransport``). The simulator reproduces each provider's quirks, so passing here
means the adapter handles real wire behavior, without a network or a key.

Two cases need a real socket (``live_server``), because only a real connection can drop
mid-body or go silent: the OpenAI-format connection drop, and the idle timeout.
"""

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from relay.providers.anthropic import AnthropicAdapter
from relay.providers.base import ProviderAdapter
from relay.providers.openai_compat import OpenAICompatAdapter
from relay.schemas.chat import ChatChunk, ChatRequest
from relay.schemas.errors import ErrorClass, ProviderError
from relay.simulator.engine import Simulator
from relay.simulator.plan import MockBehavior
from relay.simulator.server import create_sim_app

KINDS = ["openai", "anthropic"]


def _adapter(
    kind: str, base_url: str, transport: httpx.AsyncBaseTransport | None, **kw: Any
) -> ProviderAdapter:
    if kind == "openai":
        return OpenAICompatAdapter(
            "sim", base_url=f"{base_url}/v1", api_key="sim-key", transport=transport, **kw
        )
    return AnthropicAdapter("sim", base_url=base_url, api_key="sim-key", transport=transport, **kw)


def _in_process(kind: str, sim: Simulator) -> ProviderAdapter:
    # raise_app_exceptions=False: when the simulator cuts a body short, the client sees a
    # truncated body (as over a real connection) instead of the server's exception.
    transport = httpx.ASGITransport(app=create_sim_app(sim), raise_app_exceptions=False)
    return _adapter(kind, "http://sim.test", transport)


def _request(content: str = "hello there", **extra: Any) -> ChatRequest:
    return ChatRequest.model_validate(
        {"model": "sim-model", "messages": [{"role": "user", "content": content}], **extra}
    )


async def _collect(adapter: ProviderAdapter, req: ChatRequest) -> list[ChatChunk]:
    return [chunk async for chunk in adapter.stream(req)]


def _text(chunks: list[ChatChunk]) -> str:
    return "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)


@pytest.fixture(params=KINDS)
def kind(request: pytest.FixtureRequest) -> str:
    return str(request.param)


async def test_complete(kind: str) -> None:
    adapter = _in_process(kind, Simulator.scripted())
    response = await adapter.complete(_request())
    await adapter.aclose()
    assert response.choices[0].message.content == "mock: hello there"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.completion_tokens == 3
    assert response.usage.prompt_tokens == 2


async def test_stream_shape_ends_with_usage(kind: str) -> None:
    adapter = _in_process(kind, Simulator.scripted())
    chunks = await _collect(adapter, _request())
    await adapter.aclose()
    assert _text(chunks) == "mock: hello there"
    assert chunks[-2].choices[0].finish_reason == "stop"
    # The adapter contract: the last chunk is always usage, with no choices.
    assert chunks[-1].choices == []
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.completion_tokens == 3


async def test_max_tokens_gives_length(kind: str) -> None:
    adapter = _in_process(kind, Simulator.scripted())
    response = await adapter.complete(_request(max_tokens=1))
    chunks = await _collect(adapter, _request(max_tokens=1))
    await adapter.aclose()
    assert (response.choices[0].message.content, response.choices[0].finish_reason) == (
        "mock:",
        "length",
    )
    assert chunks[-2].choices[0].finish_reason == "length"


@pytest.mark.parametrize(
    "error_class",
    [
        ErrorClass.RATE_LIMITED,
        ErrorClass.QUOTA_EXHAUSTED,
        ErrorClass.OVERLOADED,
        ErrorClass.SERVER_ERROR,
        ErrorClass.TIMEOUT,
        ErrorClass.BAD_REQUEST,
        ErrorClass.CONTEXT_LENGTH_EXCEEDED,
        ErrorClass.MODEL_NOT_FOUND,
        ErrorClass.AUTH,
        ErrorClass.CONTENT_FILTERED,
    ],
)
async def test_each_error_class_is_recognized(kind: str, error_class: ErrorClass) -> None:
    # The simulator renders each class the way that provider does on the wire (status, type,
    # code, message); the adapter must recover the class from that alone.
    sim = Simulator.scripted(MockBehavior(error=error_class))
    adapter = _in_process(kind, sim)
    with pytest.raises(ProviderError) as complete_exc:
        await adapter.complete(_request())
    with pytest.raises(ProviderError) as stream_exc:
        await anext(adapter.stream(_request()))
    await adapter.aclose()
    assert complete_exc.value.error_class is error_class
    assert stream_exc.value.error_class is error_class
    assert sim.calls == 2  # one per call: the adapter never retries (Relay owns retries)


async def test_429_carries_retry_after_and_scope(kind: str) -> None:
    sim = Simulator.scripted(MockBehavior(error=ErrorClass.RATE_LIMITED, retry_after_s=3))
    adapter = _in_process(kind, sim)
    with pytest.raises(ProviderError) as exc:
        await adapter.complete(_request())
    await adapter.aclose()
    assert exc.value.signal is not None
    assert exc.value.signal.retry_after_s == 3
    assert exc.value.signal.limit_scope == "model"


async def test_mid_stream_failure_is_stream_broken(kind: str) -> None:
    adapter = _in_process(kind, Simulator.scripted(MockBehavior(fail_after_chunks=1)))
    received: list[str] = []
    with pytest.raises(ProviderError) as exc:
        async for chunk in adapter.stream(_request()):
            received.append(chunk.choices[0].delta.content or "" if chunk.choices else "")
    await adapter.aclose()
    assert "".join(received) == "mock:"  # content arrived first, then the failure
    assert exc.value.error_class is ErrorClass.STREAM_BROKEN


async def test_connection_drop_on_a_real_socket(
    live_server: Callable[[Any], Any],
) -> None:
    base = await live_server(create_sim_app(Simulator.scripted(MockBehavior(fail_after_chunks=1))))
    adapter = _adapter("openai", base, None)
    with pytest.raises(ProviderError) as exc:
        await _collect(adapter, _request())
    await adapter.aclose()
    # httpx saw an incomplete chunked body after content: stream_broken, not server_error.
    assert exc.value.error_class is ErrorClass.STREAM_BROKEN


async def test_idle_upstream_times_out_as_stream_broken(
    kind: str, live_server: Callable[[Any], Any]
) -> None:
    base = await live_server(create_sim_app(Simulator.scripted(MockBehavior(stall_after_chunks=1))))
    adapter = _adapter(kind, base, None, stream_idle_timeout_s=0.3)
    with pytest.raises(ProviderError) as exc:
        await _collect(adapter, _request())
    await adapter.aclose()
    assert exc.value.error_class is ErrorClass.STREAM_BROKEN


async def test_unreachable_provider_is_server_error(kind: str) -> None:
    # Port 1 on localhost: nothing listens, the connection is refused at once.
    adapter = _adapter(kind, "http://127.0.0.1:1", None)
    with pytest.raises(ProviderError) as exc:
        await adapter.complete(_request())
    await adapter.aclose()
    assert exc.value.error_class is ErrorClass.SERVER_ERROR
