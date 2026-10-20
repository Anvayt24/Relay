"""The simulator's Anthropic wire format, checked with the official Anthropic SDK.

Phase 4's Anthropic adapter will be tested against this, so the details that differ from
OpenAI are what these tests pin down:
- system as a top-level field;
- usage split across stream events;
- named SSE events ending with message_stop;
- 529 for overload;
- errors arriving as an event mid-stream.
"""

from typing import Any

import anthropic
import httpx2
import pytest

from relay.schemas.errors import ErrorClass
from relay.simulator.engine import Simulator
from relay.simulator.plan import MockBehavior
from relay.simulator.quality import QualitySpec
from relay.simulator.scenario import Capacity, ModelProfile, OutputLength, Scenario
from relay.simulator.server import create_sim_app

MODEL = "sim-model"
MESSAGES: Any = [{"role": "user", "content": "hello there"}]


def _sampled(**scenario: Any) -> Simulator:
    profile = ModelProfile(length=OutputLength(4, 8), quality=QualitySpec(base=0.5))
    return Simulator.from_scenario(Scenario(models={MODEL: profile}, **scenario), seed=3)


def _sdk(simulator: Simulator) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        base_url="http://sim.test",
        api_key="sim-key",  # a dummy key: nothing here may ever reach a real provider
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(
            transport=httpx2.ASGITransport(app=create_sim_app(simulator))
        ),
    )


async def test_message() -> None:
    async with _sdk(Simulator.scripted()) as sdk:
        message = await sdk.messages.create(
            model=MODEL, max_tokens=50, system="be brief", messages=MESSAGES
        )
    assert message.id.startswith("msg_sim-0-")
    assert message.content[0].type == "text"
    assert message.content[0].text == "mock: hello there"
    assert message.stop_reason == "end_turn"
    # The top-level system prompt counts as input, as it does for Anthropic.
    assert message.usage.input_tokens == 4  # "be brief" + "hello there"
    assert message.usage.output_tokens == 3


async def test_max_tokens_stop_reason() -> None:
    async with _sdk(Simulator.scripted()) as sdk:
        message = await sdk.messages.create(model=MODEL, max_tokens=1, messages=MESSAGES)
    assert message.stop_reason == "max_tokens"


async def test_stream_events_and_split_usage() -> None:
    async with _sdk(_sampled()) as sdk:
        whole = await sdk.messages.create(
            model=MODEL,
            max_tokens=50,
            messages=MESSAGES,
            extra_headers={"X-Sim-Request-Index": "5"},
        )
        events = []
        async with sdk.messages.stream(
            model=MODEL,
            max_tokens=50,
            messages=MESSAGES,
            extra_headers={"X-Sim-Request-Index": "5"},
        ) as stream:
            async for event in stream:
                events.append(event)
            final = await stream.get_final_message()

    types = [e.type for e in events]
    assert types[0] == "message_start"
    assert types[-1] == "message_stop"
    assert types.index("content_block_stop") < types.index("message_delta")
    # Input usage arrives first, output usage last; the SDK stitches them together.
    start = events[0]
    assert start.type == "message_start"
    assert start.message.usage.input_tokens == 2
    assert start.message.usage.output_tokens == 0
    assert final.usage.output_tokens == whole.usage.output_tokens
    assert final.content[0].type == "text"
    assert whole.content[0].type == "text"
    assert final.content[0].text == whole.content[0].text


@pytest.mark.parametrize(
    ("error_class", "sdk_error", "status", "error_type"),
    [
        (ErrorClass.OVERLOADED, anthropic.OverloadedError, 529, "overloaded_error"),
        (ErrorClass.SERVER_ERROR, anthropic.InternalServerError, 500, "api_error"),
        (ErrorClass.RATE_LIMITED, anthropic.RateLimitError, 429, "rate_limit_error"),
        # Out of credit is a 400, told apart only by its message.
        (ErrorClass.QUOTA_EXHAUSTED, anthropic.BadRequestError, 400, "invalid_request_error"),
        (ErrorClass.MODEL_NOT_FOUND, anthropic.NotFoundError, 404, "not_found_error"),
        (ErrorClass.AUTH, anthropic.AuthenticationError, 401, "authentication_error"),
    ],
)
async def test_each_error_class_has_its_wire_shape(
    error_class: ErrorClass, sdk_error: type[anthropic.APIStatusError], status: int, error_type: str
) -> None:
    async with _sdk(Simulator.scripted(MockBehavior(error=error_class))) as sdk:
        with pytest.raises(sdk_error) as exc:
            await sdk.messages.create(model=MODEL, max_tokens=50, messages=MESSAGES)
    assert exc.value.status_code == status
    body: Any = exc.value.body
    assert body["error"]["type"] == error_type


async def test_quota_message_identifies_it() -> None:
    async with _sdk(Simulator.scripted(MockBehavior(error=ErrorClass.QUOTA_EXHAUSTED))) as sdk:
        with pytest.raises(anthropic.BadRequestError) as exc:
            await sdk.messages.create(model=MODEL, max_tokens=50, messages=MESSAGES)
    assert "credit balance is too low" in exc.value.message


async def test_capacity_429_has_retry_after_and_rate_limit_headers() -> None:
    async with _sdk(_sampled(capacity=Capacity(rpm=1, tpm=100_000))) as sdk:
        raw = await sdk.messages.with_raw_response.create(
            model=MODEL, max_tokens=50, messages=MESSAGES
        )
        assert raw.headers["anthropic-ratelimit-requests-remaining"] == "0"
        assert raw.headers["anthropic-ratelimit-requests-reset"].endswith("Z")
        with pytest.raises(anthropic.RateLimitError) as exc:
            await sdk.messages.create(model=MODEL, max_tokens=50, messages=MESSAGES)
    assert exc.value.response.headers["retry-after"] == "60"


async def test_mid_stream_failure_is_an_error_event() -> None:
    async with _sdk(Simulator.scripted(MockBehavior(fail_after_chunks=1))) as sdk:
        received = []
        with pytest.raises(anthropic.APIStatusError) as exc:
            async with sdk.messages.stream(model=MODEL, max_tokens=50, messages=MESSAGES) as stream:
                async for text in stream.text_stream:
                    received.append(text)
    # The stream had started (200), then the provider reported an overload in-band.
    assert received == ["mock:"]
    body: Any = exc.value.body
    assert body["error"]["type"] == "overloaded_error"


async def test_missing_max_tokens_is_400() -> None:
    transport = httpx2.ASGITransport(app=create_sim_app(Simulator.scripted()))
    async with httpx2.AsyncClient(transport=transport, base_url="http://sim.test") as client:
        response = await client.post("/v1/messages", json={"model": MODEL, "messages": MESSAGES})
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
