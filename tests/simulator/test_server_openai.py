"""The simulator's OpenAI wire format, checked with the official OpenAI SDK.

If the real SDK can parse every response, stream and error, then Phase 4's adapters (and
LiteLLM and the naive client in Phase 8) can talk to the simulator as they would to a real
provider. SDK retries are off (max_retries=0) so each test sees exactly one attempt.

Most tests run in-process through httpx2's ASGITransport. The connection-drop tests need a
real socket, because "the server closed the connection mid-body" only exists on a real
connection; they run uvicorn on 127.0.0.1 with an ephemeral port.
"""

from typing import Any

import httpx2
import openai
import pytest

from relay.schemas.errors import ErrorClass
from relay.simulator.engine import Simulator
from relay.simulator.plan import MockBehavior
from relay.simulator.quality import QualitySpec
from relay.simulator.scenario import Capacity, ModelProfile, OutputLength, Scenario
from relay.simulator.server import create_sim_app

MODEL = "sim-model"
MESSAGES: Any = [{"role": "user", "content": "hello there"}]


class _FrozenClock:
    # Capacity refills with time. A frozen clock makes the x-ratelimit numbers exact instead
    # of depending on how many milliseconds passed between two requests on a busy machine.
    def now(self) -> float:
        return 0.0


def _sampled(**scenario: Any) -> Simulator:
    profile = ModelProfile(length=OutputLength(4, 8), quality=QualitySpec(base=0.5))
    return Simulator.from_scenario(
        Scenario(models={MODEL: profile}, **scenario), seed=3, clock=_FrozenClock()
    )


def _sdk(simulator: Simulator, base_url: str = "http://sim.test/v1") -> openai.AsyncOpenAI:
    transport = (
        None
        if not base_url.startswith("http://sim.test")
        else httpx2.ASGITransport(app=create_sim_app(simulator))
    )
    return openai.AsyncOpenAI(
        base_url=base_url,
        api_key="sim-key",  # a dummy key: nothing here may ever reach a real provider
        max_retries=0,
        http_client=openai.DefaultAsyncHttpxClient(transport=transport),
    )


async def test_completion(make_request: Any) -> None:
    sim = _sampled()
    async with _sdk(sim) as sdk:
        raw = await sdk.chat.completions.with_raw_response.create(model=MODEL, messages=MESSAGES)
    completion = raw.parse()
    assert completion.id.startswith("chatcmpl-sim-0-")
    assert raw.headers["x-sim-request-index"] == "0"
    assert completion.choices[0].message.content
    assert completion.usage is not None
    assert completion.usage.prompt_tokens == 2
    # The same request through the engine directly gives the same reply: one codebase.
    direct = await sim.open(make_request(MODEL), index=0).complete()
    assert completion.choices[0].message.content == direct.choices[0].message.content


async def test_stream_matches_the_completion_for_the_same_index() -> None:
    async with _sdk(_sampled()) as sdk:
        completion = await sdk.chat.completions.create(
            model=MODEL, messages=MESSAGES, extra_headers={"X-Sim-Request-Index": "7"}
        )
        stream = await sdk.chat.completions.create(
            model=MODEL,
            messages=MESSAGES,
            stream=True,
            stream_options={"include_usage": True},
            extra_headers={"X-Sim-Request-Index": "7"},
        )
        chunks = [chunk async for chunk in stream]
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == completion.choices[0].message.content
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.completion_tokens == completion.usage.completion_tokens  # type: ignore[union-attr]


async def test_usage_chunk_only_when_asked() -> None:
    async with _sdk(_sampled()) as sdk:
        stream = await sdk.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True)
        chunks = [chunk async for chunk in stream]
    assert all(c.usage is None for c in chunks)


async def test_capacity_429_has_retry_after_and_rate_limit_headers() -> None:
    async with _sdk(_sampled(capacity=Capacity(rpm=1, tpm=100_000))) as sdk:
        raw = await sdk.chat.completions.with_raw_response.create(model=MODEL, messages=MESSAGES)
        assert raw.headers["x-ratelimit-remaining-requests"] == "0"
        assert raw.headers["x-ratelimit-limit-requests"] == "1"
        with pytest.raises(openai.RateLimitError) as exc:
            await sdk.chat.completions.create(model=MODEL, messages=MESSAGES)
    headers = exc.value.response.headers
    assert headers["retry-after"] == "60"  # 1 request per minute: a full minute to refill
    assert headers["retry-after-ms"] == "60000"
    assert headers["x-ratelimit-reset-requests"] == "1m0s"
    assert exc.value.code == "rate_limit_exceeded"


@pytest.mark.parametrize(
    ("error_class", "sdk_error", "status", "code"),
    [
        (ErrorClass.QUOTA_EXHAUSTED, openai.RateLimitError, 429, "insufficient_quota"),
        (ErrorClass.OVERLOADED, openai.InternalServerError, 503, None),
        (ErrorClass.SERVER_ERROR, openai.InternalServerError, 500, None),
        (ErrorClass.TIMEOUT, openai.InternalServerError, 504, None),
        (
            ErrorClass.CONTEXT_LENGTH_EXCEEDED,
            openai.BadRequestError,
            400,
            "context_length_exceeded",
        ),
        (ErrorClass.CONTENT_FILTERED, openai.BadRequestError, 400, "content_filter"),
        (ErrorClass.MODEL_NOT_FOUND, openai.NotFoundError, 404, "model_not_found"),
        (ErrorClass.AUTH, openai.AuthenticationError, 401, "invalid_api_key"),
    ],
)
async def test_each_error_class_has_its_wire_shape(
    error_class: ErrorClass, sdk_error: type[openai.APIStatusError], status: int, code: str | None
) -> None:
    async with _sdk(Simulator.scripted(MockBehavior(error=error_class))) as sdk:
        with pytest.raises(sdk_error) as exc:
            await sdk.chat.completions.create(model=MODEL, messages=MESSAGES)
    assert exc.value.status_code == status
    assert exc.value.code == code


async def test_an_error_before_the_first_chunk_is_a_status_not_a_stream() -> None:
    async with _sdk(Simulator.scripted(MockBehavior(error=ErrorClass.OVERLOADED))) as sdk:
        with pytest.raises(openai.InternalServerError) as exc:
            await sdk.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True)
    assert exc.value.status_code == 503


async def test_unknown_model_in_a_scenario_is_404() -> None:
    async with _sdk(_sampled()) as sdk:
        with pytest.raises(openai.NotFoundError):
            await sdk.chat.completions.create(model="no-such-model", messages=MESSAGES)


async def test_bad_body_is_400() -> None:
    transport = httpx2.ASGITransport(app=create_sim_app(_sampled()))
    async with httpx2.AsyncClient(transport=transport, base_url="http://sim.test") as client:
        response = await client.post("/v1/chat/completions", json={"model": MODEL})
        bad_index = await client.post(
            "/v1/chat/completions",
            json={"model": MODEL, "messages": MESSAGES},
            headers={"X-Sim-Request-Index": "abc"},
        )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert bad_index.status_code == 400


async def test_truth_endpoint_matches_the_engine() -> None:
    sim = _sampled()
    transport = httpx2.ASGITransport(app=create_sim_app(sim))
    async with httpx2.AsyncClient(transport=transport, base_url="http://sim.test") as client:
        answers = [
            (await client.get("/sim/truth", params={"model": MODEL, "index": i})).json()
            for i in range(10)
        ]
        unknown = await client.get("/sim/truth", params={"model": "nope", "index": 0})
        models = (await client.get("/v1/models")).json()
    assert [a["correct"] for a in answers] == [sim.truth(MODEL, "default", i) for i in range(10)]
    assert answers[0]["q"] == 0.5
    assert unknown.status_code == 404
    assert [m["id"] for m in models["data"]] == [MODEL]


# ---------------------------------------------------------------- real socket
# `live_server` (tests/conftest.py) runs an app under uvicorn on a free localhost port.


async def test_mid_stream_failure_drops_the_connection(live_server: Any) -> None:
    base_url = (
        await live_server(create_sim_app(Simulator.scripted(MockBehavior(fail_after_chunks=2))))
        + "/v1"
    )
    received: list[str] = []
    async with _sdk(Simulator.scripted(), base_url) as sdk:
        stream = await sdk.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True)
        with pytest.raises(openai.APIConnectionError) as exc:
            async for chunk in stream:
                received.append(chunk.choices[0].delta.content or "")
    # The client got the first two words, then the connection was cut: no [DONE], no error
    # event. Underneath, the HTTP client saw an incomplete chunked body.
    assert "".join(received) == "mock: hello"
    assert isinstance(exc.value.__cause__, httpx2.RemoteProtocolError)


async def test_non_streaming_body_can_be_cut_off(live_server: Any) -> None:
    base_url = (
        await live_server(
            create_sim_app(Simulator.scripted(MockBehavior(error=ErrorClass.STREAM_BROKEN)))
        )
        + "/v1"
    )
    async with _sdk(Simulator.scripted(), base_url) as sdk:
        with pytest.raises(openai.APIConnectionError):
            await sdk.chat.completions.create(model=MODEL, messages=MESSAGES)
