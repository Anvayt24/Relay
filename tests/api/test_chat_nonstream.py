from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from prometheus_client import REGISTRY

from relay.providers.mock import MockBehavior, MockProvider
from relay.schemas.errors import ErrorClass

BASE_URL = "http://relay.test"
CHAT = "/v1/chat/completions"


def _body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "relay/auto",
        "messages": [{"role": "user", "content": "hello relay"}],
    }
    body.update(overrides)
    return body


@pytest.fixture
def client_for(
    make_app: Callable[[MockProvider], FastAPI],
) -> Callable[[MockProvider], httpx.AsyncClient]:
    def _client(provider: MockProvider) -> httpx.AsyncClient:
        transport = httpx.ASGITransport(app=make_app(provider))
        return httpx.AsyncClient(transport=transport, base_url=BASE_URL)

    return _client


@pytest.fixture
async def flaky_client(
    client_for: Callable[[MockProvider], httpx.AsyncClient],
) -> AsyncIterator[httpx.AsyncClient]:
    provider = MockProvider(MockBehavior(error=ErrorClass.RATE_LIMITED, retry_after_s=1.2))
    async with client_for(provider) as c:
        yield c


async def test_happy_path(client: httpx.AsyncClient) -> None:
    r = await client.post(CHAT, json=_body())

    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    assert data["id"].startswith("chatcmpl-")
    assert data["model"] == "mock/strong"
    assert data["choices"][0]["message"] == {"role": "assistant", "content": "mock: hello relay"}
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["total_tokens"] == (
        data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]
    )


async def test_relay_headers_show_what_served_the_request(client: httpx.AsyncClient) -> None:
    r = await client.post(CHAT, json=_body(model="relay/cheap"))
    assert r.headers["x-relay-route"] == "default"
    assert r.headers["x-relay-provider"] == "mock"
    assert r.headers["x-relay-model"] == "mock/cheap"


async def test_validation_error_is_400_in_openai_envelope(client: httpx.AsyncClient) -> None:
    r = await client.post(CHAT, json={"model": "relay/auto"})

    # FastAPI's default is 422 with {"detail": [...]}; OpenAI clients expect 400 + "error".
    assert r.status_code == 400
    error = r.json()["error"]
    assert error["type"] == "invalid_request"
    assert error["param"] == "messages"
    assert error["message"]
    assert r.headers["x-should-retry"] == "false"


async def test_malformed_json_is_400(client: httpx.AsyncClient) -> None:
    r = await client.post(CHAT, content=b'{"model": ', headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request"
    assert r.json()["error"]["param"] is None


async def test_unsupported_parameter_names_the_param(client: httpx.AsyncClient) -> None:
    r = await client.post(CHAT, json=_body(tools=[{"type": "function"}]))
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "unsupported_parameter"
    assert r.json()["error"]["param"] == "tools"


async def test_nested_param_path(client: httpx.AsyncClient) -> None:
    r = await client.post(CHAT, json=_body(messages=[{"role": "tool", "content": "x"}]))
    assert r.json()["error"]["param"] == "messages[0].role"


async def test_param_from_model_level_validator(client: httpx.AsyncClient) -> None:
    r = await client.post(CHAT, json=_body(max_tokens=5, max_completion_tokens=6))
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "max_completion_tokens"


async def test_unknown_model_is_404(client: httpx.AsyncClient) -> None:
    r = await client.post(CHAT, json=_body(model="nope/nothing"))
    assert r.status_code == 404
    assert r.json()["error"]["type"] == "model_not_found"
    assert r.json()["error"]["param"] == "model"


async def test_provider_error_is_mapped_and_not_retryable(
    flaky_client: httpx.AsyncClient,
) -> None:
    r = await flaky_client.post(CHAT, json=_body())

    assert r.status_code == 429
    assert r.json()["error"]["type"] == "rate_limited"
    assert r.headers["x-should-retry"] == "false"
    # Retry-After is whole seconds (RFC 9110), rounded up so the client never comes back early.
    assert r.headers["retry-after"] == "2"


async def test_upstream_auth_failure_is_502_not_401(
    client_for: Callable[[MockProvider], httpx.AsyncClient],
) -> None:
    async with client_for(MockProvider(MockBehavior(error=ErrorClass.AUTH))) as c:
        r = await c.post(CHAT, json=_body())
    assert r.status_code == 502
    assert r.json()["error"]["type"] == "auth"


async def test_unknown_fields_are_counted(client: httpx.AsyncClient) -> None:
    before = REGISTRY.get_sample_value("relay_request_ignored_fields_total") or 0.0
    r = await client.post(CHAT, json=_body(logit_bias={}, presence_penalty=0.1))
    assert r.status_code == 200
    assert REGISTRY.get_sample_value("relay_request_ignored_fields_total") == before + 2


async def test_priority_header(client: httpx.AsyncClient) -> None:
    ok = await client.post(CHAT, json=_body(), headers={"X-Relay-Priority": "batch"})
    assert ok.status_code == 200

    bad = await client.post(CHAT, json=_body(), headers={"X-Relay-Priority": "urgent"})
    assert bad.status_code == 400
    assert bad.json()["error"]["param"] == "x-relay-priority"


async def test_unknown_route_uses_the_envelope(client: httpx.AsyncClient) -> None:
    r = await client.get("/v1/nothing-here")
    assert r.status_code == 404
    assert "error" in r.json()


async def test_unexpected_exception_is_500_without_internals(
    make_app: Callable[[MockProvider], FastAPI],
) -> None:
    class ExplodingProvider(MockProvider):
        async def complete(self, req: Any) -> Any:
            raise RuntimeError("secret internal detail")

    # raise_app_exceptions=False: Starlette re-raises after sending the 500 (so the server
    # logs it); here we want to look at the response the client would get instead.
    transport = httpx.ASGITransport(app=make_app(ExplodingProvider()), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as c:
        r = await c.post(CHAT, json=_body(), headers={"X-Request-ID": "boom-1"})

    assert r.status_code == 500
    assert r.json()["error"]["type"] == "internal_error"
    assert "secret" not in r.text
    assert r.headers["x-request-id"] == "boom-1"
    assert r.headers["x-should-retry"] == "false"


async def test_list_models(client: httpx.AsyncClient) -> None:
    r = await client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    ids = {m["id"]: m["owned_by"] for m in data["data"]}
    assert ids["relay/auto"] == "relay"
    assert ids["mock/strong"] == "mock"
    assert all(m["object"] == "model" for m in data["data"])
