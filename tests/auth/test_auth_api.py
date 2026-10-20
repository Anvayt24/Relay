"""Auth end to end: header -> key hash -> store -> tenant, and every failure's status."""

from collections.abc import AsyncIterator
from typing import Any

import httpx
import httpx2
import openai
import pytest
from fastapi import FastAPI

from relay.api.app import create_app
from relay.auth.keys import generate_key, hash_key
from relay.auth.store import CachedKeyStore, MemoryKeyStore
from relay.providers.registry import NO_KEYS

from ..conftest import TEST_PEPPER  # type: ignore[import-not-found]
from ..conftest import test_settings as relay_settings  # type: ignore[import-not-found]

CHAT = "/v1/chat/completions"
BODY = {"model": "relay/cheap", "messages": [{"role": "user", "content": "hi"}]}
KEY = generate_key()


def _app(store: MemoryKeyStore | CachedKeyStore, **settings: Any) -> FastAPI:
    return create_app(
        relay_settings(auth_disabled=False, **settings), provider_keys=NO_KEYS, key_store=store
    )


@pytest.fixture
def store() -> MemoryKeyStore:
    return MemoryKeyStore({hash_key(KEY, TEST_PEPPER): "test"})


@pytest.fixture
async def client(store: MemoryKeyStore) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=_app(store))
    async with httpx.AsyncClient(transport=transport, base_url="http://relay.test") as c:
        yield c


async def test_valid_key_through_the_openai_sdk(store: MemoryKeyStore) -> None:
    http_client = openai.DefaultAsyncHttpxClient(transport=httpx2.ASGITransport(app=_app(store)))
    async with openai.AsyncOpenAI(
        base_url="http://relay.test/v1", api_key=KEY, http_client=http_client
    ) as sdk:
        completion = await sdk.chat.completions.create(**BODY)  # type: ignore[call-overload]
    assert completion.choices[0].message.content == "mock: hi"


@pytest.mark.parametrize(
    "authorization",
    [None, "", "Bearer", f"Basic {KEY}", "Bearer sk-not-a-relay-key", f"Bearer {generate_key()}"],
)
async def test_bad_credentials_are_401(
    client: httpx.AsyncClient, authorization: str | None
) -> None:
    headers = {"Authorization": authorization} if authorization is not None else {}
    r = await client.post(CHAT, json=BODY, headers=headers)
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_api_key"
    assert r.headers["x-should-retry"] == "false"
    assert "x-request-id" in r.headers


async def test_auth_comes_before_body_validation(client: httpx.AsyncClient) -> None:
    # An unauthenticated caller learns nothing about the API, not even that the body is bad.
    r = await client.post(CHAT, json={"nonsense": True})
    assert r.status_code == 401


async def test_models_list_requires_auth(client: httpx.AsyncClient) -> None:
    assert (await client.get("/v1/models")).status_code == 401
    ok = await client.get("/v1/models", headers={"Authorization": f"Bearer {KEY}"})
    assert ok.status_code == 200


async def test_health_needs_no_auth(client: httpx.AsyncClient) -> None:
    assert (await client.get("/healthz")).status_code == 200


async def test_store_down_unknown_key_is_503_not_401(store: MemoryKeyStore) -> None:
    store.down = True
    transport = httpx.ASGITransport(app=_app(store))
    async with httpx.AsyncClient(transport=transport, base_url="http://relay.test") as c:
        r = await c.post(CHAT, json=BODY, headers={"Authorization": f"Bearer {KEY}"})
    # The key may be fine; we just can't check. 401 would tell a good client it's bad.
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "auth_unavailable"
    assert r.headers["retry-after"] == "5"


async def test_store_down_known_key_still_works(store: MemoryKeyStore) -> None:
    transport = httpx.ASGITransport(app=_app(store))
    headers = {"Authorization": f"Bearer {KEY}"}
    async with httpx.AsyncClient(transport=transport, base_url="http://relay.test") as c:
        assert (await c.post(CHAT, json=BODY, headers=headers)).status_code == 200
        store.down = True  # Postgres goes away after the key was seen
        assert (await c.post(CHAT, json=BODY, headers=headers)).status_code == 200


async def test_key_for_a_removed_tenant_is_401() -> None:
    store = MemoryKeyStore({hash_key(KEY, TEST_PEPPER): "deleted-tenant"})
    transport = httpx.ASGITransport(app=_app(store))
    async with httpx.AsyncClient(transport=transport, base_url="http://relay.test") as c:
        r = await c.post(CHAT, json=BODY, headers={"Authorization": f"Bearer {KEY}"})
    assert r.status_code == 401


def test_auth_disabled_is_refused_in_prod() -> None:
    with pytest.raises(ValueError, match="only allowed"):
        create_app(relay_settings(env="prod", auth_disabled=True), provider_keys=NO_KEYS)


def test_auth_on_requires_a_pepper() -> None:
    with pytest.raises(ValueError, match="RELAY_KEY_PEPPER"):
        create_app(relay_settings(auth_disabled=False, key_pepper=None), provider_keys=NO_KEYS)
