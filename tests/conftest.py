from collections.abc import AsyncIterator

import httpx
import pytest

from relay.api.app import create_app
from relay.settings import Settings


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    """An HTTP client wired straight into the ASGI app: no server, no ports, no network.

    ASGITransport calls the app in-process, so tests are fast and deterministic, yet they
    still exercise the real HTTP layer (routing, headers, status codes, serialization).
    """
    app = create_app(Settings(env="test"))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://relay.test") as c:
        yield c
