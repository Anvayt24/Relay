"""Metrics may be served on their own port, which is not the public one.

`/metrics` shows tenants, models and spend, so it must never share a published port with the
API. With RELAY_METRICS_PORT set, the API's own `/metrics` goes away and the same numbers are
served on that port instead (compose keeps it internal and Prometheus scrapes it there).
"""

import socket

import httpx

from relay.api.app import create_app
from relay.providers.mock import MockProvider

from ..conftest import test_settings as relay_settings  # type: ignore[import-not-found]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def test_metrics_move_to_their_own_port_when_asked() -> None:
    port = free_port()
    app = create_app(relay_settings(metrics_port=port), providers={"mock": MockProvider()})
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://relay.test") as api:
            await api.post(
                "/v1/chat/completions",
                json={"model": "relay/cheap", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert (await api.get("/metrics")).status_code == 404  # not on the API port
        async with httpx.AsyncClient() as c:
            scraped = await c.get(f"http://127.0.0.1:{port}/metrics")
        assert scraped.status_code == 200 and "relay_requests_total" in scraped.text
    # Shutdown closes it.
    with socket.socket() as s:
        assert s.connect_ex(("127.0.0.1", port)) != 0


async def test_without_a_metrics_port_the_api_still_serves_metrics() -> None:
    app = create_app(relay_settings(), providers={"mock": MockProvider()})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://relay.test") as api:
        assert (await api.get("/metrics")).status_code == 200
