import httpx


async def test_healthz_reports_ok(client: httpx.AsyncClient) -> None:
    resp = await client.get("/healthz")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_readyz_reports_ready_with_dependency_map(client: httpx.AsyncClient) -> None:
    resp = await client.get("/readyz")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    # Empty until dependencies exist; the key is part of the contract from day one.
    assert body["dependencies"] == {}


async def test_metrics_exposes_prometheus_text_format(client: httpx.AsyncClient) -> None:
    resp = await client.get("/metrics")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    # python_info is exported by prometheus_client's default registry on every platform.
    assert "python_info" in resp.text
