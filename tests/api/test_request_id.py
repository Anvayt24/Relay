import httpx
import pytest

from relay.api.request_id import resolve_request_id

CHAT = "/v1/chat/completions"
BODY = {"model": "relay/auto", "messages": [{"role": "user", "content": "hi"}]}


def test_valid_id_is_kept() -> None:
    assert resolve_request_id("abc-123_X.y") == "abc-123_X.y"


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "a" * 129,
        "has space",
        "line\r\nbreak",  # log injection: a fake second log line
        "semi;colon",
    ],
)
def test_invalid_or_missing_id_is_replaced(raw: str | None) -> None:
    generated = resolve_request_id(raw)
    assert generated != raw
    assert len(generated) == 32  # uuid4().hex


def test_max_length_id_is_kept() -> None:
    assert resolve_request_id("a" * 128) == "a" * 128


async def test_id_is_echoed(client: httpx.AsyncClient) -> None:
    r = await client.post(CHAT, json=BODY, headers={"X-Request-ID": "my-trace-1"})
    assert r.headers["x-request-id"] == "my-trace-1"


async def test_bad_id_is_replaced_in_response(client: httpx.AsyncClient) -> None:
    r = await client.post(CHAT, json=BODY, headers={"X-Request-ID": "bad id!"})
    assert r.headers["x-request-id"] != "bad id!"
    assert len(r.headers["x-request-id"]) == 32


async def test_id_is_generated_when_missing(client: httpx.AsyncClient) -> None:
    r = await client.get("/healthz")
    assert len(r.headers["x-request-id"]) == 32


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"model": "relay/auto"}, 400),  # missing messages
        ({**BODY, "model": "nope"}, 404),
    ],
)
async def test_id_is_present_on_errors(
    client: httpx.AsyncClient, body: dict[str, object], status: int
) -> None:
    r = await client.post(CHAT, json=body, headers={"X-Request-ID": "err-1"})
    assert r.status_code == status
    assert r.headers["x-request-id"] == "err-1"
