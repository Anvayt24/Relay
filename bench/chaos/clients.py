"""The clients being compared. Each turns one logical request into a ``Result``.

- ``relay_client``: one call to Relay, which owns retries and fallback.
- ``naive_retry_client``: what teams write without a gateway. Up to 5 tries against the
  primary with exponential backoff and **no jitter**, then one try against the fallback.
- ``direct_client``: one call to the primary, no retry, no fallback.

All three speak plain HTTP with httpx, so the measurement is of the *behavior*, with no SDK
in the loop. The OpenAI SDK is verified against Relay separately (tests/api).

LiteLLM is deliberately not here: the baseline is deferred.
"""

import asyncio  # noqa: F401  (tests patch `clients.asyncio.sleep`: the same module naive_retry sleeps with)
import json
import re
import time
from typing import Any

import httpx

from bench.chaos.metrics import Result
from relay.demo.naive import naive_retry

CHAT = "/v1/chat/completions"
INDEX_IN_ID = re.compile(r"chatcmpl-sim-(\d+)-")


def body(stream: bool, content: str = "hello") -> dict[str, Any]:
    return {
        "model": "relay/auto",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 20,
        "stream": stream,
        "stream_options": {"include_usage": True} if stream else None,
    }


async def _post(
    client: httpx.AsyncClient, url: str, payload: dict[str, Any], model: str | None = None
) -> tuple[int, dict[str, str], bool, bool, int | None]:
    """One HTTP call. Returns (status, headers, complete, silent_truncation, sim_index).

    ``complete`` means a full answer arrived: JSON parsed, or a stream that reached [DONE]
    without an error event.
    """
    payload = {k: v for k, v in payload.items() if v is not None}
    if model is not None:
        payload = {**payload, "model": model}
    try:
        if not payload.get("stream"):
            r = await client.post(url, json=payload)
            index = None
            try:
                data = r.json()
                index = _index(data.get("id", ""))
                complete = r.status_code == 200 and bool(data.get("choices"))
            except ValueError:
                complete = False
            return r.status_code, dict(r.headers), complete, False, index

        async with client.stream("POST", url, json=payload) as r:
            if r.status_code != 200:
                await r.aread()
                return r.status_code, dict(r.headers), False, False, None
            done = errored = False
            index = None
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                try:
                    event = json.loads(data)
                except ValueError:
                    errored = True
                    break
                if "error" in event:
                    errored = True
                    break
                if index is None:
                    index = _index(event.get("id", ""))
            complete = done and not errored
            return 200, dict(r.headers), complete, not done and not errored, index
    except (TimeoutError, httpx.HTTPError):
        # A reset or timeout mid-body: not complete, and *not* silent (the client saw it).
        return 0, {}, False, False, None


def _index(response_id: str) -> int | None:
    m = INDEX_IN_ID.match(response_id)
    return int(m.group(1)) if m else None


async def relay_client(
    client: httpx.AsyncClient, base: str, stream: bool, t0: float, content: str = "hello"
) -> Result:
    started = time.monotonic()
    status, headers, ok, silent, index = await _post(client, base + CHAT, body(stream, content))
    served_by = headers.get("x-relay-provider")
    return Result(
        ok=ok,
        latency_s=time.monotonic() - started,
        status=status,
        degraded=headers.get("x-relay-degraded") == "true",
        silent_truncation=silent,
        stream=stream,
        served_by=served_by,
        index=index,
        started_s=started - t0,
    )


async def direct_client(
    client: httpx.AsyncClient, primary: str, model: str, stream: bool, t0: float
) -> Result:
    started = time.monotonic()
    status, _, ok, silent, index = await _post(client, primary + CHAT, body(stream), model)
    return Result(
        ok=ok,
        latency_s=time.monotonic() - started,
        status=status,
        silent_truncation=silent,
        stream=stream,
        served_by="primary",
        index=index,
        started_s=started - t0,
    )


async def naive_retry_client(
    client: httpx.AsyncClient,
    primary: str,
    fallback: str,
    primary_model: str,
    fallback_model: str,
    stream: bool,
    t0: float,
) -> Result:
    started = time.monotonic()

    async def try_primary() -> tuple[bool, tuple[int, bool, int | None]]:
        status, _, ok, silent, index = await _post(
            client, primary + CHAT, body(stream), primary_model
        )
        return ok, (status, silent, index)

    async def try_fallback() -> tuple[bool, tuple[int, bool, int | None]]:
        status, _, ok, silent, index = await _post(
            client, fallback + CHAT, body(stream), fallback_model
        )
        return ok, (status, silent, index)

    ok, (status, silent, index), served_by = await naive_retry(try_primary, try_fallback)
    return Result(
        ok=ok,
        latency_s=time.monotonic() - started,
        status=status,
        # A success on the primary was never counted as a truncation; the fallback's result
        # reports whatever the fallback did.
        silent_truncation=silent if served_by == "fallback" else False,
        stream=stream,
        served_by=served_by,
        index=index,
        started_s=started - t0,
    )
