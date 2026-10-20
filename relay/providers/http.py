"""Plumbing shared by the HTTP adapters: SSE parsing, rate-limit headers, transport errors.

Everything here is about *reading what a provider sent* defensively. Providers are other
people's servers: they add comment lines, send errors as lists, cut connections, and
change header formats. Each helper below handles one of those, so the adapters stay about
translation.
"""

import email.utils
import json
import math
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from relay.schemas.errors import ErrorClass, ProviderError


@dataclass(frozen=True)
class SSEEvent:
    event: str | None  # the `event:` field; OpenAI-style streams never set it, Anthropic always
    data: str


async def sse_events(lines: AsyncIterator[str]) -> AsyncIterator[SSEEvent]:
    """Parse a Server-Sent Events stream into events.

    The SSE rules that matter here:
    - an event is a group of ``field: value`` lines ended by a blank line;
    - several ``data:`` lines in one event are joined with ``\\n``;
    - lines starting with ``:`` are comments. OpenRouter sends ``: OPENROUTER PROCESSING`` as
      a keep-alive while a slow model thinks; a naive ``json.loads`` on every line would choke.
    """
    event: str | None = None
    data: list[str] = []
    async for line in lines:
        if line == "":
            if data:
                yield SSEEvent(event, "\n".join(data))
            event, data = None, []
        elif line.startswith(":"):
            continue
        else:
            name, _, value = line.partition(":")
            value = value.removeprefix(" ")  # exactly one optional space after the colon
            if name == "data":
                data.append(value)
            elif name == "event":
                event = value
            # `id:` and `retry:` are for browser reconnects; no provider relies on them.
    if data:  # a final event not followed by a blank line
        yield SSEEvent(event, "\n".join(data))


def retry_after(headers: Mapping[str, str]) -> float | None:
    """Seconds to wait, from ``retry-after-ms`` (preferred, more precise) or ``retry-after``.

    ``retry-after`` may be a number of seconds *or* an HTTP date (RFC 9110); both happen.
    """
    ms = headers.get("retry-after-ms")
    if ms is not None:
        try:
            return max(0.0, float(ms) / 1000)
        except ValueError:
            pass
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - time.time())


def int_header(headers: Mapping[str, str], *names: str) -> int | None:
    """The first of ``names`` present as an integer. Providers name the same thing differently
    (``x-ratelimit-remaining-requests`` vs ``anthropic-ratelimit-requests-remaining``)."""
    for name in names:
        value = headers.get(name)
        if value is not None:
            try:
                return math.floor(float(value))
            except ValueError:
                continue
    return None


def error_payload(body: bytes) -> tuple[dict[str, Any], str]:
    """(the error object, a lowercase blob for keyword matching) from an error response body.

    Shapes seen in the wild:
    - ``{"error": {"message", "type", "code"}}`` (OpenAI, Groq, OpenRouter);
    - ``[{"error": {...}}]``: a *list* (Gemini's compatibility endpoint);
    - ``{"type": "error", "error": {"type", "message"}}`` (Anthropic);
    - plain text or HTML from a proxy in front of the provider.
    """
    text = body.decode("utf-8", errors="replace")
    try:
        parsed: Any = json.loads(text)
    except ValueError:
        return {"message": text[:500]}, text.lower()
    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    error = parsed.get("error", parsed) if isinstance(parsed, dict) else {"message": str(parsed)}
    if not isinstance(error, dict):
        error = {"message": str(error)}
    blob = " ".join(str(error.get(k, "")) for k in ("type", "code", "status", "message"))
    return error, blob.lower()


def transport_error(exc: httpx.HTTPError, *, content_started: bool) -> ProviderError:
    """Map a connection-level failure (no HTTP status to look at) to Appendix B.

    The same exception means different things depending on *when* it happens:
    - before any content: a timeout is ``timeout`` (fall back, PRD FR-5), anything else is
      ``server_error``;
    - after content started: anything is ``stream_broken``, because the answer is now
      incomplete and, after the first byte, Relay never switches models (FR-5).
    """
    if content_started:
        return ProviderError(ErrorClass.STREAM_BROKEN, f"connection failed mid-stream: {exc!r}")
    if isinstance(exc, httpx.TimeoutException):
        return ProviderError(ErrorClass.TIMEOUT, f"upstream timed out: {exc!r}")
    return ProviderError(ErrorClass.SERVER_ERROR, f"upstream connection failed: {exc!r}")
