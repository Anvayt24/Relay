"""Plumbing shared by the HTTP adapters: the client, SSE parsing, rate-limit headers, and
turning what went wrong into a ``ProviderError``.

Everything here is about *reading what a provider sent* defensively. Providers are other
people's servers: they add comment lines, send errors as lists, cut connections, and
change header formats. Each helper below handles one of those, so the adapters stay about
translation.
"""

import email.utils
import json
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from relay.config.models import LimitScope
from relay.schemas.errors import ErrorClass, ProviderError, ProviderSignal

# Valid JSON of the wrong shape (a string where an object belongs, a missing key, a bad
# number). Caught around all parsing of a provider's body, so it becomes a ProviderError that
# fails over and feeds the breaker, instead of a 500 the breaker never hears about.
MALFORMED = (ValueError, KeyError, TypeError, IndexError, AttributeError)

# Errors that say "back off": they carry a signal so admission pauses the right scope (FR-4).
_LIMITS = frozenset({ErrorClass.RATE_LIMITED, ErrorClass.QUOTA_EXHAUSTED, ErrorClass.OVERLOADED})


def pooled_client(
    base_url: str,
    headers: Mapping[str, str],
    *,
    connect_timeout_s: float,
    read_timeout_s: float,
    transport: httpx.AsyncBaseTransport | None,
) -> httpx.AsyncClient:
    """One pooled client per adapter, reused by every request: TCP and TLS setup happen once
    per connection, not once per call. No I/O here; the adapter closes it in ``aclose()``.

    The default read timeout is for non-streaming calls: the provider sends nothing until
    the whole answer is ready, so it must cover the full generation."""
    return httpx.AsyncClient(
        base_url=base_url,
        headers=dict(headers),
        timeout=httpx.Timeout(
            connect=connect_timeout_s, read=read_timeout_s, write=10.0, pool=connect_timeout_s
        ),
        transport=transport,
    )


def stream_timeout(*, connect_timeout_s: float, idle_timeout_s: float) -> httpx.Timeout:
    """For streams the read timeout is the longest silence between two chunks, not the whole
    answer: a stream that stalls is cut, a long one that keeps talking is not."""
    return httpx.Timeout(
        connect=connect_timeout_s, read=idle_timeout_s, write=10.0, pool=connect_timeout_s
    )


def upstream_error(
    provider: str,
    status: int,
    error: Mapping[str, Any],
    error_class: ErrorClass,
    headers: Mapping[str, str],
    limit_scope: LimitScope,
) -> ProviderError:
    """The ``ProviderError`` for an HTTP error response, once classified. Written once for
    every adapter (found in review: Anthropic's copy sent no signal for a quota error)."""
    signal = None
    if error_class in _LIMITS:
        signal = ProviderSignal(retry_after_s=retry_after(headers), limit_scope=limit_scope)
    message = f"{provider} {status}: {str(error.get('message', ''))[:300]}"
    return ProviderError(error_class, message, signal)


def stream_failure(content_seen: bool, message: str) -> ProviderError:
    """A stream that went wrong. Before any content it is an ordinary upstream fault (it can
    fall back); after it, the answer is incomplete and Relay never switches models, so it is
    ``stream_broken`` (Appendix B, FR-5)."""
    return ProviderError(
        ErrorClass.STREAM_BROKEN if content_seen else ErrorClass.SERVER_ERROR, message
    )


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
