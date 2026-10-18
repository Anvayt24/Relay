"""Server-Sent Events encoding, as used by OpenAI-compatible streams.

SSE is a text protocol over a single long HTTP response (``Content-Type: text/event-stream``).
Each event is one or more ``field: value`` lines followed by a blank line. OpenAI-style streams
use only ``data:``, and the blank line (``\\n\\n``) is what marks the end of an event:

    data: {"id": "...", "choices": [{"delta": {"content": "Hel"}}]}\\n\\n
    data: {"id": "...", "choices": [{"delta": {"content": "lo"}}]}\\n\\n
    data: [DONE]\\n\\n

A raw newline inside ``data`` would split the event. ``json.dumps`` escapes newlines in
strings (``"\\n"``), so one JSON object always fits on one ``data:`` line.

``[DONE]`` is OpenAI's convention, not part of SSE. It means "this stream completed
normally", which is why a broken stream must *not* send it (FR-5).
"""

import json
from collections.abc import Mapping
from typing import Any

from relay.api.errors import error_body
from relay.schemas.chat import ChatChunk
from relay.schemas.errors import ErrorClass

DONE = b"data: [DONE]\n\n"


def encode_event(payload: Mapping[str, Any]) -> bytes:
    return b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n"


def encode_chunk(chunk: ChatChunk, extra: Mapping[str, Any] | None = None) -> bytes:
    # exclude_none: chunks are sent many times per response, so absent fields stay absent
    # (no "usage": null on every chunk). The SDK fills in defaults.
    payload = chunk.model_dump(mode="json", exclude_none=True)
    if extra:
        payload.update(extra)
    return encode_event(payload)


def encode_error(error_class: ErrorClass, message: str) -> bytes:
    # The OpenAI SDK raises APIError when an event's JSON has an "error" key. A stream that
    # just stops, without [DONE] and without this, ends *silently* in the SDK: the client
    # would think a truncated answer was complete.
    return encode_event(error_body(error_class, message))
