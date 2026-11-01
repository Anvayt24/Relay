"""Structured logs that carry the request (PRD FR-8: "request ID propagated through logs").

Every log record emitted while a request is being handled gets that request's
``request_id``, and the ``trace_id`` / ``span_id`` of the current span, so a log line leads
straight to the request's trace in Tempo and its row in Postgres.

How it gets there without passing anything around:
- ``REQUEST_ID`` is a ``ContextVar``. The request-ID middleware sets it for the duration of the
  request; asyncio copies context into tasks, so everything the request awaits sees it.
- ``RequestContextFilter`` reads it (and the current OTel span) and puts the ids on every
  record, whichever logger emitted it.
- ``JsonFormatter`` writes one JSON object per line: easy to ship, grep and parse.

``RELAY_LOG_FORMAT=text`` keeps human-readable lines (the ids at the end) for local runs.
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from opentelemetry import trace

if TYPE_CHECKING:
    from relay.settings import Settings

REQUEST_ID: ContextVar[str | None] = ContextVar("relay_request_id", default=None)

# The loggers uvicorn configures itself (with propagate=False): ours replace their handlers.
_UVICORN = ("uvicorn", "uvicorn.error", "uvicorn.access")
# Libraries that log every HTTP call at INFO. Logging is synchronous I/O, and one line per
# upstream call would put logging on the request path (FR-8), so they are held at WARNING.
_CHATTY = ("httpx", "httpcore", "huggingface_hub", "urllib3")


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = REQUEST_ID.get()
        span = trace.get_current_span().get_span_context()
        record.trace_id = format(span.trace_id, "032x") if span.is_valid else None
        record.span_id = format(span.span_id, "016x") if span.is_valid else None
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
            "trace_id": getattr(record, "trace_id", None),
            "span_id": getattr(record, "span_id", None),
        }
        if record.exc_info:
            out["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False)


TEXT_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s: %(message)s "
    "[request_id=%(request_id)s trace_id=%(trace_id)s]"
)


class _StderrHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Writes to whatever ``sys.stderr`` is *now*. A plain StreamHandler keeps the stream it
    was built with; anything that swaps stderr (test runners, some process managers) would
    then leave it writing to a closed file."""

    @property
    def stream(self) -> object:
        return sys.stderr

    @stream.setter
    def stream(self, value: object) -> None:
        return None


def make_handler(fmt: str) -> logging.Handler:
    handler = _StderrHandler()
    handler.addFilter(RequestContextFilter())
    handler.setFormatter(JsonFormatter() if fmt == "json" else logging.Formatter(TEXT_FORMAT))
    handler.set_name("relay")
    return handler


def configure_logging(settings: "Settings") -> None:
    """Install Relay's handler on the root logger and on uvicorn's own loggers. Idempotent:
    a second app in the same process (tests) replaces the handler instead of adding one."""
    handler = make_handler(settings.log_format)
    level = logging.getLevelName(settings.log_level.upper())
    for name in ("", *_UVICORN):
        log = logging.getLogger(name)
        for old in [h for h in log.handlers if h.get_name() == "relay" or name in _UVICORN]:
            log.removeHandler(old)
        log.addHandler(handler)
        if name == "":
            log.setLevel(level)
    for name in _CHATTY:
        logging.getLogger(name).setLevel(logging.WARNING)
