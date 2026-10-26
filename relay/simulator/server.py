"""HTTP mode: the simulator as a network service (PRD §11.1, "wire-compatible").

Relay's real adapters (Phase 4), the naive retry client and LiteLLM (Phase 8) all talk HTTP.
To compare them fairly they must hit **the same endpoints with the same scenario**, so the
simulator serves both provider formats over HTTP:

- ``POST /v1/chat/completions``: OpenAI's format (also Groq's and OpenRouter's);
- ``POST /v1/messages``: Anthropic's format.

Both routes share one flow (``_serve``) and differ only in the translation (``openai_wire``,
``anthropic_wire``). The flow mirrors how a real provider fails:

1. **Before the response starts** (bad body, unknown model, capacity 429, or an injected error
   before the first token): an ordinary HTTP error with a status, a body and headers.
2. **After it starts** (a mid-stream failure): there is no status left to send.
   - OpenAI's format: the connection just drops (``ConnectionDropped``). The client sees an
     incomplete body.
   - Anthropic's format: an ``event: error`` frame, then the stream ends.

As in Relay's own streaming (Phase 2), the first chunk is pulled *before* answering: that is
what decides between case 1 and case 2.

Simulator-only headers (a real provider would ignore them, and so does Relay):
- ``X-Sim-Request-Index``: use this index instead of the next one. A paired experiment sends
  the candidate and reference calls with the same index.
- ``X-Sim-Cohort``: the cohort in ``q[model, cohort, t]``.

The response carries ``x-sim-request-index``, and ``GET /sim/truth`` returns the hidden
correctness for an (index, model, cohort): the answer key that only the experiment harness
reads.

Run it with ``python -m relay.simulator --scenario realistic --seed 0``.
"""

from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from relay.api.sse import DONE, encode_event
from relay.schemas.chat import ChatChunk, ChatRequest, ChatResponse
from relay.schemas.errors import ErrorClass, ProviderError
from relay.simulator import anthropic_wire, openai_wire
from relay.simulator.capacity import LimitState
from relay.simulator.engine import Call, CapacityExceeded, Simulator
from relay.simulator.scenario import ErrorMix

ErrorFn = Callable[[ProviderError, LimitState | None], tuple[int, dict[str, Any], dict[str, str]]]


class ConnectionDropped(Exception):
    """Raised inside a response body to cut the connection partway, as a crashing upstream
    does. The server has already sent the status and some bytes, so the only thing it can
    still do is close the socket. Uvicorn does that, and logs the exception: that log line is
    expected."""


@dataclass(frozen=True)
class _Wire:
    error: ErrorFn
    rate_limit_headers: Callable[[LimitState | None], dict[str, str]]
    body: Callable[[Call, ChatResponse], dict[str, Any]]
    events: Callable[
        [Call, ChatChunk, AsyncGenerator[ChatChunk, None]], AsyncGenerator[bytes, None]
    ]


async def _openai_events(
    call: Call, first: ChatChunk, rest: AsyncGenerator[ChatChunk, None]
) -> AsyncGenerator[bytes, None]:
    yield encode_event(openai_wire.chunk_body(first))
    include_usage = openai_wire.include_usage(call.req)
    try:
        async for chunk in rest:
            if chunk.usage is not None and not include_usage:
                continue
            yield encode_event(openai_wire.chunk_body(chunk))
    except ProviderError as err:
        raise ConnectionDropped(err.message) from err
    yield DONE


async def _anthropic_events(
    call: Call, first: ChatChunk, rest: AsyncGenerator[ChatChunk, None]
) -> AsyncGenerator[bytes, None]:
    # `first` is the role-only chunk; message_start plays its part in Anthropic's format.
    async with aclosing(anthropic_wire.events(call, rest)) as events:
        async for event in events:
            yield event


OPENAI = _Wire(
    error=openai_wire.error,
    rate_limit_headers=openai_wire.rate_limit_headers,
    body=lambda _call, response: openai_wire.response_body(response),
    events=_openai_events,
)
ANTHROPIC = _Wire(
    error=anthropic_wire.error,
    rate_limit_headers=anthropic_wire.rate_limit_headers,
    body=anthropic_wire.response_body,
    events=_anthropic_events,
)


def create_sim_app(simulator: Simulator, *, control: bool = False) -> FastAPI:
    # A separate app, not mounted on Relay: the simulator is the *upstream*. It shares none
    # of Relay's middleware, auth or error handling, as a real provider would not.
    app = FastAPI(title="Relay provider simulator", openapi_url=None)
    app.state.simulator = simulator

    @app.post("/v1/chat/completions")
    async def openai_chat(request: Request) -> Response:
        auth = request.headers.get("authorization", "")
        api_key = auth.removeprefix("Bearer ").strip() or "default"
        try:
            req = ChatRequest.model_validate(await request.json())
        except ValueError as exc:  # bad JSON, or a pydantic ValidationError (a ValueError)
            return _error_response(OPENAI, _bad_request(exc), None)
        return await _serve(simulator, OPENAI, request, req, api_key)

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request) -> Response:
        api_key = request.headers.get("x-api-key") or "default"
        try:
            req = anthropic_wire.to_chat_request(await request.json())
        except ValueError as exc:
            return _error_response(ANTHROPIC, _bad_request(exc), None)
        return await _serve(simulator, ANTHROPIC, request, req, api_key)

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        names = sorted(simulator.scenario.models) if simulator.scenario else []
        data = [{"id": n, "object": "model", "created": 0, "owned_by": "simulator"} for n in names]
        return {"object": "list", "data": data}

    @app.get("/sim/truth")
    async def truth(model: str, index: int, cohort: str = "default") -> dict[str, Any]:
        try:
            correct = simulator.truth(model, cohort, index)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        assert simulator.scenario is not None  # truth() succeeded, so there is a scenario
        q = simulator.scenario.q(model, cohort, index)
        return {"model": model, "cohort": cohort, "index": index, "correct": correct, "q": q}

    @app.get("/sim/stats")
    async def stats(bucket_ms: int = 100) -> dict[str, Any]:
        # Upstream load as the provider saw it: totals, per model, and a histogram of
        # arrivals per bucket (default 100 ms) for burst analysis.
        histogram: dict[int, int] = {}
        for t in simulator.arrivals:
            b = int(t * 1000 // bucket_ms)
            histogram[b] = histogram.get(b, 0) + 1
        return {
            "calls": simulator.calls,
            "by_model": dict(simulator.calls_by_model),
            "bucket_ms": bucket_ms,
            "arrivals": histogram,
        }

    if control:
        # Operator endpoints. Only mounted with --control: a simulator that anyone can
        # break is fine on localhost for chaos runs, never on a shared network.
        @app.post("/sim/control")
        async def inject(body: dict[str, Any]) -> dict[str, Any]:
            try:
                incident = simulator.inject(
                    duration_s=float(body["duration_s"]),
                    models=tuple(body.get("models", ())),
                    errors=ErrorMix(**body["errors"]) if body.get("errors") else None,
                    latency_multiplier=float(body.get("latency_multiplier", 1.0)),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {"start_s": incident.start_s, "end_s": incident.end_s}

        @app.post("/sim/clear")
        async def clear() -> dict[str, str]:
            simulator.clear_incidents()
            return {"status": "cleared"}

        @app.post("/sim/reset")
        async def reset() -> dict[str, str]:
            simulator.reset_stats()
            return {"status": "reset"}

    return app


async def _serve(
    simulator: Simulator, wire: _Wire, request: Request, req: ChatRequest, api_key: str
) -> Response:
    raw_index = request.headers.get("x-sim-request-index")
    if raw_index is not None and not raw_index.isdigit():
        return _error_response(wire, _bad_request("x-sim-request-index must be an integer"), None)
    index = int(raw_index) if raw_index is not None else None
    cohort = request.headers.get("x-sim-cohort", "default")

    # 1. Gates: unknown model, capacity. These fail instantly, before any latency.
    try:
        call = simulator.open(req, api_key=api_key, index=index, cohort=cohort)
    except ProviderError as err:
        limits = err.limits if isinstance(err, CapacityExceeded) else None
        return _error_response(wire, err, limits)
    headers = {**wire.rate_limit_headers(call.limits), "x-sim-request-index": str(call.plan.index)}

    # 2a. Non-streaming.
    if not req.stream:
        try:
            response = await call.complete()
        except ProviderError as err:
            if err.error_class is ErrorClass.STREAM_BROKEN:
                # The body is cut off partway: the client gets invalid, truncated JSON.
                return StreamingResponse(
                    _truncated_body(), media_type="application/json", headers=headers
                )
            return _error_response(wire, err, call.limits, headers)
        return JSONResponse(wire.body(call, response), headers=headers)

    # 2b. Streaming: pull the first chunk before committing to a 200.
    chunks = call.stream()
    try:
        first = await anext(chunks)
    except ProviderError as err:
        return _error_response(wire, err, call.limits, headers)
    return StreamingResponse(
        _closing(wire.events(call, first, chunks), chunks),
        media_type="text/event-stream",
        headers={**headers, "cache-control": "no-cache"},
    )


async def _closing(
    events: AsyncGenerator[bytes, None], chunks: AsyncGenerator[ChatChunk, None]
) -> AsyncGenerator[bytes, None]:
    # Close both generators when the response ends, whatever the reason (finished, dropped,
    # client gone). Closing `events` alone would leave `chunks` suspended until garbage
    # collection: the same trap MockProvider.stream() guards against with aclosing().
    async with aclosing(chunks), aclosing(events):
        async for event in events:
            yield event


async def _truncated_body() -> AsyncIterator[bytes]:
    yield b'{"id": "chatcmpl-'
    raise ConnectionDropped("simulated connection reset")


def _bad_request(exc: Exception | str) -> ProviderError:
    message = str(exc) if isinstance(exc, str) else f"Invalid request body: {exc}"
    return ProviderError(ErrorClass.BAD_REQUEST, message)


def _error_response(
    wire: _Wire,
    err: ProviderError,
    limits: LimitState | None,
    extra_headers: dict[str, str] | None = None,
) -> Response:
    status, body, headers = wire.error(err, limits)
    return JSONResponse(body, status_code=status, headers={**(extra_headers or {}), **headers})
