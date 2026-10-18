"""Turns every failure into the OpenAI error envelope:

    {"error": {"message": ..., "type": ..., "param": ..., "code": ...}}

OpenAI SDKs parse this shape into typed exceptions (``RateLimitError``, ``NotFoundError``, ...).
A gateway that is "OpenAI-compatible" only on success is not compatible.

Every error response sets ``x-should-retry: false``. The Stainless-generated OpenAI SDKs obey
this header before their own rules, which would otherwise retry 408/409/429/5xx twice. Relay
is the single retry owner (FR-1): when it returns an error it has already retried, or retrying
cannot help. If the SDK retried too, one request would multiply into 3x Relay's attempts
upstream, which is the retry-storm pattern. Phase 6 revisits this for Relay's own admission
429s, where a client retry after ``Retry-After`` is the right behavior.
"""

from collections.abc import Sequence
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from relay.api.request_id import HEADER as REQUEST_ID_HEADER
from relay.schemas.errors import ErrorClass, RelayError


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RelayError)
    async def _relay_error(request: Request, exc: RelayError) -> JSONResponse:
        return _error_response(
            request,
            exc.status,
            exc.error_class,
            exc.message,
            exc.param,
            retry_after=exc.retry_after_header(),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default is 422 + {"detail": [...]}. OpenAI returns 400 for every invalid
        # request, and clients branch on that. Only the first error is reported, like OpenAI.
        first: dict[str, Any] = dict(exc.errors()[0])
        error_class = (
            ErrorClass.UNSUPPORTED_PARAMETER
            if first["type"] == ErrorClass.UNSUPPORTED_PARAMETER
            else ErrorClass.INVALID_REQUEST
        )
        ctx = first.get("ctx") or {}
        if "param" in ctx:
            param = ctx["param"]
        elif first["type"] == "json_invalid":
            param = None  # the loc is a character offset, not a parameter
        else:
            param = _param_from_loc(first["loc"])
        return _error_response(request, 400, error_class, first["msg"], param)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        # Unknown routes (404) and wrong methods (405) come from the router, not our code.
        return _error_response(request, exc.status_code, ErrorClass.INVALID_REQUEST, exc.detail)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Never put exception text in the body: it can contain internals (paths, keys, SQL).
        # The request ID lets an operator find the stack trace in the server log.
        return _error_response(
            request, 500, ErrorClass.INTERNAL_ERROR, "internal error; quote the request ID"
        )


def _error_response(
    request: Request,
    status: int,
    error_class: ErrorClass,
    message: str,
    param: str | None = None,
    retry_after: str | None = None,
) -> JSONResponse:
    headers = {"x-should-retry": "false"}
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    # Starlette runs the Exception/500 handler in ServerErrorMiddleware, which sits *outside*
    # all user middleware, so RequestIdMiddleware never sees that response. Setting the
    # header here covers it; for every other error the middleware sets the same value again.
    request_id = request.scope.get("state", {}).get("request_id")
    if request_id:
        headers[REQUEST_ID_HEADER] = request_id
    return JSONResponse(
        status_code=status, content=error_body(error_class, message, param), headers=headers
    )


def error_body(
    error_class: ErrorClass, message: str, param: str | None = None
) -> dict[str, dict[str, str | None]]:
    """The OpenAI error envelope. It is shared by JSON errors and by the SSE error payload
    sent mid-stream, so both look the same to the SDK."""
    return {
        "error": {
            "message": message,
            "type": error_class.value,
            "param": param,
            "code": error_class.value,
        }
    }


def _param_from_loc(loc: Sequence[int | str]) -> str | None:
    """("body", "messages", 0, "role") -> "messages[0].role", the way OpenAI names params."""
    param = ""
    for part in loc[1:]:  # loc[0] is where it came from: "body", "header", "query"
        param += f"[{part}]" if isinstance(part, int) else (f".{part}" if param else part)
    return param or None
