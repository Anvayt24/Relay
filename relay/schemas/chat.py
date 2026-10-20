"""The internal chat schema: the OpenAI wire format, the subset Relay supports (PRD FR-1).

All normalization happens here, once, at the boundary. Everything downstream (router, cache,
adapters) sees one canonical shape:
- ``max_completion_tokens`` is folded into ``max_tokens``;
- ``stop`` is always a list;
- message ``content`` is always a string;
- the ``developer`` role is ``system``.

Errors use ``PydanticCustomError`` with Relay's error class as the error *type*. The model then
stays usable outside FastAPI (it raises a normal ``ValidationError``), and the API layer reads
the class back from ``error["type"]`` when it builds the response.
"""

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from relay.schemas.errors import ErrorClass


def _unsupported(message: str, param: str | None = None) -> PydanticCustomError:
    ctx = {"param": param} if param else None
    return PydanticCustomError(ErrorClass.UNSUPPORTED_PARAMETER.value, message, ctx)


def _invalid(message: str, param: str) -> PydanticCustomError:
    # Model-level validators have no field location, so the offending param travels in ctx.
    return PydanticCustomError(ErrorClass.INVALID_REQUEST.value, message, {"param": param})


class Message(BaseModel):
    # Message-level extras (e.g. "name") are harmless to drop; the dangerous ones (tool calls)
    # can only matter with tools, which are rejected at the request level.
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant"]
    content: str

    @field_validator("role", mode="before")
    @classmethod
    def _normalize_role(cls, v: Any) -> Any:
        if v == "developer":
            # Newer OpenAI SDKs send "developer" where older ones sent "system"; to every
            # provider Relay targets, it means the same thing.
            return "system"
        if v in ("tool", "function"):
            raise _unsupported("tool messages are not supported")
        return v

    @field_validator("content", mode="before")
    @classmethod
    def _flatten_content_parts(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return v
        texts: list[str] = []
        for part in v:
            if not isinstance(part, dict) or part.get("type") != "text":
                kind = part.get("type") if isinstance(part, dict) else type(part).__name__
                raise _unsupported(f"content part of type {kind!r} is not supported")
            texts.append(str(part.get("text", "")))
        # Newline-joined: providers treat parts as separate blocks, so keep them visibly apart.
        return "\n".join(texts)


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatRequest(BaseModel):
    # "allow" instead of "ignore": unknown fields are kept so they can be counted. They
    # are never forwarded, because adapters build provider payloads from named fields only.
    model_config = ConfigDict(extra="allow")

    model: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1)
    stop: list[str] | None = Field(default=None, max_length=4)
    seed: int | None = None
    user: str | None = None
    n: int = 1
    stream: bool = False
    stream_options: StreamOptions | None = None

    # Declared only so they can be refused by name. Silently dropping `tools` would return a
    # plain-text answer to a client expecting a tool call: a silent behavior change (FR-1).
    tools: None = None
    tool_choice: None = None
    response_format: None = None

    @model_validator(mode="before")
    @classmethod
    def _fold_max_tokens_aliases(cls, data: Any) -> Any:
        # "before": this runs on the raw dict, so the alias never reaches field validation
        # and never lands in the extras.
        if not isinstance(data, dict) or "max_completion_tokens" not in data:
            return data
        data = dict(data)  # never mutate the caller's dict
        mct = data.pop("max_completion_tokens")
        mt = data.get("max_tokens")
        if mt is not None and mct is not None and mt != mct:
            raise _invalid(
                "max_tokens and max_completion_tokens are aliases but have different values",
                "max_completion_tokens",
            )
        if mt is None:
            data["max_tokens"] = mct
        return data

    @field_validator("tools", "tool_choice", "response_format", mode="before")
    @classmethod
    def _reject_unsupported(cls, v: Any) -> Any:
        if v is not None:
            raise _unsupported("this parameter is not supported yet")
        return v

    @field_validator("n")
    @classmethod
    def _only_one_choice(cls, v: int) -> int:
        if v != 1:
            raise _unsupported("only n=1 is supported")
        return v

    @field_validator("stop", mode="before")
    @classmethod
    def _stop_as_list(cls, v: Any) -> Any:
        return [v] if isinstance(v, str) else v

    @model_validator(mode="after")
    def _stream_options_need_stream(self) -> Self:
        if self.stream_options is not None and not self.stream:
            raise _invalid("stream_options is only allowed when stream=true", "stream_options")
        return self

    @property
    def ignored_fields(self) -> list[str]:
        return sorted(self.model_extra or {})


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: Literal["stop", "length"]


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


# ---------------------------------------------------------------- streaming
# A stream is a sequence of chunks:
# - one role-only chunk;
# - content chunks;
# - a chunk with finish_reason;
# - a final usage-only chunk (empty choices).
# Relay only forwards that last one when the client set stream_options.include_usage.


class ChunkDelta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: ChunkDelta
    finish_reason: Literal["stop", "length"] | None = None


class ChatChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]
    usage: Usage | None = None

    @property
    def has_content(self) -> bool:
        """True for the chunk FR-5 calls the "first byte": the first one carrying text. The
        role-only chunk before it does not count, because a failure can still be hidden from
        the client at that point."""
        return any(choice.delta.content for choice in self.choices)

    @property
    def is_finished(self) -> bool:
        return any(choice.finish_reason is not None for choice in self.choices)
