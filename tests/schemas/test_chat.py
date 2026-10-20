from typing import Any

import pytest
from pydantic import ValidationError

from relay.schemas.chat import ChatRequest


def _req(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"model": "relay/auto", "messages": [{"role": "user", "content": "hi"}]}
    body.update(overrides)
    return body


def _only_error(exc: pytest.ExceptionInfo[ValidationError]) -> dict[str, Any]:
    errors = exc.value.errors()
    assert len(errors) == 1, errors
    return dict(errors[0])


def test_minimal_request_gets_defaults() -> None:
    req = ChatRequest.model_validate(_req())
    assert req.n == 1
    assert req.stream is False
    assert req.max_tokens is None
    assert req.ignored_fields == []


def test_max_completion_tokens_is_folded_into_max_tokens() -> None:
    req = ChatRequest.model_validate(_req(max_completion_tokens=50))
    assert req.max_tokens == 50
    assert "max_completion_tokens" not in req.ignored_fields


def test_max_tokens_aliases_may_agree() -> None:
    req = ChatRequest.model_validate(_req(max_tokens=50, max_completion_tokens=50))
    assert req.max_tokens == 50


def test_conflicting_max_tokens_aliases_are_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        ChatRequest.model_validate(_req(max_tokens=10, max_completion_tokens=50))
    err = _only_error(exc)
    assert err["type"] == "invalid_request"
    assert err["ctx"]["param"] == "max_completion_tokens"


def test_n_greater_than_one_is_unsupported() -> None:
    with pytest.raises(ValidationError) as exc:
        ChatRequest.model_validate(_req(n=2))
    err = _only_error(exc)
    assert err["type"] == "unsupported_parameter"
    assert err["loc"] == ("n",)


def test_stop_string_is_normalized_to_list() -> None:
    assert ChatRequest.model_validate(_req(stop="END")).stop == ["END"]
    assert ChatRequest.model_validate(_req(stop=["a", "b"])).stop == ["a", "b"]


def test_more_than_four_stop_sequences_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(_req(stop=["a", "b", "c", "d", "e"]))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tools", [{"type": "function", "function": {"name": "f"}}]),
        ("tool_choice", "auto"),
        ("response_format", {"type": "json_object"}),
    ],
)
def test_tool_and_format_params_are_rejected_not_dropped(field: str, value: Any) -> None:
    with pytest.raises(ValidationError) as exc:
        ChatRequest.model_validate(_req(**{field: value}))
    err = _only_error(exc)
    assert err["type"] == "unsupported_parameter"
    assert err["loc"] == (field,)


def test_explicit_null_tools_is_the_same_as_absent() -> None:
    req = ChatRequest.model_validate(_req(tools=None, tool_choice=None, response_format=None))
    assert req.tools is None


def test_tool_role_is_unsupported() -> None:
    body = _req(messages=[{"role": "tool", "content": "x", "tool_call_id": "1"}])
    with pytest.raises(ValidationError) as exc:
        ChatRequest.model_validate(body)
    err = _only_error(exc)
    assert err["type"] == "unsupported_parameter"
    assert err["loc"] == ("messages", 0, "role")


def test_developer_role_is_normalized_to_system() -> None:
    body = _req(
        messages=[{"role": "developer", "content": "be brief"}, {"role": "user", "content": "hi"}]
    )
    req = ChatRequest.model_validate(body)
    assert req.messages[0].role == "system"


def test_text_content_parts_are_joined_into_a_string() -> None:
    parts = [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]
    req = ChatRequest.model_validate(_req(messages=[{"role": "user", "content": parts}]))
    assert req.messages[0].content == "first\nsecond"


def test_image_content_part_is_unsupported() -> None:
    parts = [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
    ]
    with pytest.raises(ValidationError) as exc:
        ChatRequest.model_validate(_req(messages=[{"role": "user", "content": parts}]))
    err = _only_error(exc)
    assert err["type"] == "unsupported_parameter"
    assert err["loc"] == ("messages", 0, "content")


def test_empty_messages_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(_req(messages=[]))


def test_stream_options_require_stream() -> None:
    with pytest.raises(ValidationError) as exc:
        ChatRequest.model_validate(_req(stream_options={"include_usage": True}))
    err = _only_error(exc)
    assert err["type"] == "invalid_request"
    assert err["ctx"]["param"] == "stream_options"


def test_stream_options_with_stream_ok() -> None:
    req = ChatRequest.model_validate(_req(stream=True, stream_options={"include_usage": True}))
    assert req.stream_options is not None
    assert req.stream_options.include_usage is True


def test_unknown_fields_are_kept_and_listed() -> None:
    req = ChatRequest.model_validate(_req(logit_bias={"1": 2}, frequency_penalty=0.5))
    assert req.ignored_fields == ["frequency_penalty", "logit_bias"]


def test_temperature_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(_req(temperature=3.0))
