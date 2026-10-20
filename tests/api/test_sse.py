import json

from relay.api.sse import DONE, encode_chunk, encode_error, encode_event
from relay.schemas.chat import ChatChunk, ChunkChoice, ChunkDelta
from relay.schemas.errors import ErrorClass


def _chunk(content: str) -> ChatChunk:
    return ChatChunk(
        id="c1",
        created=0,
        model="mock/strong",
        choices=[ChunkChoice(delta=ChunkDelta(content=content))],
    )


def test_event_framing() -> None:
    assert encode_event({"a": 1}) == b'data: {"a":1}\n\n'
    assert DONE == b"data: [DONE]\n\n"


def test_newlines_in_content_stay_on_one_data_line() -> None:
    # A raw newline would end the "data:" line early and corrupt the event.
    raw = encode_chunk(_chunk("line one\nline two"))
    assert raw.count(b"\n") == 2  # only the terminating blank line
    assert raw.endswith(b"\n\n")


def test_chunk_omits_none_fields() -> None:
    payload = json.loads(encode_chunk(_chunk("hi"))[len(b"data: ") :])
    assert "usage" not in payload
    assert payload["object"] == "chat.completion.chunk"
    assert payload["choices"][0]["delta"] == {"content": "hi"}


def test_extra_top_level_fields() -> None:
    payload = json.loads(encode_chunk(_chunk("hi"), extra={"relay": {"x": 1}})[len(b"data: ") :])
    assert payload["relay"] == {"x": 1}


def test_error_payload_is_the_openai_envelope() -> None:
    raw = encode_error(ErrorClass.STREAM_BROKEN, "upstream went idle")
    payload = json.loads(raw[len(b"data: ") :])
    assert payload == {
        "error": {
            "message": "upstream went idle",
            "type": "stream_broken",
            "param": None,
            "code": "stream_broken",
        }
    }
