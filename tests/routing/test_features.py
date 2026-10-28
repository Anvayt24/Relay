import math

from hypothesis import given
from hypothesis import strategies as st

from relay.routing.features import (
    N_SCALARS,
    SCALAR_NAMES,
    count_turns,
    last_user_text,
    request_scalars,
    scalar_features,
)
from relay.schemas.chat import ChatRequest

from .conftest import chat


def test_the_router_reads_the_last_user_message_only() -> None:
    req = ChatRequest.model_validate(
        {
            "model": "relay/auto",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "the latest question  \n"},
            ],
        }
    )
    assert last_user_text(req) == "the latest question"  # normalized like the cache's keys
    assert count_turns(req) == 2


def test_scalar_block_matches_its_declared_layout() -> None:
    values = scalar_features("print('hi')\n```py\nx\n```", turns=3)
    assert len(values) == N_SCALARS == len(SCALAR_NAMES)
    named = dict(zip(SCALAR_NAMES, values, strict=True))
    assert named["turns"] == 3.0
    assert named["has_code_fence"] == 1.0
    assert named["has_tools"] == 0.0
    assert named["script_latin"] == 1.0
    assert math.isclose(named["log_chars"], math.log1p(len("print('hi')\n```py\nx\n```")))


def test_script_one_hot_has_exactly_one_hot_entry() -> None:
    for text in ("hello", "आप कैसे हैं", "你好世界", "Привет", "12345", ""):
        named = dict(zip(SCALAR_NAMES, scalar_features(text), strict=True))
        assert sum(v for k, v in named.items() if k.startswith("script_")) == 1.0


@given(st.text(max_size=300))
def test_features_are_finite_for_any_text(text: str) -> None:
    assert all(math.isfinite(v) for v in scalar_features(text))


def test_request_scalars_use_the_same_text_the_embedding_uses() -> None:
    text, scalars = request_scalars(chat("Hello there  "))
    assert text == "Hello there"
    assert scalars[0] == math.log1p(len("Hello there"))
