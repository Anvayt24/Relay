from decimal import Decimal

from relay.admission.tokens import output_cap, prompt_tokens, text_tokens
from relay.config.models import ModelSpec
from relay.schemas.chat import ChatRequest


def _req(content: str, **extra: object) -> ChatRequest:
    return ChatRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": content}], **extra}
    )


def test_text_tokens_count_utf8_bytes() -> None:
    assert text_tokens("") == 0
    assert text_tokens("abc") == 1
    assert text_tokens("abcd") == 2  # rounds up
    # CJK: 3 bytes per character in UTF-8, so roughly a token per character, which is about
    # what real tokenizers give. bytes/3 stays in the safe range without a tokenizer.
    assert text_tokens("你好世界") == 4


def test_english_is_overestimated_not_under() -> None:
    # Real tokenizers give ~1 token per 4 characters of English; the estimate must be above.
    text = "The quick brown fox jumps over the lazy dog. " * 20
    assert prompt_tokens(_req(text)) > len(text) / 4


def test_prompt_estimate_adds_overhead_and_margin() -> None:
    # "hello there" = 11 bytes -> 4 tokens, + 4 message overhead = 8, x 1.10 -> 9 (rounded up).
    assert prompt_tokens(_req("hello there")) == 9


def test_output_cap_defaults_to_the_models() -> None:
    model = ModelSpec(
        provider="p",
        provider_model="m",
        tier="cheap",
        price_in=Decimal(0),
        price_out=Decimal(0),
        default_max_tokens=777,
    )
    assert output_cap(_req("x"), model) == 777
    assert output_cap(_req("x", max_tokens=5), model) == 5
