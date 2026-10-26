"""Cache keys and eligibility: the rules that decide when an old answer may be reused."""

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from relay.cache import keys
from relay.cache.eligibility import check
from relay.config.tenants import TenantCache
from relay.schemas.chat import ChatRequest


def req(content: str = "hello", **extra: Any) -> ChatRequest:
    return ChatRequest.model_validate(
        {
            "model": "relay/auto",
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            **extra,
        }
    )


# ---------------------------------------------------------------- keys
def test_trailing_whitespace_and_unicode_form_do_not_change_the_key() -> None:
    assert keys.exact_key("t", req("café")) == keys.exact_key("t", req("café  \n"))
    assert keys.exact_key("t", req("a b")) != keys.exact_key("t", req("a  b"))  # inner kept


@pytest.mark.parametrize(
    "change",
    [
        {"temperature": 0.5},
        {"top_p": 0.9},
        {"max_tokens": 10},
        {"stop": ["x"]},
        {"seed": 1},
    ],
)
def test_every_output_affecting_parameter_changes_both_keys(change: dict[str, Any]) -> None:
    base = req()
    other = req(**change)
    assert keys.exact_key("t", base) != keys.exact_key("t", other)
    if "max_tokens" not in change:  # max_tokens is bucketed in the semantic namespace
        assert keys.semantic_namespace("t", base) != keys.semantic_namespace("t", other)


def test_tenant_alias_and_system_prompt_separate_namespaces() -> None:
    base = req()
    assert keys.exact_key("a", base) != keys.exact_key("b", base)
    assert keys.semantic_namespace("a", base) != keys.semantic_namespace("b", base)
    other_model = ChatRequest.model_validate({**base.model_dump(), "model": "relay/cheap"})
    assert keys.semantic_namespace("t", base) != keys.semantic_namespace("t", other_model)
    with_system = ChatRequest.model_validate(
        {
            "model": "relay/auto",
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hello"},
            ],
        }
    )
    assert keys.semantic_namespace("t", base) != keys.semantic_namespace("t", with_system)


def test_max_tokens_buckets_and_the_length_guard() -> None:
    assert [keys.max_tokens_bucket(n) for n in (1, 32, 33, 64, 65, 1000, None)] == [
        "32", "32", "64", "64", "128", "1024", "default",
    ]  # fmt: skip
    # A reply that stopped naturally at 900 tokens may not answer a request capped at 50...
    assert not keys.fits(900, 50)
    # ...but a 40-token reply fits under any cap that allows 40.
    assert keys.fits(40, 50) and keys.fits(40, 1000)


def test_only_a_single_user_message_is_semantic() -> None:
    assert keys.user_text(req("hi")) == "hi"
    multi = ChatRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
            ],
        }
    )
    assert keys.user_text(multi) is None


def test_lexical_guard_signature_sees_numbers_and_names_not_sentence_starts() -> None:
    sig = keys.guard_signature
    assert sig("What is 15% of 200?") != sig("What is 15% of 300?")
    assert sig("Is Paris the capital of France?") != sig("Is Berlin the capital of France?")
    assert sig("What is the capital?") == sig("Which is the capital?")  # no entities either way


@given(st.text(min_size=1), st.text(min_size=1))
def test_exact_key_separates_different_messages(a: str, b: str) -> None:
    same = keys.normalize(a) == keys.normalize(b)
    assert (keys.exact_key("t", req(a)) == keys.exact_key("t", req(b))) == same


@given(st.text(min_size=1))
def test_normalization_is_idempotent(text: str) -> None:
    once = keys.normalize(text)
    assert keys.normalize(once) == once


# ---------------------------------------------------------------- eligibility
ON = TenantCache(enabled=True, semantic=True)


@pytest.mark.parametrize(
    ("request_kwargs", "policy", "no_store", "exact", "semantic", "reason"),
    [
        ({}, ON, False, True, True, "ok"),
        ({}, TenantCache(), False, False, False, "disabled"),  # off unless the tenant opts in
        ({}, ON, True, False, False, "no_store"),
        ({"temperature": 0.7}, ON, False, False, False, "temperature"),
        ({"temperature": None}, ON, False, False, False, "temperature"),  # unset != 0
        # n > 1 never reaches the cache: the schema rejects it (FR-1); check() still guards it.
        ({}, TenantCache(enabled=True, max_prompt_tokens=1), False, False, False, "too_large"),
    ],
)
def test_eligibility_table(
    request_kwargs: dict[str, Any],
    policy: TenantCache,
    no_store: bool,
    exact: bool,
    semantic: bool,
    reason: str,
) -> None:
    payload = {"messages": [{"role": "user", "content": "a prompt of some length"}]}
    base = {"model": "relay/auto", "temperature": 0, **payload, **request_kwargs}
    base = {k: v for k, v in base.items() if v is not None}
    e = check(ChatRequest.model_validate(base), policy, no_store=no_store)
    assert (e.exact, e.semantic, e.reason) == (exact, semantic, reason)


def test_unset_temperature_is_eligible_only_by_tenant_opt_in() -> None:
    unset = ChatRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    )
    assert not check(unset, TenantCache(enabled=True), no_store=False).exact
    assert check(
        unset, TenantCache(enabled=True, allow_unset_temperature=True), no_store=False
    ).exact
