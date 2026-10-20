from decimal import Decimal

from relay.config.models import ModelSpec
from relay.config.pricing import Pricing, list_cost_nano, usd
from relay.schemas.chat import PromptTokensDetails, Usage


def _model(**kw: object) -> ModelSpec:
    values: dict[str, object] = {
        "provider": "p",
        "provider_model": "m",
        "tier": "cheap",
        "price_in": Decimal("1.00"),
        "price_out": Decimal("4.00"),
        **kw,
    }
    return ModelSpec.model_validate(values)


def _usage(prompt: int, completion: int, cached: int = 0, written: int = 0) -> Usage:
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=cached) if cached else None,
        cache_write_tokens=written,
    )


def test_list_cost_in_nano_usd() -> None:
    # $1/1M in, $4/1M out: 1,000 in + 500 out = $0.001 + $0.002 = $0.003 = 3,000,000 nano.
    assert list_cost_nano(_model(), _usage(1000, 500)) == 3_000_000


def test_cached_and_cache_write_tokens_use_their_own_prices() -> None:
    model = _model(price_cached_in=Decimal("0.10"), price_cache_write=Decimal("1.25"))
    # 1,000 prompt tokens = 200 uncached + 600 cached + 200 written; 0 output.
    nano = list_cost_nano(model, _usage(1000, 0, cached=600, written=200))
    assert nano == (200 * 1000 + 600 * 100 + 200 * 1250)


def test_cached_price_defaults_to_the_input_price() -> None:
    assert list_cost_nano(_model(), _usage(1000, 0, cached=1000)) == 1_000_000


def test_fractions_of_a_nano_round_up_once() -> None:
    # One token at $0.0375/1M is 37.5 nano: charged 38, never 37 (tiny requests aren't free).
    model = _model(price_in=Decimal("0.0375"), price_out=Decimal(0))
    assert list_cost_nano(model, _usage(1, 0)) == 38


def test_billed_is_zero_on_a_free_tier_but_list_is_not() -> None:
    pricing = Pricing(model=_model(billing="free"), strong=None, price_table_version="v")
    cost = pricing.cost(_usage(1000, 500))
    assert (cost.billed_nano, cost.list_nano) == (0, 3_000_000)


def test_counterfactual_prices_the_same_usage_on_the_strong_model() -> None:
    strong = _model(tier="strong", price_in=Decimal("10"), price_out=Decimal("40"))
    pricing = Pricing(model=_model(), strong=strong, price_table_version="v")
    cost = pricing.cost(_usage(1000, 500))
    assert cost.counterfactual_strong_nano == 30_000_000
    assert (
        Pricing(model=_model(), strong=None, price_table_version="v")
        .cost(_usage(1, 1))
        .counterfactual_strong_nano
        is None
    )


def test_usd_text_is_exact() -> None:
    assert usd(1400) == "0.0000014"
    assert usd(3_000_000_000) == "3"
    assert usd(0) == "0"
