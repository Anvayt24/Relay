"""Config validation (PRD FR-3): every rule is one test, and an invalid config never loads."""

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from relay.config.loader import load_config
from relay.config.schema import RelayConfig

TEST_YAML = Path(__file__).with_name("test.yaml")


def _base() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(TEST_YAML.read_text(encoding="utf-8"))
    return copy.deepcopy(data)


def test_shipped_config_loads() -> None:
    config = load_config()  # relay/config/relay.yaml
    assert {"dev", "live", "demo"} <= set(config.tenants)
    assert config.tenants["dev"].id == "dev"  # ids come from the mapping keys


def test_gemini_is_not_allowed_for_any_shipped_tenant() -> None:
    # Gemini's free tier may let Google use prompts (PRD §13): never on by default.
    for tenant in load_config().tenants.values():
        assert "gemini" not in tenant.allowed_providers


def test_the_test_config_loads() -> None:
    assert RelayConfig.model_validate(_base()).price_table_version == "test-1"


def _invalid(data: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        RelayConfig.model_validate(data)


def test_model_with_unknown_provider() -> None:
    data = _base()
    data["models"]["mock/cheap"]["provider"] = "nowhere"
    _invalid(data, "unknown provider 'nowhere'")


def test_model_without_a_price() -> None:
    data = _base()
    del data["models"]["mock/cheap"]["price_in"]
    _invalid(data, "price_in")


def test_dangling_alias() -> None:
    data = _base()
    data["routes"]["default"]["aliases"]["relay/cheap"] = "mock/missing"
    _invalid(data, "unknown model 'mock/missing'")


def test_fallback_graph_with_unknown_model() -> None:
    data = _base()
    data["routes"]["default"]["fallback_graph"] = {"mock/cheap": [{"to": "mock/missing"}]}
    _invalid(data, "fallback graph names unknown model")


def test_fallback_graph_cycle() -> None:
    data = _base()
    data["routes"]["default"]["fallback_graph"] = {
        "mock/cheap": [{"to": "mock/strong"}],
        "mock/strong": [{"to": "mock/cheap"}],
    }
    _invalid(data, "cycle: mock/cheap -> mock/strong -> mock/cheap")


def test_fallback_graph_self_loop() -> None:
    data = _base()
    data["routes"]["default"]["fallback_graph"] = {"mock/cheap": [{"to": "mock/cheap"}]}
    _invalid(data, "cycle")


def test_acyclic_diamond_is_fine() -> None:
    # Two paths to the same node is not a cycle.
    data = _base()
    data["routes"]["default"]["fallback_graph"] = {
        "mock/strong": [{"to": "mock/cheap"}],
        "mock/cheap": [],
    }
    RelayConfig.model_validate(data)


def test_tenant_with_unknown_route() -> None:
    data = _base()
    data["tenants"]["test"]["route"] = "nope"
    _invalid(data, "unknown route 'nope'")


def test_tenant_with_unknown_provider_or_model() -> None:
    data = _base()
    data["tenants"]["test"]["allowed_providers"] = ["mock", "nowhere"]
    _invalid(data, "unknown provider 'nowhere'")
    data = _base()
    data["tenants"]["limited"]["allowed_models"] = ["mock/missing"]
    _invalid(data, "unknown model 'mock/missing'")


def test_tenant_route_alias_the_tenant_cannot_use() -> None:
    # `limited` may only call mock/cheap; pointing it at a route whose relay/strong goes to
    # mock/strong would make that alias 404 forever. Caught at load.
    data = _base()
    data["tenants"]["limited"]["route"] = "default"
    _invalid(data, "may not call")


def test_typos_are_errors_not_ignored() -> None:
    data = _base()
    data["models"]["mock/cheap"]["prise_in"] = 1  # extra="forbid"
    _invalid(data, "prise_in")


def test_negative_price_is_rejected() -> None:
    data = _base()
    data["models"]["mock/cheap"]["price_out"] = -1
    _invalid(data, "price_out")


def test_load_config_names_the_file_on_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    data = _base()
    data["tenants"]["test"]["route"] = "nope"
    bad.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad.yaml"):
        load_config(bad)
