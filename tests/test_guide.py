"""The guide documents the configuration, so it must keep up with it: every setting and every
YAML field exists in guide/configuration.md, and every link in the guide and README resolves."""

import re
from pathlib import Path

import pytest

from relay.config.models import ModelSpec, ProviderSpec
from relay.config.schema import EvidenceSpec, ReliabilityConfig, RouteSpec
from relay.config.tenants import RequestLimits, Tenant, TenantCache, TenantDeadlines, TenantLimits
from relay.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "guide"
CONFIGURATION = (GUIDE / "configuration.md").read_text(encoding="utf-8")

# Set by the process itself, never by an operator.
INTERNAL = {"key_pepper_generated"}


def test_every_environment_setting_is_documented() -> None:
    missing = [
        f"RELAY_{name.upper()}"
        for name in Settings.model_fields
        if name not in INTERNAL and f"RELAY_{name.upper()}" not in CONFIGURATION
    ]
    assert missing == [], f"guide/configuration.md does not mention {missing}"


@pytest.mark.parametrize(
    "model",
    [
        ProviderSpec,
        ModelSpec,
        RouteSpec,
        EvidenceSpec,
        Tenant,
        TenantLimits,
        RequestLimits,
        TenantDeadlines,
        TenantCache,
        ReliabilityConfig,
    ],
)
def test_every_yaml_field_is_documented(model: type) -> None:
    fields = [n for n in model.model_fields if n != "id"]  # type: ignore[attr-defined]
    missing = [n for n in fields if n not in CONFIGURATION]
    assert missing == [], f"{model.__name__}: guide/configuration.md does not mention {missing}"


def test_the_top_level_keys_are_documented() -> None:
    for key in ("price_table_version", "global_budget_usd_daily", "providers", "models"):
        assert key in CONFIGURATION


def _links(markdown: Path) -> list[str]:
    text = markdown.read_text(encoding="utf-8")
    return [m for m in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", text) if not m.startswith("http")]


@pytest.mark.parametrize("page", [ROOT / "README.md", *sorted(GUIDE.glob("*.md"))], ids=str)
def test_relative_links_point_at_files_that_exist(page: Path) -> None:
    for link in _links(page):
        assert (page.parent / link).exists(), f"{page.name} links to {link}, which is missing"


def test_the_guide_does_not_point_readers_at_files_the_repository_does_not_publish() -> None:
    # docs/ is not published, so a link into it breaks on a clone.
    for page in [ROOT / "README.md", *GUIDE.glob("*.md")]:
        assert "docs/" not in page.read_text(encoding="utf-8"), f"{page.name} links into docs/"
