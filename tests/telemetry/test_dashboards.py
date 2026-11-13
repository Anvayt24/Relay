"""The provisioned Grafana setup: valid files, and dashboards that only query
metrics Relay actually exports. A renamed metric would otherwise blank a panel silently."""

import json
import re
from pathlib import Path

import yaml
from prometheus_client import REGISTRY

from relay.api.app import create_app
from relay.providers.mock import MockProvider

from ..conftest import test_settings as relay_settings  # type: ignore[import-not-found]

PROVISIONING = Path("deploy/grafana/provisioning")
DASHBOARDS = sorted((PROVISIONING / "dashboards").glob("*.json"))
METRIC = re.compile(r"\brelay_[a-z0-9_]+")
SUFFIXES = {
    "counter": ("_total", "_created"),
    "histogram": ("_bucket", "_sum", "_count", "_created"),
    "gauge": ("",),
}


def exported_names() -> set[str]:
    """Every series name Relay's metrics can produce (families may have no samples yet)."""
    create_app(relay_settings(), providers={"mock": MockProvider()})  # import every module
    import relay.evidence.board
    import relay.evidence.shadow
    import relay.evidence.worker
    import relay.policy.poller  # noqa: F401

    names = set()
    for family in REGISTRY.collect():
        for suffix in SUFFIXES.get(family.type, ("",)):
            names.add(family.name + suffix)
    return names


def test_the_overview_dashboard_exists_and_parses() -> None:
    assert [d.name for d in DASHBOARDS] == ["relay-overview.json"]
    board = json.loads(DASHBOARDS[0].read_text(encoding="utf-8"))
    titles = " ".join(p["title"] for p in board["panels"]).lower()
    # The dashboards an operator expects.
    for needed in ("cost saved", "cache hit rate", "per-provider", "fallback rate", "breaker"):
        assert needed in titles, needed


def test_every_panel_queries_a_metric_relay_exports() -> None:
    names = exported_names()
    for path in DASHBOARDS:
        board = json.loads(path.read_text(encoding="utf-8"))
        for panel in board["panels"]:
            for target in panel.get("targets", []):
                for metric in METRIC.findall(target["expr"]):
                    assert metric in names, f"{path.name}: {panel['title']!r} uses {metric}"


def test_provisioning_files_are_valid_and_consistent() -> None:
    uids = set()
    for path in sorted((PROVISIONING / "datasources").glob("*.yml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        uids |= {ds["uid"] for ds in doc["datasources"]}
    assert uids == {"prometheus", "tempo"}
    provider = yaml.safe_load((PROVISIONING / "dashboards" / "provider.yml").read_text("utf-8"))
    assert provider["providers"][0]["type"] == "file"
    for path in DASHBOARDS:
        board = json.loads(path.read_text(encoding="utf-8"))
        for panel in board["panels"]:
            if panel["type"] != "row":
                assert panel["datasource"]["uid"] in uids


def test_the_stack_wires_traces_to_tempo() -> None:
    compose = yaml.safe_load(Path("deploy/docker-compose.yml").read_text(encoding="utf-8"))
    assert "tempo" in compose["services"]
    collector = yaml.safe_load(Path("deploy/otel-collector.yaml").read_text(encoding="utf-8"))
    assert "otlp/tempo" in collector["service"]["pipelines"]["traces"]["exporters"]
    api_env = compose["services"]["api"]["environment"]
    # The observability services are an opt-in profile (self-hosting keeps the default stack
    # lean), so the endpoint is read from .env: pointing at a collector that is not running
    # would make the exporter log errors forever. .env.example says what to set.
    assert api_env["RELAY_OTEL_ENDPOINT"] == "${RELAY_OTEL_ENDPOINT:-}"
    assert "RELAY_OTEL_ENDPOINT=http://otel-collector:4317" in Path(".env.example").read_text(
        encoding="utf-8"
    )
    for service in ("otel-collector", "tempo", "prometheus", "grafana"):
        assert compose["services"][service]["profiles"] == ["observability"]
    assert api_env["RELAY_LOG_FORMAT"] == "json"
