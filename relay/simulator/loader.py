"""Load a ``Scenario`` from YAML, so chaos scenarios are data you can read and diff.

    models:
      sim-primary:
        latency: {ttft_median_s: 0.3, ttft_p95_s: 0.6, tokens_per_s: 200}
        length: {median_tokens: 12, p95_tokens: 30}
        errors: {rate_limited: 0.2, retry_after_s: 2}
        quality: {base: 0.95, shifts: [[400, 0.80]]}      # a level is q, or {reference, w}
    capacity: {rpm: 600, tpm: 400000, scope: model}
    incidents:
      - {start_s: 10, end_s: 40, errors: {server_error: 1.0}}

Every dataclass validates itself, so a bad probability or an infeasible win rate fails here,
before any traffic. Unknown keys are errors for the same reason a typo in relay.yaml is: a
silently ignored ``rate_limitd: 0.8`` would make a chaos run measure nothing.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from relay.simulator.quality import Level, QualitySpec, WinRate
from relay.simulator.scenario import (
    Capacity,
    ErrorMix,
    Incident,
    LatencyModel,
    ModelProfile,
    OutputLength,
    Scenario,
)


def load_scenario(path: str | Path) -> Scenario:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return scenario_from_dict(data)


def scenario_from_dict(data: Mapping[str, Any]) -> Scenario:
    _only(data, {"models", "capacity", "incidents"}, "scenario")
    return Scenario(
        models={name: _profile(p or {}) for name, p in data["models"].items()},
        capacity=Capacity(**data["capacity"]) if data.get("capacity") else None,
        incidents=tuple(_incident(i) for i in data.get("incidents", [])),
    )


def _profile(p: Mapping[str, Any]) -> ModelProfile:
    _only(p, {"latency", "length", "errors", "quality", "cohort_quality"}, "model")
    return ModelProfile(
        latency=LatencyModel(**p.get("latency", {})),
        length=OutputLength(**p.get("length", {})),
        errors=ErrorMix(**p.get("errors", {})),
        quality=_quality(p.get("quality", {})),
        cohort_quality={c: _quality(q) for c, q in p.get("cohort_quality", {}).items()},
    )


def _quality(q: Mapping[str, Any]) -> QualitySpec:
    _only(q, {"base", "shifts"}, "quality")
    return QualitySpec(
        base=_level(q.get("base", 1.0)),
        shifts=tuple((int(at), _level(level)) for at, level in q.get("shifts", [])),
    )


def _level(level: Any) -> Level:
    if isinstance(level, Mapping):
        _only(level, {"reference", "w"}, "win rate")
        return WinRate(reference=level["reference"], w=float(level["w"]))
    return float(level)


def _incident(i: Mapping[str, Any]) -> Incident:
    _only(i, {"start_s", "end_s", "models", "errors", "latency_multiplier"}, "incident")
    return Incident(
        start_s=float(i["start_s"]),
        end_s=float(i["end_s"]),
        models=tuple(i.get("models", ())),
        errors=ErrorMix(**i["errors"]) if i.get("errors") else None,
        latency_multiplier=float(i.get("latency_multiplier", 1.0)),
    )


def _only(data: Mapping[str, Any], allowed: set[str], what: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown key(s) in {what}: {sorted(unknown)}")
