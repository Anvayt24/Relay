"""Loads and validates the YAML config (PRD FR-3: "an invalid config fails startup").

The shipped default is ``relay/config/relay.yaml``, inside the package, so it travels with
the code into the image. ``RELAY_CONFIG_PATH`` points at another file.

``yaml.safe_load``, never ``yaml.load``: the full loader can build arbitrary Python objects
from tags in the file, which turns "edit a config file" into "run code".
"""

from pathlib import Path

import yaml

from relay.config.schema import RelayConfig

DEFAULT_CONFIG = Path(__file__).with_name("relay.yaml")


def load_config(path: str | Path | None = None) -> RelayConfig:
    source = Path(path) if path else DEFAULT_CONFIG
    with source.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    try:
        return RelayConfig.model_validate(data)
    except ValueError as exc:
        raise ValueError(f"invalid config {source}: {exc}") from exc
