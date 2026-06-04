"""Filesystem serialization helpers for experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def json_ready(value: Any) -> Any:
    """Convert common Python values into JSON-safe values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    """Write pretty JSON with stable key ordering."""

    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_yaml(path: Path, payload: Any) -> None:
    """Write readable YAML without forced key sorting."""

    path.write_text(yaml.safe_dump(json_ready(payload), sort_keys=False), encoding="utf-8")


def read_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML mapping from disk."""

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
