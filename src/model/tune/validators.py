"""Pydantic validation for clean tuning configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from src.model.experiment.io import read_yaml


KNOWN_WEIGHT = 0.5
OOD_WEIGHT = 0.5
PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_TUNE_CONFIG_PATH = PROJECT_DIR / "config" / "model" / "tune.yaml"


class TuneConfigModel(BaseModel):
    """Validated configuration for threshold selection and objective."""

    model_config = ConfigDict(frozen=True)

    threshold_step: float = 0.01
    minimum_known_balanced_accuracy: float = 0.80
    known_weight: float = KNOWN_WEIGHT
    ood_weight: float = OOD_WEIGHT


TuneConfig = TuneConfigModel


def default_threshold_grid(step: float = 0.01) -> tuple[float, ...]:
    """Return probability thresholds from 0.00 through 1.00."""

    if step <= 0 or step > 1:
        raise ValueError("threshold_step must be in the interval (0, 1].")
    count = int(round(1 / step))
    return tuple(round(index * step, 4) for index in range(count + 1))


def load_tune_config(
    config_path: Path | None = None, overrides: dict[str, Any] | None = None
) -> TuneConfigModel:
    """Load tune config from YAML and apply explicit overrides."""

    raw = read_yaml(config_path or DEFAULT_TUNE_CONFIG_PATH)
    raw.update(clean_overrides(overrides))
    return TuneConfigModel.model_validate(raw)


def clean_overrides(overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Drop unset override values from parsed CLI arguments."""

    if not overrides:
        return {}
    return {key: value for key, value in overrides.items() if value is not None}
