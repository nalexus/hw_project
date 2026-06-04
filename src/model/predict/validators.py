"""Pydantic validation for clean prediction configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from src.model.experiment.io import read_yaml
from src.model.predict.length import length_bucket


PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_PREDICT_CONFIG_PATH = PROJECT_DIR / "config" / "model" / "predict.yaml"


class PredictConfigModel(BaseModel):
    """Validated clean-package prediction policy config."""

    model_config = ConfigDict(frozen=True)

    policy: str
    calibration: str | None = None
    threshold: float | None = None
    default_threshold: float | None = None
    bucket_thresholds: dict[str, float] = {}


def normalize_policy(threshold=0.16, threshold_policy=None) -> dict[str, Any]:
    """Return one canonical reject-policy dictionary."""

    if threshold_policy is None:
        return {"name": "global", "params": {"threshold": threshold}}
    if "name" in threshold_policy:
        return threshold_policy
    if "policy" in threshold_policy:
        return policy_from_runtime_config(threshold_policy)
    raise ValueError("threshold_policy must include 'name' or 'policy'")


def policy_from_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    """Convert runtime config fields into predictor policy format."""

    validated = PredictConfigModel.model_validate(config)
    if validated.policy == "global":
        return {"name": "global", "params": {"threshold": float(validated.threshold)}}
    if validated.policy == "length_bucket":
        return {
            "name": "length_bucket",
            "params": {
                "default_threshold": float(validated.default_threshold),
                "bucket_thresholds": {
                    key: float(value) for key, value in validated.bucket_thresholds.items()
                },
            },
        }
    raise ValueError(f"Unsupported runtime policy: {validated.policy}")


def threshold_for_text(policy: dict[str, Any], text: str) -> float:
    """Resolve the rejection threshold for one document."""

    name = policy["name"]
    params = policy["params"]
    if name == "global":
        return float(params["threshold"])
    if name == "length_bucket":
        bucket = length_bucket(text)
        return float(params["bucket_thresholds"].get(bucket, params["default_threshold"]))
    raise ValueError(f"Unsupported threshold policy: {name}")


def load_predict_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load clean-package prediction defaults from YAML."""

    return PredictConfigModel.model_validate(
        read_yaml(config_path or DEFAULT_PREDICT_CONFIG_PATH)
    ).model_dump()
