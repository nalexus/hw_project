"""Pydantic validation for clean training configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.model.experiment.io import read_yaml
from src.model.tune.validators import (
    DEFAULT_TUNE_CONFIG_PATH,
    TuneConfigModel,
    default_threshold_grid,
    load_tune_config,
)


MODEL_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = MODEL_DIR.parents[1]
DEFAULT_DATASET_DIR = PROJECT_DIR / "data" / "dataset"
DEFAULT_RUNS_DIR = PROJECT_DIR / "best_pipeline_search_runs"
DEFAULT_TRAIN_CONFIG_PATH = PROJECT_DIR / "config" / "model" / "train.yaml"


class TrainingConfigModel(BaseModel):
    """Validated configuration for one reproducible training run."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    dataset_dir: Path = DEFAULT_DATASET_DIR
    runs_dir: Path = DEFAULT_RUNS_DIR
    validation_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42
    cv_folds: int = 3
    calibration_cv_folds: int = 3
    top_candidates: int = 3
    candidate_limit: int | None = None
    near_duplicate_jaccard: float = 0.92
    fail_on_near_duplicates: bool = False
    include_synthetic: bool = True
    promote_selected: bool = False
    threshold_policy: Literal["length_bucket", "global"] = "length_bucket"
    run_name: str | None = None
    tune: TuneConfigModel = Field(default_factory=TuneConfigModel)

    @property
    def threshold_step(self) -> float:
        """Return threshold step for current tuning policy."""

        return self.tune.threshold_step

    @property
    def threshold_grid(self) -> tuple[float, ...]:
        """Return threshold grid for compatibility with tuning helpers."""

        return default_threshold_grid(self.tune.threshold_step)

    @property
    def minimum_known_balanced_accuracy(self) -> float:
        """Return selected minimum known-class balanced accuracy."""

        return self.tune.minimum_known_balanced_accuracy


TrainingConfig = TrainingConfigModel


def load_training_config(
    train_config_path: Path | None = None,
    tune_config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> TrainingConfigModel:
    """Load train+tune YAML config and apply explicit overrides."""

    raw = read_yaml(train_config_path or DEFAULT_TRAIN_CONFIG_PATH)
    clean = clean_overrides(overrides)
    tune = load_tune_config(tune_config_path or DEFAULT_TUNE_CONFIG_PATH, clean_tune(clean))
    payload = {
        "dataset_dir": resolve_model_path(
            clean.get("dataset_dir", raw.get("dataset_dir", str(DEFAULT_DATASET_DIR)))
        ),
        "runs_dir": resolve_model_path(
            clean.get("runs_dir", raw.get("runs_dir", str(DEFAULT_RUNS_DIR)))
        ),
        "validation_size": clean.get("validation_size", raw.get("validation_size", 0.15)),
        "test_size": clean.get("test_size", raw.get("test_size", 0.15)),
        "random_state": clean.get("random_state", raw.get("random_state", 42)),
        "cv_folds": clean.get("cv_folds", raw.get("cv_folds", 3)),
        "calibration_cv_folds": clean.get(
            "calibration_cv_folds", raw.get("calibration_cv_folds", 3)
        ),
        "top_candidates": clean.get("top_candidates", raw.get("top_candidates", 3)),
        "candidate_limit": clean.get("candidate_limit", raw.get("candidate_limit")),
        "near_duplicate_jaccard": clean.get(
            "near_duplicate_jaccard", raw.get("near_duplicate_jaccard", 0.92)
        ),
        "fail_on_near_duplicates": clean.get(
            "fail_on_near_duplicates", raw.get("fail_on_near_duplicates", False)
        ),
        "include_synthetic": clean.get(
            "include_synthetic", raw.get("include_synthetic", True)
        ),
        "promote_selected": clean.get(
            "promote_selected", raw.get("promote_selected", False)
        ),
        "threshold_policy": clean.get(
            "threshold_policy", raw.get("threshold_policy", "length_bucket")
        ),
        "run_name": clean.get("run_name", raw.get("run_name")),
        "tune": tune,
    }
    return TrainingConfigModel.model_validate(payload)


def clean_tune(overrides: dict[str, Any]) -> dict[str, Any]:
    """Extract tuning-related override values from training CLI args."""

    keys = {
        "threshold_step",
        "minimum_known_balanced_accuracy",
        "known_weight",
        "ood_weight",
    }
    return {key: value for key, value in overrides.items() if key in keys}


def clean_overrides(overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Drop unset override values from parsed CLI arguments."""

    if not overrides:
        return {}
    return {key: value for key, value in overrides.items() if value is not None}


def resolve_model_path(configured: str | Path) -> Path:
    """Resolve model-local relative paths for YAML-driven config."""

    path = Path(configured)
    if path.is_absolute():
        return path
    return PROJECT_DIR / path
