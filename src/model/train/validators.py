"""Load the small configuration needed for one model-training run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class TrainingConfigModel(BaseModel):
    """Validated configuration for class-only splitting and candidate CV."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    dataset_dir: Path
    other_label: str
    exclusions_config_path: Path
    text_config_path: Path
    candidates_config_path: Path
    runs_dir: Path
    test_frac: float
    random_state: int
    cv_folds: int
    candidate_limit: int | None
    promote_selected: bool
    run_name: str | None

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
        overrides: dict[str, Any] | None = None,
    ) -> "TrainingConfigModel":
        """Build validated training config from one YAML file and CLI overrides."""

        config_path = Path(config_path).resolve()
        raw = cls._read_yaml(config_path)
        clean = cls._clean_overrides(overrides)
        payload = {
            "dataset_dir": cls._resolve_path(
                config_path,
                clean.get("dataset_dir", raw["dataset_dir"]),
            ),
            "other_label": raw["other_label"],
            "exclusions_config_path": cls._resolve_path(
                config_path,
                raw["exclusions_config_path"],
            ),
            "text_config_path": cls._resolve_path(
                config_path,
                raw["text_config_path"],
            ),
            "candidates_config_path": cls._resolve_path(
                config_path,
                raw["candidates_config_path"],
            ),
            "runs_dir": cls._resolve_path(
                config_path,
                clean.get("runs_dir", raw["runs_dir"]),
            ),
            "test_frac": clean.get("test_frac", raw["test_frac"]),
            "random_state": clean.get("random_state", raw["random_state"]),
            "cv_folds": clean.get("cv_folds", raw["cv_folds"]),
            "candidate_limit": clean.get("candidate_limit", raw["candidate_limit"]),
            "promote_selected": clean.get(
                "promote_selected", raw["promote_selected"]
            ),
            "run_name": clean.get("run_name", raw["run_name"]),
        }
        return cls.model_validate(payload)

    @staticmethod
    def _clean_overrides(overrides: dict[str, Any] | None) -> dict[str, Any]:
        """Drop unset values from parsed command-line arguments."""

        if not overrides:
            return {}
        return {key: value for key, value in overrides.items() if value is not None}

    @staticmethod
    def _resolve_path(config_path: Path, configured: str | Path) -> Path:
        """Resolve a relative YAML path from the directory holding that YAML file."""

        path = Path(configured)
        if path.is_absolute():
            return path
        return (config_path.parent / path).resolve()

    @staticmethod
    def _read_yaml(path: Path) -> dict[str, Any]:
        """Read one YAML mapping without the removed experiment I/O helper."""

        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
