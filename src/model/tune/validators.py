"""Load OOD-policy constraints used after model selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class TuneConfigModel(BaseModel):
    """Validated reliability and coverage constraints for the OOD policy."""

    model_config = ConfigDict(frozen=True)

    support_tail_rate: float
    target_accepted_error: float
    minimum_correct_coverage: float

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
        overrides: dict[str, Any] | None = None,
    ) -> "TuneConfigModel":
        """Build validated OOD-policy config from YAML and CLI overrides."""

        with Path(config_path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        values.update(cls._clean_overrides(overrides))
        return cls.model_validate(values)

    @staticmethod
    def _clean_overrides(overrides: dict[str, Any] | None) -> dict[str, Any]:
        """Drop unset values from parsed command-line arguments."""

        if not overrides:
            return {}
        return {key: value for key, value in overrides.items() if value is not None}
