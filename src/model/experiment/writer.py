"""Filesystem writer for experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from src.model.experiment.io import json_ready, write_json, write_yaml
from src.model.experiment.manifest import (
    build_metadata,
    runtime_config,
)
from src.model.train.schemas import ExperimentSplits, FittedCandidate
from src.model.train.validators import TrainingConfig


class ExperimentWriter:
    """Write selected-model artifacts and experiment reports to disk."""

    def __init__(self, run_dir: Path, config: TrainingConfig) -> None:
        """Store output location and run configuration."""

        self.run_dir = run_dir
        self.config = config
        self.run_dir.mkdir(parents=True, exist_ok=False)

    def write_all(
        self,
        selected: FittedCandidate,
        cv_results: list[dict[str, Any]],
        evaluations: dict[str, dict[str, Any]],
        splits: ExperimentSplits,
        leakage_report: dict[str, Any],
        baseline_eval: dict[str, Any] | None = None,
    ) -> Path:
        """Write every artifact for one selected training run."""

        model_path = self.run_dir / model_filename(self.run_dir.name)
        joblib.dump(selected.model, model_path)
        self._write_run_files(model_path, selected, cv_results, evaluations, splits, leakage_report)
        metadata = build_metadata(selected, evaluations, self.config, model_path, baseline_eval)
        write_json(self.run_dir / "metadata.json", metadata)
        return model_path

    def _write_run_files(
        self,
        model_path: Path,
        selected: FittedCandidate,
        cv_results: list[dict[str, Any]],
        evaluations: dict[str, dict[str, Any]],
        splits: ExperimentSplits,
        leakage_report: dict[str, Any],
    ) -> None:
        """Write core selected-run artifact files."""

        write_yaml(
            self.run_dir / "runtime_config.yaml",
            runtime_config(Path(model_path.name), selected),
        )
        write_json(
            self.run_dir / "runtime_config.json",
            runtime_config(Path(model_path.name), selected),
        )
        write_json(self.run_dir / "candidate_cv_results.json", cv_results)
        write_json(self.run_dir / "length_bucket_selection.json", selected.threshold_tuning)
        write_json(self.run_dir / "splits.json", splits.split_ids())
        write_json(self.run_dir / "leakage_report.json", leakage_report)
        self.write_metrics_and_predictions(evaluations)

    def write_metrics_and_predictions(self, evaluations: dict[str, dict[str, Any]]) -> None:
        """Write metrics JSON and prediction JSONL for each evaluation split."""

        for split_name, payload in evaluations.items():
            write_json(self.run_dir / f"metrics_{split_name}.json", payload["metrics"])
            write_jsonl(self.run_dir / f"predictions_{split_name}.jsonl", payload["predictions"])


def model_filename(run_name: str) -> str:
    """Return the run-local best pipeline artifact name."""

    return f"best_pipeline_{run_name}.joblib"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows with stable key ordering."""

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_ready(row), sort_keys=True) + "\n")
