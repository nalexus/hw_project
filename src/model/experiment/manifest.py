"""Pure payload builders for experiment artifacts and summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.model.train.constants import (
    LENGTH_BUCKETS,
    SYNTHETIC_DATA_VERSION,
    SYNTHETIC_PROMPT_VERSION,
)
from src.model.train.schemas import FittedCandidate
from src.model.train.validators import TrainingConfig
from src.model.tune.candidates import candidate_to_dict


def runtime_config(model_path: Path, selected: FittedCandidate) -> dict[str, Any]:
    """Return API-compatible runtime config for the selected policy."""

    policy = selected.threshold_tuning["selected"]["policy"]
    config = {
        "model_path": str(model_path),
        "policy": policy["name"],
        "calibration": "sigmoid",
    }
    if policy["name"] == "global":
        config["threshold"] = policy["params"]["threshold"]
        return config
    config["default_threshold"] = policy["params"]["default_threshold"]
    config["bucket_thresholds"] = policy["params"]["bucket_thresholds"]
    return config


def build_metadata(
    selected: FittedCandidate,
    evaluations: dict[str, dict[str, Any]],
    config: TrainingConfig,
    model_path: Path,
    baseline_eval: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return compact metadata explaining the selected training run."""

    return {
        "runner": "src.model.experiment.runner",
        "selected_model_path": model_path,
        "promote_selected": config.promote_selected,
        "selected_candidate": candidate_to_dict(selected.candidate),
        "selected_calibration": "sigmoid",
        "selected_policy": selected.threshold_tuning["selected"]["policy"],
        "threshold_selection": threshold_selection_metadata(selected, config),
        "synthetic_data_version": SYNTHETIC_DATA_VERSION,
        "synthetic_prompt_version": SYNTHETIC_PROMPT_VERSION,
        "validation_metrics": compact_metrics(evaluations["validation"]["metrics"]),
        "test_metrics": compact_metrics(evaluations["test"]["metrics"]),
        "provided_other_metrics": compact_metrics(evaluations["provided_other"]["metrics"]),
        "baseline_comparison": baseline_eval,
    }


def threshold_selection_metadata(
    selected: FittedCandidate, config: TrainingConfig
) -> dict[str, Any]:
    """Return compact threshold-selection context for metadata."""

    return {
        "known_weight": 0.5,
        "ood_weight": 0.5,
        "minimum_known_balanced_accuracy": config.minimum_known_balanced_accuracy,
        "score_formula": selected.threshold_tuning["score_formula"],
    }


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return high-signal metrics for logs and metadata."""

    keys = [
        "sample_count",
        "known_accuracy",
        "known_balanced_accuracy",
        "ood_accuracy",
        "overall_accuracy",
        "macro_f1",
        "threshold_selection_score",
    ]
    return {key: metrics[key] for key in keys if key in metrics}


def experiment_log_row(run_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Return one flat experiment-log row."""

    policy = metadata["selected_policy"]
    validation = metadata["validation_metrics"]
    test = metadata["test_metrics"]
    provided_other = metadata["provided_other_metrics"]
    return {
        "run_dir": str(run_dir),
        "candidate_id": metadata["selected_candidate"]["candidate_id"],
        "policy": policy["name"],
        "thresholds": json.dumps(policy["params"], sort_keys=True),
        "validation_known_balanced_accuracy": validation["known_balanced_accuracy"],
        "validation_ood_accuracy": validation["ood_accuracy"],
        "test_known_balanced_accuracy": test["known_balanced_accuracy"],
        "test_ood_accuracy": test["ood_accuracy"],
        "provided_other_ood_accuracy": provided_other["ood_accuracy"],
    }


def summary_lines(
    run_dir: Path,
    model_path: Path,
    selected: FittedCandidate,
    evaluations: dict[str, dict[str, Any]],
) -> list[str]:
    """Return compact human-readable run summary lines."""

    return [
        *selection_summary_lines(selected, evaluations),
        f"Run directory: {run_dir}",
        f"Selected model artifact: {model_path}",
    ]


def selection_summary_lines(
    selected: FittedCandidate,
    evaluations: dict[str, dict[str, Any]],
) -> list[str]:
    """Return selected candidate, policy, and metric summary lines."""

    policy = selected.threshold_tuning["selected"]["policy"]
    lines = [
        f"Selected candidate: {selected.candidate.candidate_id}",
        format_threshold_policy_header(policy),
        "Values table:",
        *format_threshold_policy_values(policy),
        f"test: {format_metrics(evaluations['test']['metrics'])}",
        f"provided_other: {format_provided_other_metrics(evaluations['provided_other']['metrics'])}",
    ]
    return lines


def format_threshold_policy_header(policy: dict[str, Any]) -> str:
    """Format the selected reject-threshold policy type."""

    if policy["name"] == "global":
        return 'Minimum threshold for probability of classifier top class: type = "global"'
    return 'Minimum threshold for probability of classifier top class: type = "input length based"'


def format_threshold_policy_values(policy: dict[str, Any]) -> list[str]:
    """Format selected reject-threshold policy values for console output."""

    if policy["name"] == "global":
        return format_global_threshold(policy["params"]["threshold"])
    return format_bucket_thresholds(policy["params"]["bucket_thresholds"])


def format_global_threshold(threshold: float) -> list[str]:
    """Format one global reject threshold as readable summary lines."""

    return [
        "  scope              threshold_probability",
        f"  all input lengths  {threshold:.2f}",
    ]


def format_bucket_thresholds(policy: dict[str, float]) -> list[str]:
    """Format bucket reject thresholds as readable summary lines."""

    lines = ["  length_bucket  length_range_tokens  threshold_probability"]
    lines.extend(
        f"  {name:<13} {length_range(lower, upper):<19} {policy[name]:.2f}"
        for name, lower, upper in LENGTH_BUCKETS
    )
    return lines


def length_range(lower: int, upper: int | None) -> str:
    """Format one token-count range for threshold review output."""

    return f"{lower}+" if upper is None else f"{lower}-{upper}"


def format_metrics(metrics: dict[str, Any]) -> str:
    """Format the headline metrics used during review."""

    return (
        f"known_balanced_accuracy={metrics['known_balanced_accuracy']:.4f} "
        f"known_accuracy={metrics['known_accuracy']:.4f} "
        f"ood_accuracy={metrics['ood_accuracy']:.4f} "
        f"overall_accuracy={metrics['overall_accuracy']:.4f}"
    )


def format_provided_other_metrics(metrics: dict[str, Any]) -> str:
    """Format the provided-other final check without irrelevant known metrics."""

    return f"overall_accuracy={metrics['overall_accuracy']:.4f}"
