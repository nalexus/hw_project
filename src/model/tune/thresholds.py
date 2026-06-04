"""Reject-policy threshold tuning for selected length-bucket behavior."""

from __future__ import annotations

from typing import Any

from src.model.evaluate.metrics import metric_summary
from src.model.evaluate.policies import (
    GlobalThresholdPolicy,
    LengthBucketThresholdPolicy,
    ThresholdPolicy,
)
from src.model.evaluate.predictions import rows_with_final_predictions
from src.model.train.constants import LENGTH_BUCKETS
from src.model.tune.validators import default_threshold_grid
from src.model.tune.selection import threshold_selection_key, threshold_selection_score


def threshold_values(step: float) -> list[float]:
    """Return threshold search values from 0.00 through 1.00."""

    return list(default_threshold_grid(step))


def tune_length_bucket_thresholds(
    rows: list[dict[str, Any]],
    thresholds: list[float],
    known_labels: list[str],
    minimum_known_balanced_accuracy: float,
) -> dict[str, Any]:
    """Select a length-bucket reject policy on validation rows."""

    global_tuning = tune_global_threshold(
        rows, thresholds, known_labels, minimum_known_balanced_accuracy
    )
    default_threshold = global_tuning["selected"]["policy"]["params"]["threshold"]
    bucket_thresholds = {name: default_threshold for name, _, _ in LENGTH_BUCKETS}
    steps = []
    for bucket_name in bucket_thresholds:
        best = best_bucket_threshold(
            rows,
            thresholds,
            known_labels,
            minimum_known_balanced_accuracy,
            default_threshold,
            bucket_thresholds,
            bucket_name,
        )
        bucket_thresholds = best["policy"]["params"]["bucket_thresholds"]
        steps.append(best)
    policy = length_bucket_policy(default_threshold, bucket_thresholds)
    selected = evaluate_policy_result(rows, policy, known_labels)
    return {
        "selection_mode": "constrained",
        "score_formula": "0.5 * val_known_balanced_accuracy + 0.5 * val_ood_accuracy",
        "global_start": global_tuning,
        "bucket_steps": steps,
        "selected": selected,
    }


def tune_global_threshold(
    rows: list[dict[str, Any]],
    thresholds: list[float],
    known_labels: list[str],
    minimum_known_balanced_accuracy: float,
) -> dict[str, Any]:
    """Select one global threshold with a known-accuracy constraint."""

    candidates = [
        evaluate_policy_result(rows, GlobalThresholdPolicy(threshold), known_labels)
        for threshold in thresholds
    ]
    feasible = feasible_results(candidates, minimum_known_balanced_accuracy)
    return {
        "selection_mode": "constrained" if feasible else "fallback_no_feasible_threshold",
        "selected": max(feasible or candidates, key=threshold_selection_key),
        "candidates": candidates,
    }


def best_bucket_threshold(
    rows: list[dict[str, Any]],
    thresholds: list[float],
    known_labels: list[str],
    minimum_known_balanced_accuracy: float,
    default_threshold: float,
    current_thresholds: dict[str, float],
    bucket_name: str,
) -> dict[str, Any]:
    """Return the best threshold update for one length bucket."""

    candidates = []
    for threshold in thresholds:
        thresholds_by_bucket = {**current_thresholds, bucket_name: threshold}
        policy = length_bucket_policy(default_threshold, thresholds_by_bucket)
        candidates.append(evaluate_policy_result(rows, policy, known_labels))
    feasible = feasible_results(candidates, minimum_known_balanced_accuracy)
    return max(feasible or candidates, key=threshold_selection_key)


def evaluate_policy_result(
    rows: list[dict[str, Any]],
    policy: ThresholdPolicy,
    known_labels: list[str],
) -> dict[str, Any]:
    """Evaluate one policy and attach threshold-selection score."""

    metrics = metric_summary(rows_with_final_predictions(rows, policy), known_labels)
    metrics["threshold_selection_score"] = threshold_selection_score(
        metrics["known_balanced_accuracy"], metrics["ood_accuracy"]
    )
    return {"policy": policy.to_dict(), "metrics": metrics}


def feasible_results(
    candidates: list[dict[str, Any]], minimum_known_balanced_accuracy: float
) -> list[dict[str, Any]]:
    """Filter candidates that satisfy known-class quality constraint."""

    return [
        item
        for item in candidates
        if item["metrics"]["known_balanced_accuracy"] >= minimum_known_balanced_accuracy
    ]


def length_bucket_policy(
    default_threshold: float, bucket_thresholds: dict[str, float]
) -> LengthBucketThresholdPolicy:
    """Build one length-bucket threshold policy."""

    return LengthBucketThresholdPolicy(default_threshold, bucket_thresholds)
