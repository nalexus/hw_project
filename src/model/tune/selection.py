"""Shared objective and ranking helpers for tuning."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.model.tune.validators import KNOWN_WEIGHT, OOD_WEIGHT


def threshold_selection_score(known_balanced_accuracy: float, ood_accuracy: float) -> float:
    """Score threshold candidates with equal known and OOD weights."""

    return KNOWN_WEIGHT * known_balanced_accuracy + OOD_WEIGHT * ood_accuracy


def threshold_selection_key(result: dict[str, Any]) -> tuple[float, float, float, float]:
    """Rank policy candidates by selected validation objective."""

    metrics = result["metrics"]
    return (
        metrics["threshold_selection_score"],
        metrics["known_balanced_accuracy"],
        metrics["ood_accuracy"],
        -policy_tie_breaker(result["policy"]),
    )


def policy_tie_breaker(policy: dict[str, Any]) -> float:
    """Return one threshold-like value for stable tie-breaking."""

    if policy["name"] == "global":
        return float(policy["params"]["threshold"])
    thresholds = policy["params"]["bucket_thresholds"].values()
    return float(np.mean(list(thresholds)))
