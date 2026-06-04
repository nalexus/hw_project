"""Classification metrics for clean model experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.model.evaluate.schemas import (
    FinalPrediction,
    MetricSummary,
    final_prediction_dict,
)
from src.model.train.constants import OTHER_LABEL


class MetricsCalculator(ABC):
    """Calculate evaluation metrics for final prediction rows."""

    @abstractmethod
    def summarize(
        self,
        rows: list[FinalPrediction | dict[str, Any]],
        known_labels: list[str],
    ) -> MetricSummary:
        """Return metrics for final prediction rows."""


class ClassificationMetricsCalculator(MetricsCalculator):
    """Calculate classification metrics used by clean model experiments."""

    def summarize(
        self,
        rows: list[FinalPrediction | dict[str, Any]],
        known_labels: list[str],
    ) -> MetricSummary:
        """Return metrics for final prediction rows."""

        return MetricSummary(_metric_summary_dict(normalize_rows(rows), known_labels))


def metric_summary(
    rows: list[FinalPrediction | dict[str, Any]], known_labels: list[str]
) -> dict[str, Any]:
    """Return overall and grouped metrics for final prediction rows."""

    return ClassificationMetricsCalculator().summarize(rows, known_labels).to_dict()


def _metric_summary_dict(
    rows: list[dict[str, Any]], known_labels: list[str]
) -> dict[str, Any]:
    """Return overall and grouped metrics for final prediction rows."""

    if not rows:
        return empty_metrics()
    expected = [row["expected_label"] for row in rows]
    predicted = [row["predicted_label"] for row in rows]
    metrics = {
        "sample_count": len(rows),
        "overall_accuracy": safe_accuracy(expected, predicted),
        "known_accuracy": filtered_accuracy(rows, known=True),
        "known_balanced_accuracy": known_balanced_accuracy(rows, known_labels),
        "ood_accuracy": filtered_accuracy(rows, known=False),
        "ood_recall": filtered_accuracy(rows, known=False),
        "macro_f1": safe_macro_f1(expected, predicted, [*known_labels, OTHER_LABEL]),
    }
    metrics["by_expected_label"] = grouped_metrics(rows, "expected_label", known_labels)
    metrics["by_source"] = grouped_metrics(rows, "source", known_labels)
    metrics["by_length_bucket"] = grouped_metrics(rows, "length_bucket", known_labels)
    return metrics


def empty_metrics() -> dict[str, Any]:
    """Return stable zero metrics for empty evaluation inputs."""

    return {
        "sample_count": 0,
        "overall_accuracy": 0.0,
        "known_accuracy": 0.0,
        "known_balanced_accuracy": 0.0,
        "ood_accuracy": 0.0,
        "ood_recall": 0.0,
        "macro_f1": 0.0,
        "by_expected_label": {},
        "by_source": {},
        "by_length_bucket": {},
    }


def known_balanced_accuracy(rows: list[dict[str, Any]], known_labels: list[str]) -> float:
    """Return macro recall across known labels only."""

    recalls = []
    for label in known_labels:
        label_rows = [row for row in rows if row["expected_label"] == label]
        if label_rows:
            correct = sum(row["predicted_label"] == label for row in label_rows)
            recalls.append(correct / len(label_rows))
    return float(np.mean(recalls)) if recalls else 0.0


def filtered_accuracy(rows: list[dict[str, Any]], known: bool) -> float:
    """Return accuracy over known or OOD rows."""

    selected = [row for row in rows if (row["expected_label"] != OTHER_LABEL) == known]
    if not selected:
        return 0.0
    return safe_accuracy(
        [row["expected_label"] for row in selected],
        [row["predicted_label"] for row in selected],
    )


def grouped_metrics(
    rows: list[dict[str, Any]], key: str, known_labels: list[str]
) -> dict[str, Any]:
    """Return metric summaries grouped by one row field."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {
        name: metric_summary_shallow(items, known_labels)
        for name, items in sorted(groups.items())
    }


def normalize_rows(
    rows: list[FinalPrediction | dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return dictionary rows from typed or dictionary final predictions."""

    return [final_prediction_dict(row) for row in rows]


def metric_summary_shallow(
    rows: list[dict[str, Any]], known_labels: list[str]
) -> dict[str, Any]:
    """Return metrics without recursive grouping."""

    expected = [row["expected_label"] for row in rows]
    predicted = [row["predicted_label"] for row in rows]
    return {
        "sample_count": len(rows),
        "accuracy": safe_accuracy(expected, predicted),
        "known_accuracy": filtered_accuracy(rows, known=True),
        "ood_recall": filtered_accuracy(rows, known=False),
        "known_balanced_accuracy": known_balanced_accuracy(rows, known_labels),
    }


def safe_accuracy(expected: list[str], predicted: list[str]) -> float:
    """Return accuracy or zero for empty inputs."""

    return float(accuracy_score(expected, predicted)) if expected else 0.0


def safe_macro_f1(expected: list[str], predicted: list[str], labels: list[str]) -> float:
    """Return macro F1 with a stable zero-division policy."""

    if not expected:
        return 0.0
    return float(
        f1_score(expected, predicted, labels=labels, average="macro", zero_division=0)
    )
