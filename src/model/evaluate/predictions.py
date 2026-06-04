"""Prediction construction and policy application helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.model.evaluate.callers import SklearnModelCaller
from src.model.evaluate.policies import ThresholdPolicy, ThresholdPolicyFactory
from src.model.evaluate.schemas import (
    FinalPrediction,
    ModelCallResult,
    RawPrediction,
    raw_prediction_dict,
)
from src.model.train.schemas import DocumentRecord


class RawPredictionBuilder(ABC):
    """Build raw evaluation predictions from model-call results."""

    @abstractmethod
    def build(
        self, records: list[DocumentRecord], result: ModelCallResult
    ) -> list[RawPrediction]:
        """Return raw predictions before rejection policy application."""


class DefaultRawPredictionBuilder(RawPredictionBuilder):
    """Build raw evaluation predictions from model-call results."""

    def build(
        self, records: list[DocumentRecord], result: ModelCallResult
    ) -> list[RawPrediction]:
        """Return raw predictions before rejection policy application."""

        rows = []
        for record, probs in zip(records, result.probabilities):
            order = np.argsort(probs)[::-1]
            rows.append(self._build_one(record, result.classes, probs, order))
        return rows

    def _build_one(
        self, record: DocumentRecord, classes: list[str], probs: Any, order: Any
    ) -> RawPrediction:
        """Build one raw prediction with model score and record metadata."""

        top_index = int(order[0])
        second_probability = float(probs[int(order[1])]) if len(order) > 1 else 0.0
        top_probability = float(probs[top_index])
        return RawPrediction(
            metadata=record.manifest_dict(include_text=True),
            raw_label=str(classes[top_index]),
            top_probability=top_probability,
            top2_margin=top_probability - second_probability,
        )


def build_prediction_rows(
    model: Any, records: list[DocumentRecord]
) -> list[dict[str, Any]]:
    """Create raw top-class probability rows for policy evaluation."""

    if not records:
        return []
    result = SklearnModelCaller().call(model, records)
    rows = DefaultRawPredictionBuilder().build(records, result)
    return [row.to_dict() for row in rows]


def prediction_row(record: DocumentRecord, classes, probs, order) -> dict[str, Any]:
    """Build one row with raw class probabilities and metadata."""

    result = ModelCallResult(
        classes=[str(label) for label in classes], probabilities=[probs]
    )
    return DefaultRawPredictionBuilder().build([record], result)[0].to_dict()


def apply_policy(row: dict[str, Any], policy: ThresholdPolicy | dict[str, Any]) -> str:
    """Apply one reject policy to a raw prediction row."""

    policy_obj = ThresholdPolicyFactory.from_dict(policy)
    return policy_obj.predict_label(RawPrediction.from_dict(row))


def rows_with_final_predictions(
    rows: list[dict[str, Any] | RawPrediction],
    policy: ThresholdPolicy | dict[str, Any],
) -> list[dict[str, Any]]:
    """Attach final labels produced by one reject policy."""

    policy_obj = ThresholdPolicyFactory.from_dict(policy)
    predictions = []
    for row in rows:
        raw = row if isinstance(row, RawPrediction) else RawPrediction.from_dict(row)
        predictions.append(FinalPrediction(raw, policy_obj.predict_label(raw)))
    return [row.to_dict() for row in predictions]


def raw_predictions_from_rows(
    rows: list[RawPrediction | dict[str, Any]],
) -> list[RawPrediction]:
    """Return typed raw predictions from mixed row representations."""

    return [
        row
        if isinstance(row, RawPrediction)
        else RawPrediction.from_dict(raw_prediction_dict(row))
        for row in rows
    ]
