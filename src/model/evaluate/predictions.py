"""Prediction construction and policy application helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.model.evaluate.callers import SklearnPipelineCaller
from src.model.evaluate.policies import ThresholdPolicy, ThresholdPolicyFactory
from src.model.evaluate.schemas import (
    FinalPrediction,
    PipelineCallResult,
    RawPrediction,
    raw_prediction_dict,
)
from src.model.train.schemas import DocumentRecord


class TopKSorter:
    """Return the top k class indices ordered by probability."""

    def __init__(self, k: int) -> None:
        """Store the number of top classes to return."""
        self.k = k

    def top_k_sort(self, probs: Any) -> list[int]:
        """Return the highest and second-highest probability indices."""

        topk_idx = np.argpartition(probs, -self.k)[-self.k :]
        ordered_topk = topk_idx[np.argsort(probs[topk_idx])[::-1]]
        ordered_topk = list(int(i) for i in ordered_topk)
        return ordered_topk


class RawPredictionsBuilder(ABC):
    """Builds customary raw predictions output from model-call results."""

    def build(
        self, records: list[DocumentRecord], result: PipelineCallResult
    ) -> list[RawPrediction]:
        """Return raw predictions before rejection policy application."""

        return [
            self._build_one(record, result.classes, probs)
            for record, probs in zip(records, result.probabilities)
        ]

    @abstractmethod
    def _build_one(
        self, record: DocumentRecord, classes: list[str], probs: Any
    ) -> RawPrediction:
        """Build one raw prediction from a record and one probability row."""


class RawPredictionsWithMarginBuilder(RawPredictionsBuilder):
    """Build raw predictions with top class probability and top-two classes margin."""

    def __init__(self, sorter: TopKSorter | None = None) -> None:
        """Store the sorter used to rank class probabilities."""

        self.sorter = sorter or TopKSorter(k=2)

    def _build_one(
        self, record: DocumentRecord, classes: list[str], probs: Any
    ) -> RawPrediction:
        """Build one raw prediction with top-label confidence and margin."""
        idxs = self.sorter.top_k_sort(probs)

        return RawPrediction(
            metadata=record.manifest_dict(include_text=True),
            raw_label=str(classes[idxs[0]]),
            top_probability=float(probs[idxs[0]]),
            top2_margin=float(probs[idxs[0]] - probs[idxs[1]]),
        )


def build_prediction_rows(
    model: Any, records: list[DocumentRecord]
) -> list[dict[str, Any]]:
    """Create raw top-class probability rows for policy evaluation."""

    if not records:
        return []
    result = SklearnPipelineCaller().call(model, records)
    rows = RawPredictionsWithMarginBuilder().build(records, result)
    return [row.to_dict() for row in rows]


def prediction_row(record: DocumentRecord, classes, probs, order) -> dict[str, Any]:
    """Build one row with raw class probabilities and metadata."""

    result = PipelineCallResult(
        classes=[str(label) for label in classes], probabilities=[probs]
    )
    return RawPredictionsWithMarginBuilder().build([record], result)[0].to_dict()


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
