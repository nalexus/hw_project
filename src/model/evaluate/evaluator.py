"""Facade for evaluating fitted models on document records."""

from __future__ import annotations

from typing import Any

from src.model.evaluate.callers import PipelineCaller, SklearnPipelineCaller
from src.model.evaluate.metrics import ClassificationMetricsCalculator, MetricsCalculator
from src.model.evaluate.policies import ThresholdPolicy, ThresholdPolicyFactory
from src.model.evaluate.predictions import (
    DefaultRawPredictionBuilder,
    RawPredictionBuilder,
)
from src.model.evaluate.schemas import EvaluationResult, FinalPrediction, RawPrediction
from src.model.train.schemas import DocumentRecord


class ModelEvaluator:
    """Evaluate fitted models with injectable evaluation collaborators."""

    def __init__(
        self,
        model_caller: PipelineCaller | None = None,
        raw_prediction_builder: RawPredictionBuilder | None = None,
        metrics_calculator: MetricsCalculator | None = None,
    ) -> None:
        """Store evaluation collaborators with sklearn defaults."""

        self.model_caller = model_caller or SklearnPipelineCaller()
        self.raw_prediction_builder = (
            raw_prediction_builder or DefaultRawPredictionBuilder()
        )
        self.metrics_calculator = metrics_calculator or ClassificationMetricsCalculator()

    def evaluate(
        self,
        model: Any,
        records: list[DocumentRecord],
        policy: ThresholdPolicy | dict[str, Any],
        known_labels: list[str],
    ) -> EvaluationResult:
        """Return predictions and metrics for one evaluation split."""

        raw_predictions = self._raw_predictions(model, records)
        policy_obj = ThresholdPolicyFactory.from_dict(policy)
        final_predictions = [
            FinalPrediction(row, policy_obj.predict_label(row))
            for row in raw_predictions
        ]
        metrics = self.metrics_calculator.summarize(final_predictions, known_labels)
        return EvaluationResult(metrics=metrics, predictions=final_predictions)

    def _raw_predictions(
        self, model: Any, records: list[DocumentRecord]
    ) -> list[RawPrediction]:
        """Return raw predictions without calling the model for empty input."""

        if not records:
            return []
        result = self.model_caller.call(model, records)
        return self.raw_prediction_builder.build(records, result)
