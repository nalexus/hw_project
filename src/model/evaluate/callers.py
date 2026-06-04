"""Model-calling adapters used during evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.model.evaluate.schemas import ModelCallResult
from src.model.train.schemas import DocumentRecord


class ModelCaller(ABC):
    """Call fitted models and return normalized evaluation scores."""

    @abstractmethod
    def call(self, model: Any, records: list[DocumentRecord]) -> ModelCallResult:
        """Return classes and probabilities for evaluation records."""


class SklearnModelCaller(ModelCaller):
    """Call sklearn-style classifiers that expose predict_proba and classes_."""

    def call(self, model: Any, records: list[DocumentRecord]) -> ModelCallResult:
        """Return class probabilities for record text values."""

        texts = [record.text for record in records]
        probabilities = model.predict_proba(texts) if records else np.empty((0, 0))
        return ModelCallResult(
            classes=[str(label) for label in getattr(model, "classes_", [])],
            probabilities=probabilities,
        )
