"""Model-calling adapters used during evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.model.evaluate.schemas import PipelineCallResult
from src.model.train.schemas import DocumentRecord


class PipelineCaller(ABC):
    """Call fitted models and return normalized evaluation scores."""

    @abstractmethod
    def call(self, pipeline: Any, records: list[DocumentRecord]) -> PipelineCallResult:
        """Return classes and probabilities for evaluation records."""


class SklearnPipelineCaller(PipelineCaller):
    """Call sklearn-style classifiers that expose predict_proba and classes_."""

    def call(self, pipeline: Any, records: list[DocumentRecord]) -> PipelineCallResult:
        """Return class probabilities for record text values."""

        texts = [record.text for record in records]
        probabilities = pipeline.predict_proba(texts) if records else np.empty((0, 0))
        return PipelineCallResult(
            classes=[str(label) for label in getattr(pipeline, "classes_", [])],
            probabilities=probabilities,
        )
