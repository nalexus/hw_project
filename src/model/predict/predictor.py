"""TF-IDF signals and the configured OOD-rejection policy."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any, Mapping

import numpy as np
import pandas as pd


class TfidfSignalExtractor:
    """Extract confidence and TF-IDF vocabulary-coverage signals."""

    def __init__(self, model: Any) -> None:
        """Cache the fitted vectorizer analyzer and vocabulary."""

        self.model = model
        vectorizer = model.named_steps["tfidf"]
        self.analyzer = vectorizer.build_analyzer()
        self.vocabulary = set(vectorizer.vocabulary_)

    def score(self, texts: pd.Series) -> pd.DataFrame:
        """Return one confidence and OOV-coverage row per text."""

        probabilities = self.model.predict_proba(texts)
        top_indices = probabilities.argmax(axis=1)
        return pd.DataFrame(
            {
                "raw_label": self.model.classes_[top_indices],
                "max_probability": probabilities.max(axis=1),
                "oov_ratio": [self._oov_ratio(text) for text in texts],
            },
            index=texts.index,
        )

    def _oov_ratio(self, text: str) -> float:
        """Measure the share of unigram tokens absent from the vocabulary."""

        tokens = [token for token in self.analyzer(text) if " " not in token]
        if not tokens:
            return 1.0
        return 1 - sum(token in self.vocabulary for token in tokens) / len(tokens)


@dataclass(frozen=True)
class TfidfOODPolicy:
    """Accept supported predictions and otherwise assign the configured OOD label."""

    probability_threshold: float
    max_oov_ratio: float
    other_label: str

    def apply(self, results: pd.DataFrame) -> pd.DataFrame:
        """Append acceptance and final-label columns to scored text rows."""

        accepted = self.accepts(results)
        values: dict[str, Any] = {
            "accepted": accepted,
            "predicted_label": np.where(
                accepted,
                results["raw_label"],
                self.other_label,
            ),
        }
        if "class" in results:
            raw_correct = results["raw_label"].eq(results["class"])
            values["raw_correct"] = raw_correct
            values["accepted_correct"] = accepted & raw_correct
        return results.assign(**values)

    def accepts(self, results: pd.DataFrame) -> pd.Series:
        """Return rows that satisfy both policy thresholds."""

        return (
            results["max_probability"] >= self.probability_threshold
        ) & (
            results["oov_ratio"] <= self.max_oov_ratio
        )

    def to_runtime_config(self, model_path: str) -> dict[str, float | str]:
        """Return the compact policy shape persisted beside a model artifact."""

        return {
            "model_path": model_path,
            "policy": "tfidf_ood",
            "probability_threshold": self.probability_threshold,
            "max_oov_ratio": self.max_oov_ratio,
            "other_label": self.other_label,
        }

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any] | None,
        default_threshold: float = 0.16,
    ) -> "TfidfOODPolicy":
        """Build a policy from runtime config, including legacy saved runs."""

        values = config or {}
        return cls(
            probability_threshold=float(
                values.get("probability_threshold", default_threshold)
            ),
            max_oov_ratio=float(values.get("max_oov_ratio", 1.0)),
            # Existing selected artifacts predate the configured runtime label.
            other_label=str(values.get("other_label", "other")),
        )


class PredictorMultiClass:
    """Apply the fitted TF-IDF model and OOD policy to document rows.

    ``predict`` is the single public inference path for both the API and
    offline evaluation. It preserves supplied row metadata, such as class or
    length bucket, and appends the raw model signals and final prediction.
    """

    def __init__(
        self,
        pipeline: Any,
        threshold: float = 0.16,
        threshold_policy: Mapping[str, Any] | None = None,
    ) -> None:
        """Store the fitted pipeline and its runtime OOD policy."""

        self.pipeline = pipeline
        self.policy = TfidfOODPolicy.from_mapping(threshold_policy, threshold)
        self.signal_extractor = TfidfSignalExtractor(pipeline)

    def predict(self, rows: pd.DataFrame | Iterable[str]) -> pd.DataFrame:
        """Return scored prediction rows, preserving metadata when supplied."""

        if isinstance(rows, pd.DataFrame):
            # Offline evaluation passes full rows so class and bucket survive scoring.
            results = rows.copy()
        else:
            # The API passes a batch of raw texts, which needs only the text column.
            texts = [rows] if isinstance(rows, str) else list(rows)
            results = pd.DataFrame({"text": texts})

        if "text" not in results:
            raise ValueError("Prediction rows must contain a text column.")

        signals = self.signal_extractor.score(results["text"])
        return self.policy.apply(results.join(signals))
