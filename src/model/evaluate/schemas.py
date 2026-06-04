"""Typed payloads used by model evaluation components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PipelineCallResult:
    """Normalized model output with classes and probabilities for records."""

    classes: list[str]
    probabilities: Any


@dataclass(frozen=True)
class RawPrediction:
    """Model score and record metadata before threshold policy application."""

    metadata: dict[str, Any]
    raw_label: str
    top_probability: float
    top2_margin: float

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "RawPrediction":
        """Build a raw prediction from the current dictionary row shape."""

        reserved = {"raw_label", "top_probability", "top2_margin", "predicted_label"}
        metadata = {key: value for key, value in row.items() if key not in reserved}
        return cls(
            metadata=metadata,
            raw_label=str(row["raw_label"]),
            top_probability=float(row["top_probability"]),
            top2_margin=float(row.get("top2_margin", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the current artifact-compatible raw prediction payload."""

        return {
            **self.metadata,
            "raw_label": self.raw_label,
            "top_probability": self.top_probability,
            "top2_margin": self.top2_margin,
        }


@dataclass(frozen=True)
class FinalPrediction:
    """Prediction row after applying a threshold policy."""

    raw: RawPrediction
    predicted_label: str

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "FinalPrediction":
        """Build a final prediction from the current dictionary row shape."""

        return cls(
            raw=RawPrediction.from_dict(row),
            predicted_label=str(row["predicted_label"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the current artifact-compatible final prediction payload."""

        return {**self.raw.to_dict(), "predicted_label": self.predicted_label}


@dataclass(frozen=True)
class MetricSummary:
    """Dictionary-backed metric summary with an explicit domain type."""

    values: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the current artifact-compatible metrics payload."""

        return self.values


@dataclass(frozen=True)
class EvaluationResult:
    """Predictions and metrics for one evaluated split."""

    metrics: MetricSummary
    predictions: list[FinalPrediction]

    def to_dict(self) -> dict[str, Any]:
        """Return the current artifact-compatible evaluation payload."""

        return {
            "metrics": self.metrics.to_dict(),
            "predictions": [row.to_dict() for row in self.predictions],
        }


def raw_prediction_dict(row: RawPrediction | Mapping[str, Any]) -> dict[str, Any]:
    """Return a dictionary view for raw prediction-like rows."""

    return row.to_dict() if isinstance(row, RawPrediction) else dict(row)


def final_prediction_dict(row: FinalPrediction | Mapping[str, Any]) -> dict[str, Any]:
    """Return a dictionary view for final prediction-like rows."""

    return row.to_dict() if isinstance(row, FinalPrediction) else dict(row)
