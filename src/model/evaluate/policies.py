"""Threshold policy interfaces and implementations for evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.model.evaluate.schemas import RawPrediction, raw_prediction_dict
from src.model.train.constants import OTHER_LABEL


class ThresholdPolicy(ABC):
    """Select final labels from raw model probabilities."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable policy name."""

    @abstractmethod
    def predict_label(self, row: RawPrediction) -> str:
        """Return the final label for one raw prediction."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Return the current artifact-compatible policy payload."""


class GlobalThresholdPolicy(ThresholdPolicy):
    """Reject predictions below one global top-class probability threshold."""

    def __init__(self, threshold: float) -> None:
        """Store the global rejection threshold."""

        self.threshold = threshold

    @property
    def name(self) -> str:
        """Return the stable policy name."""

        return "global"

    def predict_label(self, row: RawPrediction) -> str:
        """Return raw label when probability meets the global threshold."""

        return row.raw_label if row.top_probability >= self.threshold else OTHER_LABEL

    def to_dict(self) -> dict[str, Any]:
        """Return the current artifact-compatible policy payload."""

        return {"name": self.name, "params": {"threshold": self.threshold}}


class LengthBucketThresholdPolicy(ThresholdPolicy):
    """Reject predictions using length-bucket-specific thresholds."""

    def __init__(
        self, default_threshold: float, bucket_thresholds: dict[str, float]
    ) -> None:
        """Store fallback and bucket-specific rejection thresholds."""

        self.default_threshold = default_threshold
        self.bucket_thresholds = dict(bucket_thresholds)

    @property
    def name(self) -> str:
        """Return the stable policy name."""

        return "length_bucket"

    def predict_label(self, row: RawPrediction) -> str:
        """Return raw label when probability meets the row bucket threshold."""

        row_dict = raw_prediction_dict(row)
        threshold = self.bucket_thresholds.get(
            row_dict["length_bucket"], self.default_threshold
        )
        return row.raw_label if row.top_probability >= threshold else OTHER_LABEL

    def to_dict(self) -> dict[str, Any]:
        """Return the current artifact-compatible policy payload."""

        return {
            "name": self.name,
            "params": {
                "default_threshold": self.default_threshold,
                "bucket_thresholds": self.bucket_thresholds,
            },
        }


class ThresholdPolicyFactory:
    """Build threshold policy objects from supported dictionary shapes."""

    @staticmethod
    def from_dict(policy: ThresholdPolicy | dict[str, Any]) -> ThresholdPolicy:
        """Return a threshold policy from artifact or runtime config payloads."""

        if isinstance(policy, ThresholdPolicy):
            return policy
        if "name" in policy:
            return ThresholdPolicyFactory._from_artifact_policy(policy)
        if "policy" in policy:
            return ThresholdPolicyFactory._from_runtime_policy(policy)
        raise ValueError(f"Unsupported policy payload: {policy}")

    @staticmethod
    def _from_artifact_policy(policy: dict[str, Any]) -> ThresholdPolicy:
        """Return a policy from the experiment artifact policy shape."""

        params = policy["params"]
        if policy["name"] == "global":
            return GlobalThresholdPolicy(float(params["threshold"]))
        if policy["name"] == "length_bucket":
            return LengthBucketThresholdPolicy(
                float(params["default_threshold"]),
                {key: float(value) for key, value in params["bucket_thresholds"].items()},
            )
        raise ValueError(f"Unsupported policy: {policy['name']}")

    @staticmethod
    def _from_runtime_policy(policy: dict[str, Any]) -> ThresholdPolicy:
        """Return a policy from the runtime config policy shape."""

        if policy["policy"] == "global":
            return GlobalThresholdPolicy(float(policy["threshold"]))
        if policy["policy"] == "length_bucket":
            return LengthBucketThresholdPolicy(
                float(policy["default_threshold"]),
                {key: float(value) for key, value in policy["bucket_thresholds"].items()},
            )
        raise ValueError(f"Unsupported policy: {policy['policy']}")
