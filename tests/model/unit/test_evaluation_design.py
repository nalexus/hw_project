import numpy as np

from src.model.evaluate import (
    GlobalThresholdPolicy,
    LengthBucketThresholdPolicy,
    ModelEvaluator,
    PipelineCallResult,
    RawPredictionsWithMarginBuilder,
    SklearnPipelineCaller,
    ThresholdPolicyFactory,
)
from src.model.train.schemas import DocumentRecord


class FixedProbabilityPipeline:
    """Small model double with sklearn-style probability output."""

    classes_ = np.array(["food", "sport"])

    def __init__(self, probabilities):
        """Store probabilities returned for every requested text."""

        self.probabilities = np.array(probabilities)

    def predict_proba(self, texts):
        """Return one probability row for each input text."""

        return np.tile(self.probabilities, (len(texts), 1))


def record(record_id: str, text: str = "short text") -> DocumentRecord:
    """Build one document record for evaluation unit tests."""

    return DocumentRecord(
        record_id=record_id,
        text=text,
        label="food",
        expected_label="food",
        source="provided_known",
        split="test",
        path=None,
        text_hash="hash",
        word_count=len(text.split()),
        length_bucket="ultra_short",
    )


def test_threshold_policy_factory_builds_supported_policy_objects():
    """Verify policy dictionaries are converted to policy objects."""

    global_policy = ThresholdPolicyFactory.from_dict(
        {"name": "global", "params": {"threshold": 0.7}}
    )
    bucket_policy = ThresholdPolicyFactory.from_dict(
        {
            "name": "length_bucket",
            "params": {
                "default_threshold": 0.5,
                "bucket_thresholds": {"ultra_short": 0.8},
            },
        }
    )

    assert isinstance(global_policy, GlobalThresholdPolicy)
    assert isinstance(bucket_policy, LengthBucketThresholdPolicy)
    assert bucket_policy.to_dict()["params"]["bucket_thresholds"]["ultra_short"] == 0.8


def test_sklearn_model_caller_returns_classes_and_probabilities():
    """Verify sklearn-specific model calling returns scores for records."""

    records = [record("a"), record("b")]
    model = FixedProbabilityPipeline([0.6, 0.4])

    result = SklearnPipelineCaller().call(model, records)

    assert result.classes == ["food", "sport"]
    assert result.probabilities.shape == (2, 2)


def test_raw_prediction_builder_preserves_record_metadata():
    """Verify raw prediction assembly is independent from model calling."""

    records = [record("a")]
    result = PipelineCallResult(["food", "sport"], np.array([[0.6, 0.4]]))

    prediction = RawPredictionsWithMarginBuilder().build(records, result)[0]

    assert prediction.raw_label == "food"
    assert prediction.top_probability == 0.6
    assert np.isclose(prediction.top2_margin, 0.2)
    assert prediction.to_dict()["record_id"] == "a"


def test_model_evaluator_returns_artifact_compatible_payload():
    """Verify evaluator facade returns current metrics and prediction shapes."""

    records = [record("a")]
    model = FixedProbabilityPipeline([0.6, 0.4])
    policy = {"name": "global", "params": {"threshold": 0.7}}

    evaluation = ModelEvaluator().evaluate(model, records, policy, ["food", "sport"])
    payload = evaluation.to_dict()

    assert payload["predictions"][0]["predicted_label"] == "other"
    assert payload["metrics"]["sample_count"] == 1
    assert "by_expected_label" in payload["metrics"]
