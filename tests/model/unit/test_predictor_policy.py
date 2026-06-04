import numpy as np

from src.model.predict import PredictorMultiClass, length_bucket


class FixedProbabilityPipeline:
    """Small test double that returns the same class probabilities."""

    classes_ = np.array(["food", "sport"])

    def __init__(self, probabilities):
        """Store probabilities returned for every requested text."""

        self.probabilities = np.array(probabilities)

    def predict_proba(self, texts):
        """Return one fixed probability row per input text."""

        return np.tile(self.probabilities, (len(texts), 1))


def repeated_text(count: int) -> str:
    """Build deterministic text with the requested token count."""

    return " ".join(["word"] * count)


def test_global_threshold_preserves_baseline_reject_behavior():
    """Verify the old scalar threshold behavior remains supported."""

    pipeline = FixedProbabilityPipeline([0.84, 0.16])
    predictor = PredictorMultiClass(pipeline=pipeline, threshold=0.90)

    prediction = predictor.predict(np.array(["I am having breakfast"]))[0]

    assert prediction == "other"


def test_length_bucket_policy_uses_bucket_specific_thresholds():
    """Verify the selected runtime policy can vary threshold by length."""

    pipeline = FixedProbabilityPipeline([0.60, 0.40])
    policy = {
        "name": "length_bucket",
        "params": {
            "default_threshold": 0.50,
            "bucket_thresholds": {
                "ultra_short": 0.70,
                "medium": 0.50,
            },
        },
    }
    predictor = PredictorMultiClass(pipeline=pipeline, threshold_policy=policy)
    texts = np.array(["short text", repeated_text(130)], dtype=object)

    predictions = predictor.predict(texts)

    assert length_bucket(texts[0]) == "ultra_short"
    assert length_bucket(texts[1]) == "medium"
    assert predictions == ["other", "food"]


def test_runtime_config_policy_is_accepted_directly():
    """Verify runtime_config.json policy shape works without translation code."""

    pipeline = FixedProbabilityPipeline([0.55, 0.45])
    runtime_config = {
        "policy": "length_bucket",
        "default_threshold": 0.50,
        "bucket_thresholds": {"ultra_short": 0.60, "medium": 0.50},
    }
    predictor = PredictorMultiClass(
        pipeline=pipeline, threshold_policy=runtime_config
    )

    predictions = predictor.predict(np.array(["tiny note", repeated_text(130)]))

    assert predictions == ["other", "food"]
