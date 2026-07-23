"""Unit tests for TF-IDF confidence and vocabulary-support OOD signals."""

import pandas as pd

from src.model.predict import TfidfOODPolicy, TfidfSignalExtractor
from src.model.train.trainer import ModelFactory


MODEL_CONFIG = {
    "pipeline_id": "test_unigram",
    "max_features": 100,
    "ngram_range": (1, 1),
    "min_df": 1,
    "C": 1.0,
    "class_weight": None,
}


def fitted_model():
    """Fit a tiny real pipeline with disjoint food and sport vocabularies."""

    model = ModelFactory().build_tfidf_logreg(MODEL_CONFIG)
    model.fit(
        ["recipe ingredients oven", "pasta sauce kitchen", "football team match", "tennis court serve"],
        ["food", "food", "sport", "sport"],
    )
    return model


def test_signal_extractor_reports_confidence_and_oov_coverage():
    """Verify known vocabulary has support while unseen vocabulary is OOV."""

    scores = TfidfSignalExtractor(fitted_model()).score(
        pd.Series(["recipe ingredients", "unseen vocabulary tokens"])
    )

    assert scores.loc[0, "raw_label"] == "food"
    assert scores.loc[0, "oov_ratio"] == 0.0
    assert scores.loc[1, "oov_ratio"] == 1.0
    assert scores["max_probability"].between(0, 1).all()


def test_ood_policy_rejects_unsupported_or_low_confidence_predictions():
    """Verify either signal can route a row to the direct other class."""

    results = pd.DataFrame(
        {
            "class": ["food", "other", "other"],
            "raw_label": ["food", "food", "food"],
            "max_probability": [0.80, 0.40, 0.80],
            "oov_ratio": [0.00, 0.00, 0.90],
        }
    )

    applied = TfidfOODPolicy(0.50, 0.25, "other").apply(results)

    assert applied["accepted"].tolist() == [True, False, False]
    assert applied["predicted_label"].tolist() == ["food", "other", "other"]
    assert applied["accepted_correct"].tolist() == [True, False, False]


def test_ood_policy_runtime_config_round_trips():
    """Verify persisted runtime fields recreate the same serving policy."""

    policy = TfidfOODPolicy(
        probability_threshold=0.18,
        max_oov_ratio=0.35,
        other_label="other",
    )
    runtime_config = policy.to_runtime_config("model.joblib")

    assert TfidfOODPolicy.from_mapping(runtime_config) == policy
