"""Unit tests for current model and OOD-policy evaluation summaries."""

from pathlib import Path

import pandas as pd

from src.model.data_preparation.loader import DocumentRecord
from src.model.data_preparation.text import TextProcessor
from src.model.evaluate import ClassificationScorer, ModelEvaluator, summarize_gate
from src.model.predict import TfidfOODPolicy
from src.model.train.trainer import ModelFactory
from src.model.train.validators import TrainingConfigModel


MODEL_CONFIG = {
    "pipeline_id": "test_unigram",
    "max_features": 100,
    "ngram_range": (1, 1),
    "min_df": 1,
    "C": 1.0,
    "class_weight": None,
}


def fitted_model():
    """Fit a tiny real pipeline for evaluation behavior tests."""

    model = ModelFactory().build_tfidf_logreg(MODEL_CONFIG)
    model.fit(
        ["recipe ingredients oven", "pasta sauce kitchen", "football team match", "tennis court serve"],
        ["food", "food", "sport", "sport"],
    )
    return model


def test_classification_scorer_reports_standard_multiclass_metrics():
    """Verify model-selection metrics are calculated from raw labels."""

    scores = ClassificationScorer(["food", "sport"]).score_predictions(
        ["food", "sport", "sport"],
        ["food", "food", "sport"],
    )

    assert scores["accuracy"] == 2 / 3
    assert scores["balanced_accuracy"] == 0.75
    assert scores["macro_f1"] > 0


def test_summarize_gate_reports_coverage_and_accepted_accuracy():
    """Verify policy summaries distinguish raw accuracy from accepted accuracy."""

    results = pd.DataFrame(
        {
            "class": ["food", "sport", "sport"],
            "accepted": [True, True, False],
            "raw_correct": [True, False, True],
            "accepted_correct": [True, False, False],
        }
    )

    summary = summarize_gate(results).iloc[0]

    assert summary["num_examples"] == 3
    assert summary["accepted"] == 2
    assert summary["raw_accuracy"] == 2 / 3
    assert summary["coverage"] == 2 / 3
    assert summary["accepted_accuracy"] == 0.5


def test_model_evaluator_reports_oof_signal_auroc():
    """Verify both policy signals rank OOF classification errors above successes."""

    oof_results = pd.DataFrame(
        {
            "class": ["food", "food", "food", "food"],
            "raw_label": ["food", "sport", "food", "sport"],
            "max_probability": [0.90, 0.10, 0.80, 0.20],
            "oov_ratio": [0.10, 0.90, 0.20, 0.80],
        }
    )

    diagnostics = ModelEvaluator().oof_signal_auroc(oof_results)

    assert diagnostics["confidence_error_auroc"] == 1.0
    assert diagnostics["oov_error_auroc"] == 1.0


def test_model_evaluator_returns_known_and_other_results():
    """Verify evaluator preserves separate known-class and OOD summaries."""

    model = fitted_model()
    evaluator = ModelEvaluator()
    text_processor = TextProcessor.from_config(
        TrainingConfigModel.from_yaml(
            Path("config/model/train.yaml")
        ).text_config_path
    )
    known = [
        DocumentRecord.from_text(
            "food", "food.txt", "recipe ingredients oven", text_processor
        ),
        DocumentRecord.from_text(
            "sport", "sport.txt", "football team match", text_processor
        ),
    ]
    other = [
        DocumentRecord.from_text(
            "other", "other.txt", "unseen vocabulary tokens", text_processor
        )
    ]

    known_evaluation = evaluator.evaluate_known(
        model,
        known,
        TfidfOODPolicy(0.0, 1.0, "other"),
    )
    other_evaluation = evaluator.evaluate_other(
        model,
        other,
        TfidfOODPolicy(1.0, 1.0, "other"),
    )

    assert known_evaluation.raw_metrics["balanced_accuracy"] == 1.0
    assert known_evaluation.overall.loc[0, "accepted"] == 2
    assert other_evaluation.correct_predictions == 1
    assert other_evaluation.accuracy == 1.0
