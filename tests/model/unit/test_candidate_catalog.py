"""Unit tests for the explicit notebook-derived candidate catalog."""

from pathlib import Path

from src.model.train.validators import TrainingConfigModel
from src.model.tune.tuner import CandidateCatalog


def test_candidate_catalog_loads_notebook_candidates():
    """Verify YAML exposes the three sklearn-ready baseline configurations."""

    config = TrainingConfigModel.from_yaml(Path("config/model/train.yaml"))
    candidates = CandidateCatalog(config.candidates_config_path).build()

    assert [candidate["pipeline_id"] for candidate in candidates] == [
        "tfidf_logreg_unigram",
        "tfidf_logreg_bigram",
        "tfidf_logreg_bigram_balanced",
    ]
    assert candidates[0] == {
        "pipeline_id": "tfidf_logreg_unigram",
        "max_features": 20_000,
        "ngram_range": (1, 1),
        "min_df": 1,
        "C": 1.0,
        "class_weight": None,
    }
