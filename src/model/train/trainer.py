"""Build the plain TF-IDF and logistic-regression baseline model."""

from __future__ import annotations

from typing import Any, Mapping

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class ModelFactory:
    """Build TF-IDF plus logistic-regression pipelines from candidate configs."""

    def __init__(self, random_state: int = 42) -> None:
        """Store the seed shared by every candidate fit."""

        self.random_state = random_state

    def build_tfidf_logreg(self, config: Mapping[str, Any]) -> Pipeline:
        """Create one plain pipeline using the notebook baseline settings."""

        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=config["max_features"],
                        ngram_range=tuple(config["ngram_range"]),
                        min_df=config["min_df"],
                        sublinear_tf=True,
                        stop_words="english",
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=config["C"],
                        class_weight=config["class_weight"],
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
