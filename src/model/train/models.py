"""Estimator construction and calibrated model fitting."""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.model.train.validators import TrainingConfig
from src.model.train.schemas import CandidateConfig, DocumentRecord


class ModelFactory:
    """Build sklearn estimators for configured text-classifier candidates."""

    @staticmethod
    def build_pipeline(candidate: CandidateConfig, random_state: int) -> Pipeline:
        """Create an uncalibrated TF-IDF plus logistic-regression pipeline."""

        classifier_params = dict(candidate.classifier_params)
        classifier_params.setdefault("random_state", random_state)
        classifier_params.setdefault("n_jobs", 1)
        classifier_params.setdefault("max_iter", 1000)
        return Pipeline(
            [
                ("tfidf", TfidfVectorizer(**candidate.tfidf_params)),
                ("classifier", LogisticRegression(**classifier_params)),
            ]
        )

    @staticmethod
    def build_calibrated(
        candidate: CandidateConfig, config: TrainingConfig
    ) -> CalibratedClassifierCV:
        """Create the selected sigmoid-calibrated estimator shape."""

        return CalibratedClassifierCV(
            estimator=ModelFactory.build_pipeline(candidate, config.random_state),
            method="sigmoid",
            cv=config.calibration_cv_folds,
        )

def fit_calibrated_model(candidate: CandidateConfig, records: list[DocumentRecord], config: TrainingConfig):
    """Fit the selected sigmoid-calibrated model on training records."""

    model = ModelFactory.build_calibrated(candidate, config)
    model.fit([record.text for record in records], [record.label for record in records])
    return model
