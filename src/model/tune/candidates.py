"""Candidate search-space and known-class CV tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from src.model.train.validators import TrainingConfig
from src.model.train.models import ModelFactory
from src.model.train.schemas import CandidateConfig, DocumentRecord


@dataclass
class CvProgress:
    """Print compact cross-validation progress without fold metrics."""

    total_candidates: int

    def advance(
        self, candidate_index: int, current_fold: int, fold_total: int
    ) -> None:
        """Advance the progress bar by one completed fit."""

        percent = 100 * candidate_index / self.total_candidates if self.total_candidates else 100
        fold_percent = 100 * current_fold / fold_total if fold_total else 100
        width = 24
        filled = round(width * candidate_index / self.total_candidates) if self.total_candidates else width
        bar = "#" * filled + "-" * (width - filled)
        is_complete = candidate_index == self.total_candidates and current_fold == fold_total
        print(
            f"\rCandidates cross-validation progress [{bar}] "
            f"{candidate_index}/{self.total_candidates} of candidates ({percent:.1f}%) "
            f"{current_fold}/{fold_total} folds ({fold_percent:.0f}%)",
            end="" if not is_complete else "\n",
            flush=True,
        )


def build_candidate_configs(limit: int | None = None) -> list[CandidateConfig]:
    """Return the diverse TF-IDF/logistic-regression search space."""

    raw_configs = [
        ("c01_uni_min1_c1_bal", (1, 1), 1, True, 50000, "balanced", 1.0),
        ("c02_uni_min2_c1_bal", (1, 1), 2, True, 50000, "balanced", 1.0),
        ("c03_bi_min1_c1_bal", (1, 2), 1, True, 50000, "balanced", 1.0),
        ("c04_bi_min2_c1_bal", (1, 2), 2, True, 50000, "balanced", 1.0),
        ("c05_bi_min3_c1_bal", (1, 2), 3, True, 50000, "balanced", 1.0),
        ("c06_bi_min1_c03_bal", (1, 2), 1, True, 50000, "balanced", 0.3),
        ("c07_bi_min1_c3_bal", (1, 2), 1, True, 50000, "balanced", 3.0),
        ("c08_bi_min2_c03_bal", (1, 2), 2, True, 50000, "balanced", 0.3),
        ("c09_bi_min2_c3_bal", (1, 2), 2, True, 50000, "balanced", 3.0),
        ("c10_tri_min1_c1_bal", (1, 3), 1, True, 50000, "balanced", 1.0),
        ("c11_uni_min1_c1_none", (1, 1), 1, True, 50000, None, 1.0),
        ("c12_bi_min1_c1_none", (1, 2), 1, True, 50000, None, 1.0),
        ("c13_bi_min2_c1_none", (1, 2), 2, True, 50000, None, 1.0),
        ("c14_bi_min1_c1_rawtf", (1, 2), 1, False, 50000, "balanced", 1.0),
        ("c15_bi_min2_c1_rawtf", (1, 2), 2, False, 50000, "balanced", 1.0),
        ("c16_bi_min1_c1_20k", (1, 2), 1, True, 20000, "balanced", 1.0),
        ("c17_bi_min1_c1_allfeat", (1, 2), 1, True, None, "balanced", 1.0),
        ("c18_tri_min2_c1_bal", (1, 3), 2, True, 50000, "balanced", 1.0),
        ("c19_bi_min1_c10_bal", (1, 2), 1, True, 50000, "balanced", 10.0),
        ("c20_uni_min1_c3_none", (1, 1), 1, True, 50000, None, 3.0),
    ]
    candidates = [candidate_from_raw_config(config) for config in raw_configs]
    return candidates[:limit] if limit else candidates


def candidate_from_raw_config(config: tuple[Any, ...]) -> CandidateConfig:
    """Convert one compact raw candidate tuple into a typed config."""

    candidate_id, ngram_range, min_df, sublinear_tf, max_features, weight, c_value = config
    return CandidateConfig(
        candidate_id=candidate_id,
        tfidf_params={
            "max_features": max_features,
            "ngram_range": ngram_range,
            "stop_words": "english",
            "sublinear_tf": sublinear_tf,
            "min_df": min_df,
        },
        classifier_params={"C": c_value, "class_weight": weight, "solver": "lbfgs"},
    )


def candidate_to_dict(candidate: CandidateConfig) -> dict[str, Any]:
    """Convert a candidate into JSON-safe metadata."""

    return {
        "candidate_id": candidate.candidate_id,
        "tfidf_params": candidate.tfidf_params,
        "classifier_params": candidate.classifier_params,
    }


def cross_validate_candidates(
    records: list[DocumentRecord], config: TrainingConfig, known_labels: list[str]
) -> list[dict[str, Any]]:
    """Score candidates with known-class stratified cross-validation."""

    texts = [record.text for record in records]
    labels = np.array([record.label for record in records])
    candidates = build_candidate_configs(config.candidate_limit)
    progress = CvProgress(total_candidates=len(candidates))
    splitter = StratifiedKFold(
        n_splits=config.cv_folds, shuffle=True, random_state=config.random_state
    )
    results = [
        score_candidate(candidate, texts, labels, splitter, config, known_labels, progress, index)
        for index, candidate in enumerate(candidates, 1)
    ]
    return sorted(results, key=cv_selection_key, reverse=True)


def score_candidate(
    candidate: CandidateConfig,
    texts: list[str],
    labels: np.ndarray,
    splitter: StratifiedKFold,
    config: TrainingConfig,
    known_labels: list[str],
    progress: CvProgress,
    candidate_index: int,
) -> dict[str, Any]:
    """Score one candidate and update aggregate CV progress."""

    folds = []
    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(texts, labels), 1):
        model = ModelFactory.build_pipeline(candidate, config.random_state)
        model.fit([texts[index] for index in train_idx], labels[train_idx])
        predictions = model.predict([texts[index] for index in val_idx])
        folds.append(fold_metrics(labels[val_idx], predictions, known_labels))
        progress.advance(candidate_index, fold_index, config.cv_folds)
    return candidate_cv_result(candidate, folds)


def fold_metrics(truth, predictions, known_labels: list[str]) -> dict[str, float]:
    """Return known-class metrics for one CV fold."""

    return {
        "known_accuracy": float(accuracy_score(truth, predictions)),
        "known_balanced_accuracy": label_balanced_accuracy(truth, predictions, known_labels),
        "known_macro_f1": float(
            f1_score(truth, predictions, labels=known_labels, average="macro")
        ),
    }


def candidate_cv_result(candidate: CandidateConfig, folds: list[dict[str, float]]) -> dict[str, Any]:
    """Aggregate fold metrics for one candidate."""

    return {
        "candidate": candidate_to_dict(candidate),
        "folds": folds,
        "mean_known_accuracy": mean_metric(folds, "known_accuracy"),
        "mean_known_balanced_accuracy": mean_metric(folds, "known_balanced_accuracy"),
        "mean_known_macro_f1": mean_metric(folds, "known_macro_f1"),
    }


def label_balanced_accuracy(truth, predictions, labels: list[str]) -> float:
    """Return macro-average recall over known labels."""

    recalls = []
    for label in labels:
        mask = truth == label
        recalls.append(float(np.mean(predictions[mask] == label)) if np.any(mask) else 0.0)
    return float(np.mean(recalls)) if recalls else 0.0


def cv_selection_key(result: dict[str, Any]) -> tuple[float, float, float]:
    """Rank known-class CV candidates before OOD threshold tuning."""

    return (
        result["mean_known_balanced_accuracy"],
        result["mean_known_macro_f1"],
        result["mean_known_accuracy"],
    )


def mean_metric(folds: list[dict[str, float]], key: str) -> float:
    """Return the mean of one fold metric."""

    return float(np.mean([fold[key] for fold in folds]))
