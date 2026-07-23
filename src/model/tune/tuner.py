"""Notebook-style candidate and OOD-policy tuning for the TF-IDF baseline."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

from src.model.evaluate.metrics import ClassificationScorer
from src.model.predict.predictor import TfidfOODPolicy, TfidfSignalExtractor
from src.model.train.trainer import ModelFactory


class CandidateCatalog:
    """Load the small, explicit candidate list used for model selection."""

    def __init__(self, config_path: str | Path) -> None:
        """Store the YAML location for candidate definitions."""

        self.config_path = Path(config_path)

    def build(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return candidate configs in their declared search order."""

        with self.config_path.open("r", encoding="utf-8") as handle:
            candidates = (yaml.safe_load(handle) or {})["candidates"]
        configs = [self._normalize(candidate) for candidate in candidates]
        return configs[:limit] if limit else configs

    @staticmethod
    def _normalize(candidate: dict[str, Any]) -> dict[str, Any]:
        """Convert YAML lists and scalars into sklearn-ready candidate values."""

        return {
            **candidate,
            "ngram_range": tuple(candidate["ngram_range"]),
        }


class CrossValidationTuner:
    """Select the best TF-IDF plus logistic-regression candidate by CV."""

    def __init__(
        self,
        configs: list[dict[str, Any]],
        cv_folds: int = 3,
        random_state: int = 42,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        """Store candidate definitions and the reproducible CV setup."""

        self.configs = configs
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.progress = progress
        self.factory = ModelFactory(random_state=random_state)

    def tune(self, train_df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
        """Return the best candidate and its ranked cross-validation table."""

        labels = sorted(train_df["class"].unique())
        scorer = ClassificationScorer(labels)
        splitter = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        total_fits = len(self.configs) * self.cv_folds
        rows = [
            self._score_config(
                config,
                train_df,
                splitter,
                scorer,
                candidate_index,
                total_fits,
            )
            for candidate_index, config in enumerate(self.configs)
        ]
        results = pd.DataFrame(rows).sort_values(
            ["mean_balanced_accuracy", "mean_macro_f1", "mean_accuracy"],
            ascending=False,
        ).reset_index(drop=True)
        best_config = next(
            config
            for config in self.configs
            if config["pipeline_id"] == results.loc[0, "pipeline_id"]
        )
        return best_config, results

    def _score_config(
        self,
        config: dict[str, Any],
        train_df: pd.DataFrame,
        splitter: StratifiedKFold,
        scorer: ClassificationScorer,
        candidate_index: int,
        total_fits: int,
    ) -> dict[str, Any]:
        """Score one candidate on every stratified validation fold."""

        fold_scores = []
        for fold_index, (train_idx, valid_idx) in enumerate(
            splitter.split(train_df["text"], train_df["class"]),
            start=1,
        ):
            fold_train = train_df.iloc[train_idx]
            fold_valid = train_df.iloc[valid_idx]
            model = self.factory.build_tfidf_logreg(config)
            model.fit(fold_train["text"], fold_train["class"])
            fold_scores.append(
                scorer.score_predictions(
                    fold_valid["class"], model.predict(fold_valid["text"])
                )
            )
            self._report_progress(
                "Candidate CV",
                candidate_index * self.cv_folds + fold_index,
                total_fits,
            )

        scores = pd.DataFrame(fold_scores)
        return {
            **config,
            "mean_accuracy": scores["accuracy"].mean(),
            "mean_balanced_accuracy": scores["balanced_accuracy"].mean(),
            "mean_macro_f1": scores["macro_f1"].mean(),
        }

    def _report_progress(self, stage: str, completed: int, total: int) -> None:
        """Send fit progress to the optional CLI presentation callback."""

        if self.progress is not None:
            self.progress(stage, completed, total)


class OutOfFoldSignalCollector:
    """Collect signals from models that did not train on each row."""

    def __init__(
        self,
        config: dict[str, Any],
        cv_folds: int = 3,
        random_state: int = 42,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        """Store the selected candidate and OOF cross-validation setup."""

        self.config = config
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.progress = progress
        self.factory = ModelFactory(random_state=random_state)

    def collect(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Return one confidence and vocabulary-support row per train example."""

        splitter = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        score_frames = []
        for fold_index, (fold_train_idx, fold_valid_idx) in enumerate(
            splitter.split(train_df["text"], train_df["class"]),
            start=1,
        ):
            fold_train = train_df.iloc[fold_train_idx]
            fold_valid = train_df.iloc[fold_valid_idx]
            model = self.factory.build_tfidf_logreg(self.config)
            model.fit(fold_train["text"], fold_train["class"])
            signals = TfidfSignalExtractor(model).score(fold_valid["text"])
            score_frames.append(fold_valid[["class", "bucket"]].join(signals))
            if self.progress is not None:
                self.progress("OOF signals", fold_index, self.cv_folds)

        return pd.concat(score_frames).sort_index()


class SupportAwareConfidenceTuner:
    """Set a strict confidence threshold while preserving known-data coverage."""

    def __init__(
        self,
        support_tail_rate: float = 0.01,
        target_accepted_error: float = 0.02,
        minimum_correct_coverage: float = 0.95,
    ) -> None:
        """Store the supported-vocabulary, risk, and coverage constraints."""

        self.support_tail_rate = support_tail_rate
        self.target_accepted_error = target_accepted_error
        self.minimum_correct_coverage = minimum_correct_coverage

    def tune(self, oof_results: pd.DataFrame, other_label: str) -> TfidfOODPolicy:
        """Choose the strictest OOF policy for the configured OOD label."""

        max_oov_ratio = oof_results["oov_ratio"].quantile(
            1 - self.support_tail_rate,
            interpolation="higher",
        )
        supported = oof_results["oov_ratio"] <= max_oov_ratio
        correct = oof_results["raw_label"].eq(oof_results["class"])
        candidates = []

        for probability_threshold in oof_results.loc[
            supported, "max_probability"
        ].unique():
            accepted = supported & (
                oof_results["max_probability"] >= probability_threshold
            )
            accepted_error = 1 - correct.loc[accepted].mean()
            correct_coverage = (accepted & correct).sum() / correct.sum()
            if (
                accepted_error <= self.target_accepted_error
                and correct_coverage >= self.minimum_correct_coverage
            ):
                candidates.append(
                    (probability_threshold, accepted_error, correct_coverage)
                )

        if not candidates:
            raise ValueError("No policy met both risk and coverage requirements.")

        probability_threshold, _, _ = max(
            candidates,
            key=lambda item: (item[0], -item[1]),
        )
        return TfidfOODPolicy(
            probability_threshold=float(probability_threshold),
            max_oov_ratio=float(max_oov_ratio),
            other_label=other_label,
        )
