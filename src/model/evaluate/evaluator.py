"""Evaluate the baseline model and its TF-IDF OOD policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.model.data_preparation.loader import DocumentRecord
from src.model.evaluate.metrics import ClassificationScorer, summarize_gate
from src.model.predict.predictor import TfidfOODPolicy, TfidfSignalExtractor


@dataclass(frozen=True)
class KnownEvaluation:
    """Known-class raw metrics plus policy coverage summaries."""

    raw_metrics: dict[str, float]
    overall: pd.DataFrame
    by_class: pd.DataFrame
    by_bucket: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready known-class metrics without document text."""

        return {
            "raw_metrics": self.raw_metrics,
            "policy_overall": self.overall.to_dict(orient="records"),
            "policy_by_class": self.by_class.to_dict(orient="records"),
            "policy_by_bucket": self.by_bucket.to_dict(orient="records"),
        }


@dataclass(frozen=True)
class OtherEvaluation:
    """OOD rejection summary and scored rows for the held-out other class."""

    rejection_rate: float
    correct_predictions: int
    num_examples: int
    accuracy: float
    results: pd.DataFrame

    def to_dict(self) -> dict[str, float | int]:
        """Return direct-other accuracy alongside its gate-level rejection rate."""

        return {
            "num_examples": self.num_examples,
            "correct_predictions": self.correct_predictions,
            "accuracy": self.accuracy,
            "ood_rejection_rate": self.rejection_rate,
        }


class ModelEvaluator:
    """Score known test rows and untouched ``other`` rows with one policy."""

    def evaluate_known(
        self,
        model: Any,
        records: list[DocumentRecord],
        policy: TfidfOODPolicy,
    ) -> KnownEvaluation:
        """Return overall, class, and bucket summaries for known test records."""

        results = self._scored_records(model, records, policy)
        labels = sorted(results["class"].unique())
        raw_metrics = ClassificationScorer(labels).score_predictions(
            results["class"], results["raw_label"]
        )
        overall = summarize_gate(results)
        overall.insert(
            overall.columns.get_loc("raw_accuracy") + 1,
            "balanced_accuracy",
            raw_metrics["balanced_accuracy"],
        )
        return KnownEvaluation(
            raw_metrics=raw_metrics,
            overall=overall,
            by_class=summarize_gate(results, "class"),
            by_bucket=summarize_gate(results, "bucket"),
        )

    def evaluate_other(
        self,
        model: Any,
        records: list[DocumentRecord],
        policy: TfidfOODPolicy,
    ) -> OtherEvaluation:
        """Return the direct-other rejection rate on untouched OOD examples."""

        results = self._scored_records(model, records, policy)
        correct = results["predicted_label"].eq(results["class"])
        return OtherEvaluation(
            rejection_rate=float((~results["accepted"]).mean()),
            correct_predictions=int(correct.sum()),
            num_examples=len(results),
            accuracy=float(correct.mean()),
            results=results,
        )

    def oof_signal_auroc(self, oof_results: pd.DataFrame) -> pd.Series:
        """Measure whether OOF confidence and vocabulary support separate errors."""

        is_error = ~oof_results["raw_label"].eq(oof_results["class"])
        return pd.Series(
            {
                "confidence_error_auroc": roc_auc_score(
                    is_error,
                    -oof_results["max_probability"],
                ),
                "oov_error_auroc": roc_auc_score(
                    is_error,
                    oof_results["oov_ratio"],
                ),
            }
        )

    def _scored_records(
        self,
        model: Any,
        records: list[DocumentRecord],
        policy: TfidfOODPolicy,
    ) -> pd.DataFrame:
        """Join metadata, model signals, and final policy assignments."""

        frame = pd.DataFrame(record.as_row() for record in records)
        signals = TfidfSignalExtractor(model).score(frame["text"])
        return policy.apply(frame.join(signals))
