from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


class ClassificationScorer:
    """Compute the multiclass metrics used to select TF-IDF candidates."""

    def __init__(self, labels: list[str]) -> None:
        """Store the stable known-class label order for macro F1."""

        self.labels = labels

    def score_predictions(self, y_true, y_pred) -> dict[str, float]:
        """Compute accuracy, balanced accuracy, and macro F1."""

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=self.labels,
                    average="macro",
                    zero_division=0,
                )
            ),
        }


def summarize_gate(
    results: pd.DataFrame,
    group_column: str | None = None,
) -> pd.DataFrame:
    """Report raw accuracy, acceptance coverage, and accepted accuracy."""

    if group_column is None:
        results = results.assign(_group="overall")
        group_column = "_group"

    summary = results.groupby(group_column, observed=True).agg(
        num_examples=("class", "size"),
        accepted=("accepted", "sum"),
        raw_accuracy=("raw_correct", "mean"),
        accepted_correct=("accepted_correct", "sum"),
    )
    summary["coverage"] = summary["accepted"] / summary["num_examples"]
    summary["accepted_accuracy"] = (
        summary["accepted_correct"] / summary["accepted"].replace(0, np.nan)
    )
    return summary.drop(columns="accepted_correct").reset_index()
