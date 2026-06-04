"""Exact and near-duplicate leakage checks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.model.data_preparation.text import jaccard, normalized_text, shingle_set
from src.model.train.schemas import DocumentRecord, ExperimentSplits
from src.model.train.validators import TrainingConfig


class LeakageChecker:
    """Detect duplicate text crossing split boundaries."""

    def __init__(self, config: TrainingConfig) -> None:
        """Store leakage-check thresholds."""

        self.config = config

    def check(self, splits: ExperimentSplits) -> dict[str, Any]:
        """Return leakage diagnostics and fail on exact leakage."""

        records = splits.all_records()
        exact = find_exact_cross_split(records)
        near = find_near_cross_split(records, self.config.near_duplicate_jaccard)
        if exact:
            raise ValueError(f"Exact duplicate leakage detected: {exact[:5]}")
        if near and self.config.fail_on_near_duplicates:
            raise ValueError(f"Near-duplicate leakage detected: {near[:5]}")
        return {
            "exact_cross_split": exact,
            "near_cross_split": near,
            "near_duplicate_jaccard": self.config.near_duplicate_jaccard,
            "fail_on_near_duplicates": self.config.fail_on_near_duplicates,
        }


def find_exact_cross_split(records: list[DocumentRecord]) -> list[dict[str, Any]]:
    """Find normalized exact duplicates assigned to different splits."""

    by_text: dict[str, list[DocumentRecord]] = defaultdict(list)
    for record in records:
        by_text[normalized_text(record.text)].append(record)
    issues: list[dict[str, Any]] = []
    for duplicate_records in by_text.values():
        split_names = {record.split for record in duplicate_records}
        if len(split_names) > 1 and len(duplicate_records) > 1:
            issues.append(
                {
                    "record_ids": [record.record_id for record in duplicate_records],
                    "splits": sorted(str(name) for name in split_names),
                }
            )
    return issues


def find_near_cross_split(records: list[DocumentRecord], threshold: float) -> list[dict[str, Any]]:
    """Find high-Jaccard near duplicates assigned to different splits."""

    prepared = [(record, shingle_set(record.text)) for record in records]
    issues: list[dict[str, Any]] = []
    for left_index, (left, left_shingles) in enumerate(prepared):
        for right, right_shingles in prepared[left_index + 1 :]:
            if left.split == right.split:
                continue
            score = jaccard(left_shingles, right_shingles)
            if score >= threshold:
                issues.append(near_issue(left, right, score))
            if len(issues) >= 100:
                return issues
    return issues


def near_issue(left: DocumentRecord, right: DocumentRecord, score: float) -> dict[str, Any]:
    """Build one near-duplicate issue row."""

    return {
        "left_record_id": left.record_id,
        "right_record_id": right.record_id,
        "left_split": left.split,
        "right_split": right.split,
        "jaccard": round(score, 4),
    }
