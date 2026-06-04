"""Typed data structures for clean training records and results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DocumentRecord:
    """Single provided or synthetic document with reproducibility metadata."""

    record_id: str
    text: str
    label: str
    expected_label: str
    source: str
    split: str | None
    path: str | None
    text_hash: str
    word_count: int
    length_bucket: str

    def manifest_dict(self, include_text: bool = False) -> dict[str, Any]:
        """Return a JSON-safe manifest row for this record."""

        row = {
            "record_id": self.record_id,
            "label": self.label,
            "expected_label": self.expected_label,
            "source": self.source,
            "split": self.split,
            "path": self.path,
            "text_hash": self.text_hash,
            "word_count": self.word_count,
            "length_bucket": self.length_bucket,
        }
        if include_text:
            row["text"] = self.text
        return row


@dataclass(frozen=True)
class DatasetBundle:
    """Loaded source records before split assignment."""

    provided_known: list[DocumentRecord]
    provided_other: list[DocumentRecord]
    synthetic_known: list[DocumentRecord]
    synthetic_ood: list[DocumentRecord]


@dataclass(frozen=True)
class ExperimentSplits:
    """Train, validation, test, and untouched provided-other partitions."""

    train: list[DocumentRecord]
    validation: list[DocumentRecord]
    test: list[DocumentRecord]
    provided_other: list[DocumentRecord]

    def all_records(self) -> list[DocumentRecord]:
        """Return every record with assigned split information."""

        return [*self.train, *self.validation, *self.test, *self.provided_other]

    def split_ids(self) -> dict[str, list[str]]:
        """Return exact record IDs per split."""

        return {
            "train": [record.record_id for record in self.train],
            "validation": [record.record_id for record in self.validation],
            "test": [record.record_id for record in self.test],
            "provided_other_final": [record.record_id for record in self.provided_other],
        }


@dataclass(frozen=True)
class CandidateConfig:
    """TF-IDF and logistic-regression settings for one candidate."""

    candidate_id: str
    tfidf_params: dict[str, Any]
    classifier_params: dict[str, Any]


@dataclass
class FittedCandidate:
    """Fitted model with selected threshold policy and validation metrics."""

    candidate: CandidateConfig
    cv_result: dict[str, Any]
    model: Any
    threshold_tuning: dict[str, Any]
