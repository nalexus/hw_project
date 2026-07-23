"""Deterministic class-only train/test splitting for provided documents."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from zlib import crc32

import numpy as np

from src.model.data_preparation.loader import DatasetBundle, DocumentRecord


@dataclass(frozen=True)
class DatasetSplits:
    """Known train/test records plus untouched other records for OOD checks."""

    train: list[DocumentRecord]
    test: list[DocumentRecord]
    other: list[DocumentRecord]

    def split_ids(self) -> dict[str, list[str]]:
        """Return stable class/file identifiers for each dataset partition."""

        return {
            "train": [record_id(record) for record in self.train],
            "test": [record_id(record) for record in self.test],
            "other": [record_id(record) for record in self.other],
        }


class SplitBuilder:
    """Create the notebook's reproducible class-only train/test split."""

    def __init__(self, test_frac: float = 0.15, random_state: int = 42) -> None:
        """Store and validate the test proportion and base random seed."""

        if not 0 < test_frac < 1:
            raise ValueError("test_frac must be between 0 and 1.")
        self.test_frac = test_frac
        self.random_state = random_state

    def build(self, bundle: DatasetBundle) -> DatasetSplits:
        """Split each known class independently and retain other unchanged."""

        by_label: dict[str, list[DocumentRecord]] = defaultdict(list)
        for record in bundle.known:
            by_label[record.label].append(record)

        train: list[DocumentRecord] = []
        test: list[DocumentRecord] = []
        for label, records in sorted(by_label.items()):
            class_train, class_test = self._split_one_class(label, records)
            train.extend(class_train)
            test.extend(class_test)

        return DatasetSplits(train=train, test=test, other=list(bundle.other))

    def _split_one_class(
        self,
        label: str,
        records: list[DocumentRecord],
    ) -> tuple[list[DocumentRecord], list[DocumentRecord]]:
        """Shuffle one class with its own stable seed and reserve test rows."""

        if len(records) < 2:
            raise ValueError(f"Class {label!r} needs at least two records to split.")

        seed = (self.random_state + crc32(label.encode())) % 2**32
        shuffled = [records[index] for index in np.random.RandomState(seed).permutation(len(records))]
        test_count = min(max(1, round(len(shuffled) * self.test_frac)), len(shuffled) - 1)
        return shuffled[test_count:], shuffled[:test_count]


def record_id(record: DocumentRecord) -> str:
    """Return one stable dataset-relative identifier for a document record."""

    return f"{record.label}/{record.file_name}"
