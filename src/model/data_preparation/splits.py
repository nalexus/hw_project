"""Deterministic split construction for clean training runs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import hashlib
import random
from typing import Any

from src.model.data_preparation.text import normalized_text
from src.model.train.constants import OTHER_LABEL
from src.model.train.schemas import DatasetBundle, DocumentRecord, ExperimentSplits
from src.model.train.validators import TrainingConfig


class SplitBuilder:
    """Create train, validation, test, and final provided-other splits."""

    def __init__(self, config: TrainingConfig) -> None:
        """Store split configuration."""

        self.config = config

    def build(self, bundle: DatasetBundle) -> ExperimentSplits:
        """Assign provided and synthetic records to deterministic splits."""

        provided = self._split_provided_known(bundle.provided_known)
        synthetic_known = group_by_split(bundle.synthetic_known)
        synthetic_ood = group_by_split(bundle.synthetic_ood)
        splits = ExperimentSplits(
            train=[*provided["train"], *synthetic_known["train"]],
            validation=[*provided["validation"], *synthetic_known["validation"], *synthetic_ood["validation"]],
            test=[*provided["test"], *synthetic_known["test"], *synthetic_ood["test"]],
            provided_other=[replace(record, split="provided_other_final") for record in bundle.provided_other],
        )
        assert_split_policy(splits)
        return splits

    def _split_provided_known(self, records: list[DocumentRecord]) -> dict[str, list[DocumentRecord]]:
        """Split provided known records by label and length bucket."""

        groups: dict[tuple[str, str], list[DocumentRecord]] = defaultdict(list)
        for record in records:
            groups[(record.label, record.length_bucket)].append(record)
        splits = {"train": [], "validation": [], "test": []}
        for group_key, group_records in sorted(groups.items()):
            units = self._stable_shuffle(group_key, duplicate_units(group_records))
            split_names = self._split_names(len(units))
            for split_name, unit in zip(split_names, units):
                splits[split_name].extend(replace(record, split=split_name) for record in unit)
        return {name: sorted(items, key=lambda item: item.record_id) for name, items in splits.items()}

    def _split_names(self, count: int) -> list[str]:
        """Return split names for one stratum."""

        train_count, val_count, test_count = split_counts(
            count, self.config.validation_size, self.config.test_size
        )
        return ["train"] * train_count + ["validation"] * val_count + ["test"] * test_count

    def _stable_shuffle(self, group_key: tuple[str, str], items: list[Any]) -> list[Any]:
        """Shuffle a stratum reproducibly without Python hash randomness."""

        key = f"{self.config.random_state}:{group_key[0]}:{group_key[1]}"
        seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
        shuffled = list(items)
        random.Random(seed).shuffle(shuffled)
        return shuffled


def split_counts(count: int, validation_size: float, test_size: float) -> tuple[int, int, int]:
    """Return train, validation, and test counts for one stratum."""

    if count <= 1:
        return count, 0, 0
    if count == 2:
        return 1, 0, 1
    val_count = max(1, round(count * validation_size))
    test_count = max(1, round(count * test_size))
    while val_count + test_count >= count:
        if val_count >= test_count and val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break
    return count - val_count - test_count, val_count, test_count


def duplicate_units(records: list[DocumentRecord]) -> list[list[DocumentRecord]]:
    """Group exact duplicates so they stay inside one split."""

    by_text: dict[str, list[DocumentRecord]] = defaultdict(list)
    for record in sorted(records, key=lambda item: item.record_id):
        by_text[normalized_text(record.text)].append(record)
    return [by_text[key] for key in sorted(by_text)]


def group_by_split(records: list[DocumentRecord]) -> dict[str, list[DocumentRecord]]:
    """Group synthetic records by fixed split names."""

    grouped: dict[str, list[DocumentRecord]] = {"train": [], "validation": [], "test": []}
    for record in records:
        if record.split not in grouped:
            raise ValueError(f"Unsupported synthetic split: {record.split}")
        grouped[record.split].append(record)
    return grouped


def assert_split_policy(splits: ExperimentSplits) -> None:
    """Ensure OOD examples are never in the training split."""

    if any(record.expected_label == OTHER_LABEL for record in splits.train):
        raise ValueError("Training split contains OOD rows.")
