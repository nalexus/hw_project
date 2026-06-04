"""Load copied provided data and curated synthetic records."""

from __future__ import annotations

from pathlib import Path

from src.model.data_preparation.text import hash_text, length_bucket_for_count, tokenize
from src.model.train.constants import (
    LENGTH_BUCKETS,
    OTHER_LABEL,
    PROVIDED_KNOWN,
    PROVIDED_OTHER,
    SYNTHETIC_KNOWN,
    SYNTHETIC_OOD,
)
from src.model.train.schemas import DatasetBundle, DocumentRecord


class DatasetLoader:
    """Load provided files and curated synthetic records from copied data."""

    def __init__(self, dataset_dir, include_synthetic: bool = True) -> None:
        """Store dataset location and synthetic-data inclusion flag."""

        self.dataset_dir = Path(dataset_dir)
        self.synthetic_dir = self.dataset_dir.parent / "synthetic_samples"
        self.include_synthetic = include_synthetic

    def load(self) -> DatasetBundle:
        """Read every configured data source into record objects."""

        provided_known, provided_other = self._load_provided_records()
        synthetic_known, synthetic_ood = self._load_synthetic_records()
        return DatasetBundle(provided_known, provided_other, synthetic_known, synthetic_ood)

    def _load_provided_records(self) -> tuple[list[DocumentRecord], list[DocumentRecord]]:
        """Load copied provided text files and separate final OOD records."""

        known: list[DocumentRecord] = []
        other: list[DocumentRecord] = []
        for category_dir in sorted(self.dataset_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            label = category_dir.name.lower()
            for text_file in sorted(category_dir.glob("*.txt")):
                text = text_file.read_text(encoding="utf-8", errors="ignore")
                relative_path = text_file.relative_to(self.dataset_dir).as_posix()
                source = PROVIDED_OTHER if label == OTHER_LABEL else PROVIDED_KNOWN
                record = build_record(
                    f"provided:{relative_path}", text, label, source, None, relative_path
                )
                (other if label == OTHER_LABEL else known).append(record)
        return known, other

    def _load_synthetic_records(self) -> tuple[list[DocumentRecord], list[DocumentRecord]]:
        """Load synthetic examples from the file-backed sample tree."""

        if not self.include_synthetic:
            return [], []
        if self.synthetic_dir.exists():
            return self._load_synthetic_file_records()
        raise FileNotFoundError(f"Synthetic samples directory not found: {self.synthetic_dir}")

    def _load_synthetic_file_records(self) -> tuple[list[DocumentRecord], list[DocumentRecord]]:
        """Load synthetic examples from class/length/split-prefixed text files."""

        known: list[DocumentRecord] = []
        ood: list[DocumentRecord] = []
        for class_dir in sorted(self.synthetic_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = class_dir.name.lower()
            for bucket_dir in sorted(class_dir.iterdir()):
                if not bucket_dir.is_dir():
                    continue
                for text_file in sorted(bucket_dir.glob("*.txt")):
                    record = self._load_synthetic_file(label, bucket_dir.name, text_file)
                    (ood if record.expected_label == OTHER_LABEL else known).append(record)
        return known, ood

    def _load_synthetic_file(
        self, label: str, length_bucket: str, text_file: Path
    ) -> DocumentRecord:
        """Build one synthetic record from a split-prefixed text file."""

        split, identifier = parse_synthetic_filename(text_file)
        source = SYNTHETIC_OOD if label == OTHER_LABEL else SYNTHETIC_KNOWN
        if source == SYNTHETIC_OOD and split == "train":
            raise ValueError(f"Synthetic OOD cannot use train split: {text_file}")
        record_id = synthetic_record_id(source, label, split, identifier)
        record = build_record(
            record_id,
            text_file.read_text(encoding="utf-8"),
            label,
            source,
            split,
            None,
            expected_label=OTHER_LABEL if source == SYNTHETIC_OOD else label,
        )
        validate_synthetic_bucket(record, length_bucket, text_file)
        return record


def parse_synthetic_filename(text_file: Path) -> tuple[str, str]:
    """Parse split and identifier from one synthetic sample filename."""

    for split in ("validation", "train", "test"):
        prefix = f"{split}_"
        if text_file.stem.startswith(prefix):
            return split, text_file.stem.removeprefix(prefix)
    raise ValueError(f"Synthetic filename must start with split prefix: {text_file}")


def synthetic_record_id(source: str, label: str, split: str, identifier: str) -> str:
    """Return the stable synthetic record ID for a file-backed example."""

    if source == SYNTHETIC_OOD:
        return f"synthetic_ood:{split}:{identifier}"
    return f"synthetic_known:{label}:{split}:{identifier}"


def validate_synthetic_bucket(
    record: DocumentRecord, configured_bucket: str, text_file: Path
) -> None:
    """Ensure directory bucket matches the token-count-derived bucket."""

    valid_buckets = {name for name, _, _ in LENGTH_BUCKETS}
    if configured_bucket not in valid_buckets:
        raise ValueError(f"Unsupported synthetic length bucket {configured_bucket}: {text_file}")
    if record.length_bucket != configured_bucket:
        raise ValueError(
            f"Synthetic length bucket mismatch for {text_file}: "
            f"directory={configured_bucket}, computed={record.length_bucket}"
        )


def build_record(
    record_id: str,
    text: str,
    label: str,
    source: str,
    split: str | None,
    path: str | None,
    expected_label: str | None = None,
) -> DocumentRecord:
    """Build a document record with reproducibility fields."""

    word_count = len(tokenize(text))
    return DocumentRecord(
        record_id=record_id,
        text=text,
        label=label,
        expected_label=expected_label or label,
        source=source,
        split=split,
        path=path,
        text_hash=hash_text(text),
        word_count=word_count,
        length_bucket=length_bucket_for_count(word_count),
    )
