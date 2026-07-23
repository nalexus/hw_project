"""Load provided documents and apply reviewed duplicate exclusions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

from src.model.data_preparation.text import TextProcessor


@dataclass(frozen=True)
class DocumentRecord:
    """One provided document with its class, source file, and length bucket."""

    label: str
    file_name: str
    text: str
    word_count: int
    length_bucket: str

    @classmethod
    def from_text(
        cls,
        label: str,
        file_name: str,
        text: str,
        text_processor: TextProcessor,
    ) -> "DocumentRecord":
        """Create one record and derive its shared token-length metadata."""

        count = text_processor.word_count(text)
        return cls(
            label=label,
            file_name=file_name,
            text=text,
            word_count=count,
            length_bucket=text_processor.length_bucket_for_count(count),
        )

    def as_row(self) -> dict[str, str]:
        """Return the notebook-compatible row used by tuning and evaluation."""

        return {
            "class": self.label,
            "file_name": self.file_name,
            "text": self.text,
            "bucket": self.length_bucket,
        }


@dataclass(frozen=True)
class DatasetBundle:
    """Known documents for splitting and untouched documents from other."""

    known: list[DocumentRecord]
    other: list[DocumentRecord]


class DatasetLoader:
    """Read provided dataset files while excluding reviewed duplicate records."""

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        other_label: str,
        exclusions_config_path: str | Path,
        text_processor: TextProcessor,
    ) -> None:
        """Store configured OOD, exclusion, and text-processing dependencies."""

        self.dataset_dir = Path(dataset_dir)
        self.other_label = other_label
        self.exclusions_config_path = Path(exclusions_config_path)
        self.text_processor = text_processor

    def load(self) -> DatasetBundle:
        """Return filtered known records and untouched other records."""

        excluded_paths = self._excluded_paths()
        known: list[DocumentRecord] = []
        other: list[DocumentRecord] = []

        for class_dir in sorted(self.dataset_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            label = class_dir.name.lower()
            for text_file in sorted(class_dir.glob("*.txt")):
                relative_path = text_file.relative_to(self.dataset_dir).as_posix()
                # Keep other untouched for OOD evaluation and smoke tests.
                if label != self.other_label and relative_path in excluded_paths:
                    continue

                record = DocumentRecord.from_text(
                    label=label,
                    file_name=text_file.name,
                    text=text_file.read_text(encoding="utf-8", errors="ignore"),
                    text_processor=self.text_processor,
                )
                (other if label == self.other_label else known).append(record)

        return DatasetBundle(known=known, other=other)

    def _excluded_paths(self) -> frozenset[str]:
        """Load and validate the dataset-relative duplicate exclusions."""

        with self.exclusions_config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        paths = config.get("excluded_files", [])
        if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
            raise ValueError("excluded_files must be a YAML list of dataset-relative paths.")

        normalized_paths = []
        for value in paths:
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts or len(path.parts) != 2:
                raise ValueError(f"Invalid dataset-relative exclusion path: {value}")
            normalized_paths.append(path.as_posix())
        return frozenset(normalized_paths)
