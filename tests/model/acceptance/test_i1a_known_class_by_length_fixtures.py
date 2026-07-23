"""Integrity tests for I.1.A known-class length-bucket fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from src.model.data_preparation.loader import DatasetLoader
from src.model.data_preparation.text import TextProcessor
from src.model.train.validators import TrainingConfigModel


FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "i1a_known_class_by_length"
    / "fixtures.jsonl"
)
KNOWN_LABELS = {
    "business",
    "entertainment",
    "food",
    "graphics",
    "historical",
    "medical",
    "politics",
    "space",
    "sport",
    "technologie",
}
LENGTH_BUCKETS = {"ultra_short", "short", "medium", "long", "extra_long"}
REQUIRED_FIELDS = {
    "record_id",
    "expected_label",
    "length_bucket",
    "word_count",
    "generation_model",
    "prompt_version",
    "generation_seed",
    "text",
}
NEAR_DUPLICATE_JACCARD = 0.92
TRAIN_CONFIG_PATH = Path("config/model/train.yaml")


def load_fixture_rows() -> list[dict]:
    """Load JSONL fixture rows from the committed test data file."""

    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized_text(text: str, text_processor: TextProcessor) -> str:
    """Normalize fixture text for exact duplicate detection."""

    return " ".join(token.lower() for token in text_processor.tokenize(text))


def has_literal_label(
    text: str,
    expected_label: str,
    text_processor: TextProcessor,
) -> bool:
    """Return whether the expected label appears as a standalone token."""

    return expected_label.lower() in {
        token.lower() for token in text_processor.tokenize(text)
    }


def loaded_record_texts(text_processor: TextProcessor) -> list[tuple[str, str]]:
    """Load filtered provided records that fixtures must not copy."""

    config = TrainingConfigModel.from_yaml(TRAIN_CONFIG_PATH)
    bundle = DatasetLoader(
        config.dataset_dir,
        other_label=config.other_label,
        exclusions_config_path=config.exclusions_config_path,
        text_processor=text_processor,
    ).load()
    records = [*bundle.known, *bundle.other]
    return [(f"{record.label}/{record.file_name}", record.text) for record in records]


def shingle_set(
    text: str,
    text_processor: TextProcessor,
    size: int = 5,
) -> set[tuple[str, ...]]:
    """Build token shingles for fixture-to-dataset near-copy checks."""

    tokens = [token.lower() for token in text_processor.tokenize(text)]
    return {tuple(tokens[index:index + size]) for index in range(len(tokens) - size + 1)}


def jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    """Return Jaccard similarity for two fixture shingle sets."""

    return len(left & right) / len(left | right) if left or right else 0.0


def test_fixture_structure():
    """Verify the committed I.1.A fixture file is complete and well-formed."""

    rows = load_fixture_rows()
    text_processor = configured_text_processor()
    row_keys = {(row["expected_label"], row["length_bucket"]) for row in rows}
    expected_keys = {
        (label, bucket) for label in KNOWN_LABELS for bucket in LENGTH_BUCKETS
    }
    normalized_texts = [
        normalized_text(row["text"], text_processor) for row in rows
    ]

    assert len(rows) == len(expected_keys)
    assert row_keys == expected_keys
    assert len(normalized_texts) == len(set(normalized_texts))
    for row in rows:
        assert REQUIRED_FIELDS <= set(row)
        assert row["record_id"] == f"i1a:{row['expected_label']}:{row['length_bucket']}"
        assert row["expected_label"] in KNOWN_LABELS
        assert row["length_bucket"] in LENGTH_BUCKETS
        assert row["text"].strip()
        assert text_processor.word_count(row["text"]) == row["word_count"]
        assert text_processor.length_bucket(row["text"]) == row["length_bucket"]
        assert not has_literal_label(
            row["text"], row["expected_label"], text_processor
        )


def test_fixture_texts_do_not_copy_loaded_records():
    """Verify generated fixtures are not exact or near copies of loaded records."""

    rows = load_fixture_rows()
    text_processor = configured_text_processor()
    records = loaded_record_texts(text_processor)
    record_texts = {
        normalized_text(text, text_processor): record_id for record_id, text in records
    }
    record_shingles = [
        (record_id, shingle_set(text.lower(), text_processor))
        for record_id, text in records
    ]
    exact_matches = [
        (row["record_id"], record_texts[normalized_text(row["text"], text_processor)])
        for row in rows
        if normalized_text(row["text"], text_processor) in record_texts
    ]
    near_matches = [
        (row["record_id"], record_id)
        for row in rows
        for record_id, shingles in record_shingles
        if jaccard(
            shingle_set(row["text"].lower(), text_processor), shingles
        ) >= NEAR_DUPLICATE_JACCARD
    ]

    assert exact_matches == []
    assert near_matches == []


def configured_text_processor() -> TextProcessor:
    """Build the fixture processor from the same YAML used by training."""

    return TextProcessor.from_config(
        TrainingConfigModel.from_yaml(TRAIN_CONFIG_PATH).text_config_path
    )
