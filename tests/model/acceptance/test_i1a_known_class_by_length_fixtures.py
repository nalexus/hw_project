"""Integrity tests for I.1.A known-class length-bucket fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from src.model.data_preparation.loader import DatasetLoader
from src.model.data_preparation.text import jaccard, shingle_set
from src.model.predict.length import TOKEN_RE, length_bucket, word_count
from src.model.train.validators import DEFAULT_DATASET_DIR


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


def load_fixture_rows() -> list[dict]:
    """Load JSONL fixture rows from the committed test data file."""

    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized_text(text: str) -> str:
    """Normalize fixture text for exact duplicate detection."""

    return " ".join(token.lower() for token in TOKEN_RE.findall(text))


def has_literal_label(text: str, expected_label: str) -> bool:
    """Return whether the expected label appears as a standalone token."""

    return expected_label.lower() in {
        token.lower() for token in TOKEN_RE.findall(text)
    }


def loaded_record_texts() -> list[tuple[str, str]]:
    """Load all project records that fixtures must not copy."""

    bundle = DatasetLoader(DEFAULT_DATASET_DIR, include_synthetic=True).load()
    records = (
        bundle.provided_known
        + bundle.provided_other
        + bundle.synthetic_known
        + bundle.synthetic_ood
    )
    return [(record.record_id, record.text) for record in records]


def test_fixture_structure():
    """Verify the committed I.1.A fixture file is complete and well-formed."""

    rows = load_fixture_rows()
    row_keys = {(row["expected_label"], row["length_bucket"]) for row in rows}
    expected_keys = {
        (label, bucket) for label in KNOWN_LABELS for bucket in LENGTH_BUCKETS
    }
    normalized_texts = [normalized_text(row["text"]) for row in rows]

    assert len(rows) == len(expected_keys)
    assert row_keys == expected_keys
    assert len(normalized_texts) == len(set(normalized_texts))
    for row in rows:
        assert REQUIRED_FIELDS <= set(row)
        assert row["record_id"] == f"i1a:{row['expected_label']}:{row['length_bucket']}"
        assert row["expected_label"] in KNOWN_LABELS
        assert row["length_bucket"] in LENGTH_BUCKETS
        assert row["text"].strip()
        assert word_count(row["text"]) == row["word_count"]
        assert length_bucket(row["text"]) == row["length_bucket"]
        assert not has_literal_label(row["text"], row["expected_label"])


def test_fixture_texts_do_not_copy_loaded_records():
    """Verify generated fixtures are not exact or near copies of loaded records."""

    rows = load_fixture_rows()
    records = loaded_record_texts()
    record_texts = {normalized_text(text): record_id for record_id, text in records}
    record_shingles = [
        (record_id, shingle_set(text.lower())) for record_id, text in records
    ]
    exact_matches = [
        (row["record_id"], record_texts[normalized_text(row["text"])])
        for row in rows
        if normalized_text(row["text"]) in record_texts
    ]
    near_matches = [
        (row["record_id"], record_id)
        for row in rows
        for record_id, shingles in record_shingles
        if jaccard(shingle_set(row["text"].lower()), shingles) >= NEAR_DUPLICATE_JACCARD
    ]

    assert exact_matches == []
    assert near_matches == []
