import pytest

from src.model.data_preparation.loader import DatasetLoader
from src.model.train.validators import DEFAULT_DATASET_DIR


def test_file_backed_synthetic_samples_load_expected_manifest():
    """Verify file-backed synthetic samples expose stable metadata."""

    bundle = DatasetLoader(DEFAULT_DATASET_DIR, include_synthetic=True).load()

    assert len(bundle.synthetic_known) == 200
    assert len(bundle.synthetic_ood) == 168
    assert {record.label for record in bundle.synthetic_known} == {
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
    assert {record.split for record in bundle.synthetic_ood} == {"validation", "test"}
    assert all(record.expected_label == "other" for record in bundle.synthetic_ood)


def test_synthetic_loader_requires_file_tree_when_enabled(tmp_path):
    """Verify missing synthetic samples fail clearly when synthetic data is enabled."""

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Synthetic samples directory not found"):
        DatasetLoader(dataset_dir, include_synthetic=True).load()


def test_synthetic_loader_allows_missing_file_tree_when_disabled(tmp_path):
    """Verify no-synthetic loading does not require a synthetic sample tree."""

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    bundle = DatasetLoader(dataset_dir, include_synthetic=False).load()

    assert bundle.synthetic_known == []
    assert bundle.synthetic_ood == []


def test_synthetic_loader_rejects_length_bucket_mismatch(tmp_path):
    """Verify misplaced synthetic files fail before training."""

    dataset_dir = tmp_path / "dataset"
    samples_dir = tmp_path / "synthetic_samples" / "business" / "long"
    dataset_dir.mkdir()
    samples_dir.mkdir(parents=True)
    (samples_dir / "train_001.txt").write_text("short sample", encoding="utf-8")

    with pytest.raises(ValueError, match="length bucket mismatch"):
        DatasetLoader(dataset_dir, include_synthetic=True).load()


def test_synthetic_loader_rejects_train_ood_files(tmp_path):
    """Verify synthetic OOD examples cannot enter the training split."""

    dataset_dir = tmp_path / "dataset"
    samples_dir = tmp_path / "synthetic_samples" / "other" / "ultra_short"
    dataset_dir.mkdir()
    samples_dir.mkdir(parents=True)
    (samples_dir / "train_ood_bad_001.txt").write_text("random note", encoding="utf-8")

    with pytest.raises(ValueError, match="Synthetic OOD cannot use train split"):
        DatasetLoader(dataset_dir, include_synthetic=True).load()


def manifest_rows(records):
    """Return comparable synthetic record metadata without raw text."""

    rows = [
        {
            "record_id": record.record_id,
            "label": record.label,
            "expected_label": record.expected_label,
            "source": record.source,
            "split": record.split,
            "text_hash": record.text_hash,
            "word_count": record.word_count,
            "length_bucket": record.length_bucket,
        }
        for record in records
    ]
    return sorted(rows, key=lambda row: (row["record_id"], row["text_hash"]))
