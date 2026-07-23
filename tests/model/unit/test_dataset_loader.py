"""Unit tests for reviewed duplicate exclusions in the provided-data loader."""

from pathlib import Path

from src.model.data_preparation.loader import DatasetLoader
from src.model.data_preparation.text import TextProcessor
from src.model.train.validators import TrainingConfigModel


def test_loader_excludes_reviewed_known_file_and_keeps_other_untouched(tmp_path):
    """Verify YAML exclusions apply only to known-class dataset-relative paths."""

    dataset_dir = tmp_path / "dataset"
    write_text(dataset_dir / "business" / "keep.txt", "market revenue")
    write_text(dataset_dir / "business" / "exclude.txt", "duplicate revenue")
    write_text(dataset_dir / "other" / "exclude.txt", "unrelated text")
    exclusions_path = tmp_path / "excluded_files.yaml"
    exclusions_path.write_text(
        "excluded_files:\n  - business/exclude.txt\n  - other/exclude.txt\n",
        encoding="utf-8",
    )

    bundle = DatasetLoader(
        dataset_dir,
        other_label="other",
        exclusions_config_path=exclusions_path,
        text_processor=TextProcessor.from_config(
            TrainingConfigModel.from_yaml(
                Path("config/model/train.yaml")
            ).text_config_path
        ),
    ).load()

    assert [record.file_name for record in bundle.known] == ["keep.txt"]
    assert [record.file_name for record in bundle.other] == ["exclude.txt"]


def write_text(path: Path, text: str) -> None:
    """Create one temporary dataset text file with its class directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
