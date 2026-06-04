from pathlib import Path

import numpy as np

from src.model.data_preparation.loader import DatasetLoader
from src.model.data_preparation.splits import SplitBuilder
from src.model.experiment.manifest import runtime_config, selection_summary_lines
from src.model.experiment.runner import (
    build_config,
    build_parser,
    run_training,
    tune_threshold_policy,
)
from src.model.predict import PredictorMultiClass
from src.model.predict.validators import load_predict_config
from src.model.evaluate.predictions import rows_with_final_predictions
from src.model.train.validators import DEFAULT_DATASET_DIR, load_training_config
from src.model.train.constants import OTHER_LABEL
from src.model.train.schemas import CandidateConfig, FittedCandidate
from src.model.tune.thresholds import tune_length_bucket_thresholds


class FixedProbabilityPipeline:
    """Small predictor test double with fixed class probabilities."""

    classes_ = np.array(["food", "sport"])

    def __init__(self, probabilities):
        """Store one probability row reused for all inputs."""

        self.probabilities = np.array(probabilities)

    def predict_proba(self, texts):
        """Return the fixed probability row for every text."""

        return np.tile(self.probabilities, (len(texts), 1))


def test_clean_predictor_applies_length_bucket_policy():
    """Verify clean predictor applies the selected length-bucket policy."""

    policy = {
        "policy": "length_bucket",
        "default_threshold": 0.50,
        "bucket_thresholds": {"ultra_short": 0.70, "medium": 0.50},
    }
    texts = np.array(["tiny note", " ".join(["word"] * 130)], dtype=object)
    pipeline = FixedProbabilityPipeline([0.60, 0.40])

    predictions = PredictorMultiClass(pipeline=pipeline, threshold_policy=policy).predict(
        texts
    )

    assert predictions == ["other", "food"]


def test_split_builder_keeps_provided_other_out_of_model_selection():
    """Verify provided other remains final-only in the clean split flow."""

    config = build_config(
        build_parser().parse_args(["--no-synthetic", "--candidate-limit", "1"])
    )
    bundle = DatasetLoader(DEFAULT_DATASET_DIR, include_synthetic=False).load()
    splits = SplitBuilder(config).build(bundle)

    model_selection = [*splits.train, *splits.validation, *splits.test]

    assert bundle.provided_other
    assert not any(record.expected_label == OTHER_LABEL for record in splits.train)
    assert not any(record.source == "provided_other" for record in model_selection)
    assert all(record.split == "provided_other_final" for record in splits.provided_other)


def test_split_builder_is_deterministic_for_fixed_seed():
    """Verify split IDs are reproducible for the same config."""

    config = build_config(build_parser().parse_args(["--no-synthetic"]))
    bundle = DatasetLoader(DEFAULT_DATASET_DIR, include_synthetic=False).load()

    left = SplitBuilder(config).build(bundle).split_ids()
    right = SplitBuilder(config).build(bundle).split_ids()

    assert left == right


def test_length_bucket_tuning_selects_rejecting_ultra_short_policy():
    """Verify threshold tuning can favor OOD rejection by length bucket."""

    rows = [
        row("known-food", "food", "food", "ultra_short", 0.80),
        row("known-sport", "sport", "sport", "medium", 0.80),
        row("ood-note", "other", "food", "ultra_short", 0.55),
    ]

    tuning = tune_length_bucket_thresholds(rows, [0.0, 0.5, 0.6], ["food", "sport"], 0.5)
    policy = tuning["selected"]["policy"]
    predictions = rows_with_final_predictions(rows, policy)

    assert policy["name"] == "length_bucket"
    assert policy["params"]["bucket_thresholds"]["ultra_short"] == 0.6
    assert [item["predicted_label"] for item in predictions] == ["food", "sport", "other"]


def test_runtime_config_shape_matches_api_expectations():
    """Verify artifact runtime config exposes length-bucket serving fields."""

    selected = FittedCandidate(
        candidate=CandidateConfig("c", {}, {}),
        cv_result={},
        model=None,
        threshold_tuning={
            "selected": {
                "policy": {
                    "name": "length_bucket",
                    "params": {
                        "default_threshold": 0.53,
                        "bucket_thresholds": {"ultra_short": 0.34},
                    },
                }
            }
        },
    )

    config = runtime_config(Path("selected_model.joblib"), selected)

    assert config["policy"] == "length_bucket"
    assert config["model_path"] == "selected_model.joblib"
    assert config["default_threshold"] == 0.53
    assert config["bucket_thresholds"]["ultra_short"] == 0.34


def test_runtime_config_global_shape_matches_api_expectations():
    """Verify artifact runtime config can expose one global serving threshold."""

    selected = FittedCandidate(
        candidate=CandidateConfig("c", {}, {}),
        cv_result={},
        model=None,
        threshold_tuning={
            "selected": {
                "policy": {
                    "name": "global",
                    "params": {"threshold": 0.42},
                }
            }
        },
    )

    config = runtime_config(Path("selected_model.joblib"), selected)

    assert config["policy"] == "global"
    assert config["model_path"] == "selected_model.joblib"
    assert config["threshold"] == 0.42
    assert "bucket_thresholds" not in config


def test_yaml_training_config_loads_defaults():
    """Verify the clean package loads default train+tune YAML files."""

    config = load_training_config()

    assert config.dataset_dir == DEFAULT_DATASET_DIR
    assert config.runs_dir.name == "best_pipeline_search_runs"
    assert config.threshold_policy == "length_bucket"
    assert config.tune.threshold_step == 0.01
    assert config.tune.minimum_known_balanced_accuracy == 0.8


def test_cli_overrides_yaml_training_values():
    """Verify explicit CLI flags override YAML defaults."""

    config = build_config(
        build_parser().parse_args(["--candidate-limit", "1", "--threshold-step", "0.5"])
    )

    assert config.candidate_limit == 1
    assert config.tune.threshold_step == 0.5


def test_cli_can_select_global_threshold_policy():
    """Verify CLI overrides can select the global threshold policy."""

    config = build_config(
        build_parser().parse_args(["--threshold-policy", "global"])
    )

    assert config.threshold_policy == "global"


def test_threshold_policy_helper_uses_global_policy():
    """Verify runner threshold selection honors the configured policy."""

    rows = [
        row("known-food", "food", "food", "ultra_short", 0.80),
        row("known-sport", "sport", "sport", "medium", 0.80),
        row("ood-note", "other", "food", "ultra_short", 0.55),
    ]
    config = build_config(
        build_parser().parse_args(
            ["--threshold-policy", "global", "--minimum-known-balanced-accuracy", "0.5"]
        )
    )

    tuning = tune_threshold_policy(rows, [0.0, 0.6], ["food", "sport"], config)

    assert tuning["selected"]["policy"]["name"] == "global"
    assert tuning["selected"]["policy"]["params"]["threshold"] == 0.6


def test_selection_summary_formats_global_threshold_table():
    """Verify selected global policy is explained as one threshold table."""

    selected = FittedCandidate(
        candidate=CandidateConfig("c_global", {}, {}),
        cv_result={},
        model=None,
        threshold_tuning={
            "selected": {
                "policy": {"name": "global", "params": {"threshold": 0.42}}
            }
        },
    )
    evaluations = {
        "test": {"metrics": metrics()},
        "provided_other": {"metrics": metrics()},
    }

    lines = selection_summary_lines(selected, evaluations)

    assert lines[0] == "Selected candidate: c_global"
    assert lines[1].endswith('type = "global"')
    assert "Policy criterion:" not in "\n".join(lines)
    assert any("all input lengths  0.42" in line for line in lines)


def test_yaml_predict_config_loads_defaults():
    """Verify clean predictor defaults can be loaded from YAML."""

    config = load_predict_config()

    assert config["policy"] == "length_bucket"
    assert config["bucket_thresholds"]["medium"] == 0.53


def test_clean_runner_smoke_writes_expected_artifacts(tmp_path):
    """Run a cheap end-to-end clean training smoke test."""

    dataset_dir = tmp_path / "dataset"
    build_tiny_dataset(dataset_dir)
    args = build_parser().parse_args(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--run-name",
            "smoke",
            "--no-synthetic",
            "--candidate-limit",
            "1",
            "--top-candidates",
            "1",
            "--cv-folds",
            "2",
            "--calibration-cv-folds",
            "2",
            "--threshold-step",
            "0.5",
        ]
    )

    result = run_training(build_config(args))
    run_dir = result["run_dir"]

    assert run_dir.name.startswith("smoke_")
    assert (run_dir / f"best_pipeline_{run_dir.name}.joblib").exists()
    assert (run_dir / "runtime_config.yaml").exists()
    assert (run_dir / "runtime_config.json").exists()
    assert (run_dir / "metrics_test.json").exists()
    assert (run_dir / "predictions_provided_other.jsonl").exists()


def test_clean_runner_can_promote_selected_run(tmp_path):
    """Verify promotion marks exactly one run directory as PROD."""

    dataset_dir = tmp_path / "dataset"
    build_tiny_dataset(dataset_dir)
    old_prod = tmp_path / "runs" / "old_run_20260101T000000Z_PROD"
    old_prod.mkdir(parents=True)
    (old_prod / "best_pipeline_old_run_20260101T000000Z_PROD.joblib").write_text(
        "old", encoding="utf-8"
    )
    (old_prod / "runtime_config.json").write_text(
        '{"model_path": "best_pipeline_old_run_20260101T000000Z_PROD.joblib"}',
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--run-name",
            "promote",
            "--no-synthetic",
            "--candidate-limit",
            "1",
            "--top-candidates",
            "1",
            "--cv-folds",
            "2",
            "--calibration-cv-folds",
            "2",
            "--threshold-step",
            "0.5",
            "--promote-selected",
        ]
    )

    result = run_training(build_config(args))
    run_dir = result["run_dir"]

    prod_dirs = [path for path in (tmp_path / "runs").iterdir() if "_PROD" in path.name]
    assert prod_dirs == [run_dir]
    assert run_dir.name.endswith("_PROD")
    assert (run_dir / f"best_pipeline_{run_dir.name}.joblib").exists()
    assert (tmp_path / "runs" / "old_run_20260101T000000Z").exists()


def row(record_id, expected, raw, bucket, probability):
    """Build one controlled prediction row."""

    return {
        "record_id": record_id,
        "expected_label": expected,
        "raw_label": raw,
        "length_bucket": bucket,
        "top_probability": probability,
        "source": "synthetic_ood" if expected == OTHER_LABEL else "provided_known",
    }


def metrics():
    """Build minimal metric fields used by selection summary formatting."""

    return {
        "known_balanced_accuracy": 0.9,
        "known_accuracy": 0.8,
        "ood_accuracy": 0.7,
        "overall_accuracy": 0.6,
    }


def build_tiny_dataset(dataset_dir: Path) -> None:
    """Create a tiny separable dataset for runner smoke tests."""

    examples = {
        "business": [
            "market profit revenue capital equity margin finance banking",
            "company shares investors earnings merger dividend cash",
            "retailer sales forecast costs supplier contract loan",
            "bank deposits credit losses income lending business",
            "startup funding venture capital budget hiring growth",
            "manufacturer orders prices margins profit guidance",
        ],
        "sport": [
            "tennis match serve court racket tournament points",
            "football team scored goal league coach season",
            "basketball players rebounds assists defense playoff",
            "runner training race medal stadium sprint finish",
            "champion won final ranking points match victory",
            "coach tactics defense attack tournament athletes",
        ],
        "other": [
            "drawer envelope hallway lamp shelf notebook",
            "chair folder carpet mailbox curtain receipt",
        ],
    }
    for label, texts in examples.items():
        label_dir = dataset_dir / label
        label_dir.mkdir(parents=True)
        for index, text in enumerate(texts, start=1):
            (label_dir / f"{label}_{index}.txt").write_text(text, encoding="utf-8")
