"""Integration coverage for the notebook-derived TF-IDF OOD workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.model.data_preparation.loader import DatasetLoader
from src.model.data_preparation.splits import SplitBuilder
from src.model.data_preparation.text import TextProcessor
from src.model.predict import PredictorMultiClass, TfidfOODPolicy
from src.model.train.trainer import ModelFactory
from src.model.train.validators import TrainingConfigModel
from src.model.tune.runner import TuningWorkflow, build_parser
from src.model.tune.validators import TuneConfigModel


MODEL_CONFIG = {
    "pipeline_id": "test_unigram",
    "max_features": 100,
    "ngram_range": (1, 1),
    "min_df": 1,
    "C": 1.0,
    "class_weight": None,
}
TRAIN_CONFIG_PATH = Path("config/model/train.yaml")


def test_runtime_predictor_uses_tfidf_confidence_and_oov_policy():
    """Verify runtime prediction accepts known vocabulary and rejects unseen text."""

    model = ModelFactory().build_tfidf_logreg(MODEL_CONFIG)
    model.fit(
        ["recipe ingredients oven", "football team match"],
        ["food", "sport"],
    )
    policy = TfidfOODPolicy(
        probability_threshold=0.0,
        max_oov_ratio=0.0,
        other_label="other",
    )

    prediction_rows = PredictorMultiClass(
        model,
        threshold_policy=policy.to_runtime_config("model.joblib"),
    ).predict(pd.DataFrame({
        "class": ["food", "other"],
        "bucket": ["medium", "short"],
        "text": ["recipe ingredients", "unseen vocabulary tokens"],
    }))

    assert prediction_rows["predicted_label"].tolist() == ["food", policy.other_label]
    assert prediction_rows["raw_correct"].tolist() == [True, False]
    assert prediction_rows["accepted_correct"].tolist() == [True, False]


def test_split_builder_keeps_other_out_of_train_and_test():
    """Verify untouched other rows never enter known-class model selection."""

    config = TrainingConfigModel.from_yaml(TRAIN_CONFIG_PATH)
    bundle = DatasetLoader(
        config.dataset_dir,
        other_label=config.other_label,
        exclusions_config_path=config.exclusions_config_path,
        text_processor=TextProcessor.from_config(config.text_config_path),
    ).load()
    splits = SplitBuilder().build(bundle)

    assert splits.other == bundle.other
    assert all(
        record.label != config.other_label for record in [*splits.train, *splits.test]
    )


def test_split_builder_is_deterministic_for_fixed_seed():
    """Verify class-only split IDs are reproducible for the configured seed."""

    config = TrainingConfigModel.from_yaml(TRAIN_CONFIG_PATH)
    bundle = DatasetLoader(
        config.dataset_dir,
        other_label=config.other_label,
        exclusions_config_path=config.exclusions_config_path,
        text_processor=TextProcessor.from_config(config.text_config_path),
    ).load()

    assert SplitBuilder().build(bundle).split_ids() == SplitBuilder().build(bundle).split_ids()


def test_yaml_defaults_and_cli_overrides_load_current_workflow_config():
    """Verify train and OOD-policy settings load from YAML and explicit flags."""

    defaults = TrainingConfigModel.from_yaml(TRAIN_CONFIG_PATH)
    args = build_parser().parse_args(
        [
            "--candidate-limit",
            "1",
            "--cv-folds",
            "2",
            "--support-tail-rate",
            "0.05",
        ]
    )
    overridden_train = TrainingConfigModel.from_yaml(args.train_config, vars(args))
    overridden_tune = TuneConfigModel.from_yaml(args.tune_config, vars(args))

    assert defaults.dataset_dir == (Path.cwd() / "data" / "dataset")
    assert defaults.other_label == "other"
    assert defaults.test_frac == 0.15
    assert overridden_train.candidate_limit == 1
    assert overridden_train.cv_folds == 2
    assert overridden_tune.support_tail_rate == 0.05


def test_tune_runner_smoke_writes_current_artifacts(tmp_path):
    """Run the complete CV, OOF-policy, evaluation, and persistence workflow."""

    dataset_dir = build_tiny_dataset(tmp_path / "dataset")
    result = TuningWorkflow(
        training_config(tmp_path / "runs", dataset_dir, run_name="smoke"),
        smoke_tune_config(),
    ).run()
    run_dir = result.run_dir

    assert run_dir.name == "smoke"
    assert (run_dir / "model.joblib").exists()
    assert (run_dir / "runtime_config.json").exists()
    assert (run_dir / "candidate_cv_results.json").exists()
    assert (run_dir / "oof_policy_metrics.json").exists()
    assert (run_dir / "metrics_test.json").exists()
    assert (run_dir / "metrics_other.json").exists()
    assert (run_dir / "splits.json").exists()
    runtime_config = read_json(run_dir / "runtime_config.json")
    assert runtime_config["policy"] == "tfidf_ood"
    assert runtime_config["other_label"] == "other"


def test_tune_runner_promotes_one_run_and_demotes_the_previous_one(tmp_path):
    """Verify promotion keeps exactly one API-serving run directory."""

    dataset_dir = build_tiny_dataset(tmp_path / "dataset")
    runs_dir = tmp_path / "runs"
    old_prod = runs_dir / "old_run_PROD"
    old_prod.mkdir(parents=True)
    result = TuningWorkflow(
        training_config(runs_dir, dataset_dir, run_name="promote", promote=True),
        smoke_tune_config(),
    ).run()
    run_dir = result.run_dir

    prod_dirs = [path for path in runs_dir.iterdir() if "_PROD" in path.name]
    assert prod_dirs == [run_dir]
    assert (run_dir / "model.joblib").exists()
    assert (runs_dir / "old_run").is_dir()


def training_config(
    runs_dir: Path,
    dataset_dir: Path,
    run_name: str,
    promote: bool = False,
):
    """Build a small but valid training configuration for integration tests."""

    return TrainingConfigModel.from_yaml(
        TRAIN_CONFIG_PATH,
        overrides={
            "dataset_dir": dataset_dir,
            "runs_dir": runs_dir,
            "run_name": run_name,
            "candidate_limit": 1,
            "cv_folds": 2,
            "promote_selected": promote,
        }
    )


def smoke_tune_config() -> TuneConfigModel:
    """Relax coverage only enough for the tiny separable smoke dataset."""

    return TuneConfigModel(
        support_tail_rate=0.01,
        target_accepted_error=0.50,
        minimum_correct_coverage=0.50,
    )


def build_tiny_dataset(dataset_dir: Path) -> Path:
    """Create a small separable known-class dataset plus untouched other rows."""

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
    return dataset_dir


def read_json(path: Path) -> dict:
    """Read one JSON test artifact."""

    return json.loads(path.read_text(encoding="utf-8"))
