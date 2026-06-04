"""Frozen sentence-transformer benchmark for the text-classifier reject policy."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace, SUPPRESS
from collections import defaultdict
import importlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.model.data_preparation.loader import DatasetLoader
from src.model.data_preparation.splits import SplitBuilder
from src.model.evaluate.metrics import metric_summary
from src.model.evaluate.predictions import (
    prediction_row,
    rows_with_final_predictions,
)
from src.model.experiment.io import write_json
from src.model.experiment.naming import utc_run_id
from src.model.experiment.writer import write_jsonl
from src.model.train.constants import OTHER_LABEL
from src.model.train.schemas import DocumentRecord, ExperimentSplits
from src.model.train.validators import (
    DEFAULT_TRAIN_CONFIG_PATH,
    TrainingConfig,
    load_training_config,
)
from src.model.tune.thresholds import threshold_values, tune_length_bucket_thresholds
from src.model.tune.validators import DEFAULT_TUNE_CONFIG_PATH


LOGGER = logging.getLogger(__name__)
BASELINE_RUN_NAME = "run_20260531T211109Z_PROD"
DEFAULT_C_VALUES = (0.3, 1.0, 3.0, 10.0)
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RUN_TIMESTAMP_RE = re.compile(r"\d{8}T\d{6}Z")


def build_parser() -> ArgumentParser:
    """Build CLI arguments for the frozen-embedding benchmark."""

    parser = ArgumentParser(description="Run a frozen sentence-transformer benchmark.")
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG_PATH)
    parser.add_argument("--tune-config", type=Path, default=DEFAULT_TUNE_CONFIG_PATH)
    parser.add_argument("--dataset-dir", type=Path, default=SUPPRESS)
    parser.add_argument("--runs-dir", type=Path, default=SUPPRESS)
    parser.add_argument("--run-name", default="transformer_addon")
    parser.add_argument("--random-state", type=int, default=SUPPRESS)
    parser.add_argument("--threshold-step", type=float, default=SUPPRESS)
    parser.add_argument("--minimum-known-balanced-accuracy", type=float, default=SUPPRESS)
    parser.add_argument("--no-synthetic", dest="include_synthetic", action="store_false", default=SUPPRESS)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Limit records per split for smoke runs.")
    parser.add_argument("--c-values", type=float, nargs="+", default=list(DEFAULT_C_VALUES))
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def build_config(args: Namespace) -> TrainingConfig:
    """Convert CLI arguments into the existing training configuration model."""

    return load_training_config(
        train_config_path=args.train_config,
        tune_config_path=args.tune_config,
        overrides=vars(args),
    )


def run_experiment(args: Namespace) -> dict[str, Any]:
    """Run the transformer benchmark and write all experiment artifacts."""

    config = build_config(args)
    splits = build_splits(config, args.limit)
    known_labels = known_labels_for_splits(splits)
    sentence_model, device = load_sentence_model(args.model_name)
    run_dir = prepare_run_dir(config.runs_dir, config.run_name)
    embeddings = encode_splits(sentence_model, splits, args.batch_size)
    selected = select_classifier(args.c_values, embeddings, splits, known_labels, config)
    evaluations = evaluate_selected(selected, embeddings, splits, known_labels)
    latency = measure_latency(sentence_model, selected["classifier"], splits, args.batch_size, device)
    baseline = baseline_comparison(config.runs_dir, evaluations)
    write_outputs(run_dir, args, selected, evaluations, splits, latency, baseline)
    print_summary(run_dir, selected, evaluations, baseline, latency)
    return {"run_dir": run_dir, "selected": selected, "evaluations": evaluations}


def prepare_run_dir(runs_dir: Path, run_name: str) -> Path:
    """Create a unique run directory for addon artifacts."""

    base_name = run_name.strip() or "transformer_addon"
    if RUN_TIMESTAMP_RE.search(base_name) is None:
        base_name = f"{base_name}_{utc_run_id()}"
    run_dir = runs_dir / base_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_splits(config: TrainingConfig, limit: int | None) -> ExperimentSplits:
    """Load and split records with the existing deterministic split policy."""

    bundle = DatasetLoader(config.dataset_dir, config.include_synthetic).load()
    splits = SplitBuilder(config).build(bundle)
    if limit is None:
        return splits
    return ExperimentSplits(
        train=limited_records(splits.train, limit),
        validation=limited_records(splits.validation, limit),
        test=limited_records(splits.test, limit),
        provided_other=limited_records(splits.provided_other, limit),
    )


def limited_records(records: list[DocumentRecord], limit: int) -> list[DocumentRecord]:
    """Return a stable round-robin subset by expected label."""

    if limit >= len(records):
        return records
    groups = grouped_by_expected_label(records)
    selected = []
    while len(selected) < limit and any(groups.values()):
        for label in sorted(groups):
            if groups[label]:
                selected.append(groups[label].pop(0))
            if len(selected) >= limit:
                break
    return sorted(selected, key=lambda record: record.record_id)


def grouped_by_expected_label(records: list[DocumentRecord]) -> dict[str, list[DocumentRecord]]:
    """Group records by expected label while preserving stable record order."""

    groups: dict[str, list[DocumentRecord]] = defaultdict(list)
    for record in sorted(records, key=lambda item: item.record_id):
        groups[record.expected_label].append(record)
    return groups


def known_labels_for_splits(splits: ExperimentSplits) -> list[str]:
    """Return sorted known labels from train, validation, and test splits."""

    records = [*splits.train, *splits.validation, *splits.test]
    return sorted({record.label for record in records if record.label != OTHER_LABEL})


def load_sentence_model(model_name: str):
    """Load the optional sentence-transformer model on CPU or GPU."""

    sentence_transformers = import_optional("sentence_transformers")
    device = detect_device()
    LOGGER.info("Loading %s on %s", model_name, device)
    return sentence_transformers.SentenceTransformer(model_name, device=device), device


def import_optional(module_name: str):
    """Import an optional transformer dependency with a focused install hint."""

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        message = (
            f"Missing optional dependency '{module_name}'. "
            "Install it with `uv add sentence-transformers` before running this benchmark."
        )
        raise RuntimeError(message) from exc


def detect_device() -> str:
    """Return cuda when torch can see a GPU, otherwise cpu."""

    torch = import_optional("torch")
    return "cuda" if torch.cuda.is_available() else "cpu"


def encode_splits(model: Any, splits: ExperimentSplits, batch_size: int) -> dict[str, np.ndarray]:
    """Encode every split once so C-search only trains lightweight classifiers."""

    encoded = {}
    for split_name, records in split_records(splits).items():
        LOGGER.info("Encoding %s records for %s", len(records), split_name)
        encoded[split_name] = encode_records(model, records, batch_size)
    return encoded


def encode_records(model: Any, records: list[DocumentRecord], batch_size: int) -> np.ndarray:
    """Return normalized sentence embeddings for a list of records."""

    if not records:
        return np.empty((0, 0), dtype=np.float32)
    texts = [record.text for record in records]
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def select_classifier(
    c_values: list[float],
    embeddings: dict[str, np.ndarray],
    splits: ExperimentSplits,
    known_labels: list[str],
    config: TrainingConfig,
) -> dict[str, Any]:
    """Train C candidates and select by tuned validation score."""

    candidates = []
    thresholds = threshold_values(config.threshold_step)
    for c_value in c_values:
        LOGGER.info("Training logistic classifier with C=%s", c_value)
        classifier = fit_classifier(embeddings["train"], splits.train, c_value, config.random_state)
        rows = raw_prediction_rows(splits.validation, classifier, embeddings["validation"])
        tuning = tune_length_bucket_thresholds(
            rows, thresholds, known_labels, config.minimum_known_balanced_accuracy
        )
        candidates.append(candidate_result(c_value, classifier, tuning))
    selected = max(candidates, key=classifier_selection_key)
    selected["all_candidate_results"] = [compact_candidate_result(item) for item in candidates]
    return selected


def fit_classifier(
    features: np.ndarray, records: list[DocumentRecord], c_value: float, random_state: int
) -> LogisticRegression:
    """Fit a balanced logistic-regression head on frozen embeddings."""

    classifier = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
    )
    classifier.fit(features, [record.label for record in records])
    return classifier


def raw_prediction_rows(
    records: list[DocumentRecord], classifier: LogisticRegression, features: np.ndarray
) -> list[dict[str, Any]]:
    """Build raw probability rows for threshold tuning and evaluation."""

    if not records:
        return []
    probabilities = classifier.predict_proba(features)
    rows = []
    for record, probs in zip(records, probabilities):
        order = np.argsort(probs)[::-1]
        rows.append(prediction_row(record, classifier.classes_, probs, order))
    return rows


def candidate_result(c_value: float, classifier: LogisticRegression, tuning: dict[str, Any]) -> dict[str, Any]:
    """Return a serializable candidate summary plus fitted classifier."""

    return {
        "c_value": c_value,
        "classifier": classifier,
        "classifier_params": classifier.get_params(),
        "threshold_tuning": tuning,
        "validation_metrics": tuning["selected"]["metrics"],
    }


def compact_candidate_result(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return a serializable C-search result without the fitted classifier."""

    return {
        "c_value": candidate["c_value"],
        "classifier_params": candidate["classifier_params"],
        "validation_metrics": candidate["validation_metrics"],
        "selected_policy": candidate["threshold_tuning"]["selected"]["policy"],
    }


def classifier_selection_key(candidate: dict[str, Any]) -> tuple[float, float, float]:
    """Rank classifier heads by tuned validation metrics."""

    metrics = candidate["validation_metrics"]
    return (
        metrics["threshold_selection_score"],
        metrics["known_balanced_accuracy"],
        metrics["ood_accuracy"],
    )


def evaluate_selected(
    selected: dict[str, Any],
    embeddings: dict[str, np.ndarray],
    splits: ExperimentSplits,
    known_labels: list[str],
) -> dict[str, dict[str, Any]]:
    """Evaluate the selected classifier and reject policy on all final splits."""

    evaluations = {}
    policy = selected["threshold_tuning"]["selected"]["policy"]
    for split_name, records in split_records(splits).items():
        if split_name == "train":
            continue
        rows = raw_prediction_rows(records, selected["classifier"], embeddings[split_name])
        predictions = rows_with_final_predictions(rows, policy)
        evaluations[split_name] = {
            "metrics": metric_summary(predictions, known_labels),
            "predictions": predictions,
        }
    return evaluations


def split_records(splits: ExperimentSplits) -> dict[str, list[DocumentRecord]]:
    """Return records by split name for encoding and evaluation."""

    return {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
        "provided_other": splits.provided_other,
    }


def measure_latency(
    model: Any,
    classifier: LogisticRegression,
    splits: ExperimentSplits,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    """Measure simple end-to-end embedding plus classifier latency."""

    records = [*splits.test, *splits.validation, *splits.train]
    if not records:
        return {}
    single_text = records[0].text
    batch_texts = [record.text for record in records[:batch_size]]
    single_seconds = time_prediction(model, classifier, [single_text], batch_size=1)
    batch_seconds = time_prediction(model, classifier, batch_texts, batch_size=batch_size)
    return {
        "device": device,
        "single_document_seconds": single_seconds,
        "batch_size": len(batch_texts),
        "batch_seconds": batch_seconds,
        "batch_seconds_per_document": batch_seconds / max(len(batch_texts), 1),
    }


def time_prediction(
    model: Any, classifier: LogisticRegression, texts: list[str], batch_size: int
) -> float:
    """Return wall-clock seconds for one encode plus predict_proba call."""

    start = time.perf_counter()
    features = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    classifier.predict_proba(features)
    return time.perf_counter() - start


def baseline_comparison(runs_dir: Path, evaluations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compare addon headline metrics with the current promoted baseline."""

    baseline = load_baseline_metadata(runs_dir)
    current = compact_headline_metrics(evaluations)
    baseline_metrics = {
        "test_known_balanced_accuracy": baseline["test_metrics"]["known_balanced_accuracy"],
        "test_ood_accuracy": baseline["test_metrics"]["ood_accuracy"],
        "provided_other_accuracy": baseline["provided_other_metrics"]["ood_accuracy"],
    }
    return {
        "baseline_run": BASELINE_RUN_NAME,
        "baseline": baseline_metrics,
        "transformer_addon": current,
        "promising": is_promising(baseline_metrics, current),
        "replacement_candidate": is_replacement_candidate(baseline_metrics, current),
        "provided_other_note": "Provided-other has only 6 examples; do not overclaim OOD robustness.",
    }


def load_baseline_metadata(runs_dir: Path) -> dict[str, Any]:
    """Load the current production baseline metadata from the prior run folder."""

    path = runs_dir / BASELINE_RUN_NAME / "metadata.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compact_headline_metrics(evaluations: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Return the metrics used for addon success criteria."""

    return {
        "test_known_balanced_accuracy": evaluations["test"]["metrics"]["known_balanced_accuracy"],
        "test_ood_accuracy": evaluations["test"]["metrics"]["ood_accuracy"],
        "provided_other_accuracy": evaluations["provided_other"]["metrics"]["ood_accuracy"],
    }


def is_promising(baseline: dict[str, float], current: dict[str, float]) -> bool:
    """Return whether the addon clears the first-pass promising bar."""

    known_floor = baseline["test_known_balanced_accuracy"] - 0.01
    return (
        current["test_ood_accuracy"] > baseline["test_ood_accuracy"]
        and current["test_known_balanced_accuracy"] >= known_floor
    )


def is_replacement_candidate(baseline: dict[str, float], current: dict[str, float]) -> bool:
    """Return whether the addon clears the stronger replacement bar."""

    return (
        current["test_known_balanced_accuracy"] > baseline["test_known_balanced_accuracy"]
        and current["test_ood_accuracy"] > baseline["test_ood_accuracy"]
        and current["provided_other_accuracy"] > baseline["provided_other_accuracy"]
    )


def write_outputs(
    run_dir: Path,
    args: Namespace,
    selected: dict[str, Any],
    evaluations: dict[str, dict[str, Any]],
    splits: ExperimentSplits,
    latency: dict[str, Any],
    baseline: dict[str, Any],
) -> None:
    """Write metrics, predictions, selected params, thresholds, and metadata."""

    write_json(run_dir / "selected_classifier_params.json", selected_classifier_payload(args, selected))
    write_json(run_dir / "candidate_results.json", candidate_results_payload(selected))
    write_json(run_dir / "length_bucket_selection.json", selected["threshold_tuning"])
    write_json(run_dir / "latency_summary.json", latency)
    write_json(run_dir / "baseline_comparison.json", baseline)
    write_json(run_dir / "splits.json", splits.split_ids())
    write_json(run_dir / "metadata.json", metadata_payload(args, selected, evaluations, latency, baseline))
    for split_name, payload in evaluations.items():
        write_json(run_dir / f"metrics_{split_name}.json", payload["metrics"])
        write_jsonl(run_dir / f"predictions_{split_name}.jsonl", payload["predictions"])


def selected_classifier_payload(args: Namespace, selected: dict[str, Any]) -> dict[str, Any]:
    """Return the selected classifier parameters without the fitted estimator."""

    return {
        "embedding_model": args.model_name,
        "classifier": "LogisticRegression",
        "c_value": selected["c_value"],
        "classifier_params": selected["classifier_params"],
    }


def candidate_results_payload(selected: dict[str, Any]) -> dict[str, Any]:
    """Return the selected candidate validation result in a compact shape."""

    return {
        "selected_c_value": selected["c_value"],
        "selected_validation_metrics": selected["validation_metrics"],
        "candidates": selected["all_candidate_results"],
    }


def metadata_payload(
    args: Namespace,
    selected: dict[str, Any],
    evaluations: dict[str, dict[str, Any]],
    latency: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Return compact metadata describing the addon run."""

    return {
        "runner": "transformer_addon.py",
        "embedding_model": args.model_name,
        "frozen_embeddings": True,
        "classifier": "LogisticRegression",
        "c_values": args.c_values,
        "selected_c_value": selected["c_value"],
        "selected_policy": selected["threshold_tuning"]["selected"]["policy"],
        "validation_metrics": evaluations["validation"]["metrics"],
        "test_metrics": evaluations["test"]["metrics"],
        "provided_other_metrics": evaluations["provided_other"]["metrics"],
        "latency_summary": latency,
        "baseline_comparison": baseline,
    }


def print_summary(
    run_dir: Path,
    selected: dict[str, Any],
    evaluations: dict[str, dict[str, Any]],
    baseline: dict[str, Any],
    latency: dict[str, Any],
) -> None:
    """Print compact run outcome for humans."""

    print(f"Run directory: {run_dir}")
    print(f"Selected C: {selected['c_value']}")
    print(f"Selected policy: {selected['threshold_tuning']['selected']['policy']['params']}")
    for split_name in ("validation", "test", "provided_other"):
        print(f"{split_name}: {format_metrics(evaluations[split_name]['metrics'])}")
    print(f"Promising vs baseline: {baseline['promising']}")
    print(f"Replacement candidate: {baseline['replacement_candidate']}")
    print(f"Latency: {latency}")


def format_metrics(metrics: dict[str, Any]) -> str:
    """Format headline metrics used during benchmark review."""

    return (
        f"known_balanced_accuracy={metrics['known_balanced_accuracy']:.4f} "
        f"known_accuracy={metrics['known_accuracy']:.4f} "
        f"ood_accuracy={metrics['ood_accuracy']:.4f} "
        f"overall_accuracy={metrics['overall_accuracy']:.4f}"
    )


def configure_logging(level_name: str) -> None:
    """Configure console logging for experiment progress."""

    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run the addon benchmark."""

    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    run_experiment(args)


if __name__ == "__main__":
    main()
