"""CLI runner for the clean selected training workflow."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace, SUPPRESS
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from src.model.data_preparation.leakage import LeakageChecker
from src.model.data_preparation.loader import DatasetLoader
from src.model.data_preparation.splits import SplitBuilder
from src.model.evaluate.evaluator import ModelEvaluator
from src.model.evaluate.predictions import build_prediction_rows
from src.model.experiment.manifest import selection_summary_lines
from src.model.experiment.naming import utc_run_id
from src.model.experiment.writer import ExperimentWriter
from src.model.train.models import fit_calibrated_model
from src.model.train.schemas import FittedCandidate
from src.model.train.validators import (
    DEFAULT_TRAIN_CONFIG_PATH,
    TrainingConfig,
    load_training_config,
)
from src.model.tune.candidates import build_candidate_configs, cross_validate_candidates
from src.model.tune.thresholds import (
    threshold_values,
    tune_global_threshold,
    tune_length_bucket_thresholds,
)
from src.model.tune.validators import DEFAULT_TUNE_CONFIG_PATH


LOGGER = logging.getLogger(__name__)
PROD_MARKER = "_PROD"
RUN_TIMESTAMP_RE = re.compile(r"\d{8}T\d{6}Z")


def build_parser() -> ArgumentParser:
    """Build CLI arguments for the clean training runner."""

    parser = ArgumentParser(description="Run the clean selected training workflow.")
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG_PATH)
    parser.add_argument("--tune-config", type=Path, default=DEFAULT_TUNE_CONFIG_PATH)
    parser.add_argument("--dataset-dir", type=Path, default=SUPPRESS)
    parser.add_argument("--runs-dir", type=Path, default=SUPPRESS)
    parser.add_argument("--run-name", default=SUPPRESS)
    parser.add_argument("--validation-size", type=float, default=SUPPRESS)
    parser.add_argument("--test-size", type=float, default=SUPPRESS)
    parser.add_argument("--random-state", type=int, default=SUPPRESS)
    parser.add_argument("--cv-folds", type=int, default=SUPPRESS)
    parser.add_argument("--calibration-cv-folds", type=int, default=SUPPRESS)
    parser.add_argument("--top-candidates", type=int, default=SUPPRESS)
    parser.add_argument("--candidate-limit", type=int, default=SUPPRESS)
    parser.add_argument("--threshold-step", type=float, default=SUPPRESS)
    parser.add_argument(
        "--threshold-policy",
        choices=["length_bucket", "global"],
        default=SUPPRESS,
        help="Reject-threshold policy to optimize on validation predictions.",
    )
    parser.add_argument("--minimum-known-balanced-accuracy", type=float, default=SUPPRESS)
    parser.add_argument("--near-duplicate-jaccard", type=float, default=SUPPRESS)
    parser.add_argument("--fail-on-near-duplicates", action="store_true", default=SUPPRESS)
    parser.add_argument("--no-synthetic", dest="include_synthetic", action="store_false", default=SUPPRESS)
    parser.add_argument("--promote-selected", action="store_true", default=SUPPRESS)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def build_config(args: Namespace) -> TrainingConfig:
    """Convert CLI arguments into typed training config."""

    return load_training_config(
        train_config_path=args.train_config,
        tune_config_path=args.tune_config,
        overrides=vars(args),
    )


def run_training(config: TrainingConfig) -> dict[str, Any]:
    """Run the full selected-method experiment and write artifacts."""

    run_dir = prepare_run_dir(config)
    bundle = DatasetLoader(config.dataset_dir, config.include_synthetic).load()
    splits = SplitBuilder(config).build(bundle)
    leakage_report = LeakageChecker(config).check(splits)
    known_labels = known_labels_for_splits(splits)
    LOGGER.info(
        "Split sizes train=%s validation=%s test=%s provided_other=%s",
        len(splits.train),
        len(splits.validation),
        len(splits.test),
        len(splits.provided_other),
    )
    cv_results = cross_validate_candidates(splits.train, config, known_labels)
    selected = fit_and_tune_top_candidates(cv_results, splits.train, splits.validation, config, known_labels)
    evaluations = evaluate_selected(selected, splits, known_labels)
    print_selection_summary(selected, evaluations)
    model_path = ExperimentWriter(run_dir, config).write_all(
        selected, cv_results, evaluations, splits, leakage_report
    )
    print_artifact_summary(run_dir, model_path)
    return {"run_dir": run_dir, "model_path": model_path, "selected": selected, "evaluations": evaluations}


def prepare_run_dir(config: TrainingConfig) -> Path:
    """Build the final run directory and transfer PROD ownership when needed."""

    run_name = run_folder_name(config)
    config.runs_dir.mkdir(parents=True, exist_ok=True)
    if config.promote_selected:
        demote_existing_prod_runs(config.runs_dir, run_name)
    return config.runs_dir / run_name


def run_folder_name(config: TrainingConfig) -> str:
    """Return a dated run folder name with optional PROD marker."""

    base_name = (config.run_name or "run").strip()
    if not base_name:
        raise ValueError("--run-name cannot be empty.")
    if PROD_MARKER in base_name and not config.promote_selected:
        raise ValueError("--run-name can include _PROD only with --promote-selected.")
    base_name = base_name.replace(PROD_MARKER, "")
    if RUN_TIMESTAMP_RE.search(base_name) is None:
        base_name = f"{base_name}_{utc_run_id()}"
    if config.promote_selected:
        return f"{base_name}{PROD_MARKER}"
    return base_name


def demote_existing_prod_runs(runs_dir: Path, promoted_run_name: str) -> None:
    """Remove the PROD marker from any previously promoted run directory."""

    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir() or PROD_MARKER not in run_dir.name:
            continue
        if run_dir.name == promoted_run_name:
            continue
        demoted_name = run_dir.name.replace(PROD_MARKER, "")
        demoted_dir = runs_dir / demoted_name
        if demoted_dir.exists():
            raise FileExistsError(f"Cannot demote {run_dir}; {demoted_dir} exists.")
        run_dir.rename(demoted_dir)
        rewrite_run_name_references(demoted_dir, run_dir.name, demoted_name)


def rewrite_run_name_references(run_dir: Path, old_name: str, new_name: str) -> None:
    """Rename the run-local model file and update runtime configs after demotion."""

    old_model = run_dir / f"best_pipeline_{old_name}.joblib"
    new_model = run_dir / f"best_pipeline_{new_name}.joblib"
    if old_model.exists():
        old_model.rename(new_model)
    for filename in ("runtime_config.json", "runtime_config.yaml", "metadata.json"):
        path = run_dir / filename
        if path.exists():
            text = path.read_text(encoding="utf-8")
            path.write_text(text.replace(old_name, new_name), encoding="utf-8")


def known_labels_for_splits(splits) -> list[str]:
    """Return sorted known labels present in model-selection splits."""

    records = [*splits.train, *splits.validation, *splits.test]
    return sorted({record.label for record in records if record.label != "other"})


def fit_and_tune_top_candidates(
    cv_results: list[dict[str, Any]],
    train_records,
    validation_records,
    config: TrainingConfig,
    known_labels: list[str],
) -> FittedCandidate:
    """Fit calibrated top candidates and select by threshold-tuned validation score."""

    candidates = {item.candidate_id: item for item in build_candidate_configs(config.candidate_limit)}
    fitted = []
    thresholds = threshold_values(config.threshold_step)
    top_cv_results = cv_results[: config.top_candidates]
    for index, cv_result in enumerate(top_cv_results, 1):
        candidate_id = cv_result["candidate"]["candidate_id"]
        candidate = candidates[candidate_id]
        model = fit_calibrated_model(candidate, train_records, config)
        rows = build_prediction_rows(model, validation_records)
        tuning = tune_threshold_policy(rows, thresholds, known_labels, config)
        fitted.append(FittedCandidate(candidate, cv_result, model, tuning))
        print_fit_progress(index, len(top_cv_results))
    selected = max(fitted, key=fitted_selection_key)
    print_candidate_selection_leaderboard(fitted, selected)
    return selected


def tune_threshold_policy(
    rows: list[dict[str, Any]],
    thresholds: list[float],
    known_labels: list[str],
    config: TrainingConfig,
) -> dict[str, Any]:
    """Select the configured reject-threshold policy on validation rows."""

    if config.threshold_policy == "global":
        tuning = tune_global_threshold(
            rows, thresholds, known_labels, config.minimum_known_balanced_accuracy
        )
        return {
            **tuning,
            "score_formula": "0.5 * val_known_balanced_accuracy + 0.5 * val_ood_accuracy",
        }
    return tune_length_bucket_thresholds(
        rows, thresholds, known_labels, config.minimum_known_balanced_accuracy
    )


def fitted_selection_key(candidate: FittedCandidate) -> tuple[float, float, float]:
    """Rank fitted candidates by selected validation policy metrics."""

    metrics = candidate.threshold_tuning["selected"]["metrics"]
    return (
        metrics["threshold_selection_score"],
        metrics["known_balanced_accuracy"],
        metrics["ood_accuracy"],
    )


def print_fit_progress(completed: int, total: int) -> None:
    """Print compact calibrated-candidate fitting progress."""

    percent = 100 * completed / total if total else 100
    width = 24
    filled = round(width * completed / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\rFitting top-{total} best candidates on all train data progress "
        f"[{bar}] {completed}/{total} ({percent:4.1f}%)",
        end="" if completed < total else "\n",
        flush=True,
    )


def print_candidate_selection_leaderboard(
    candidates: list[FittedCandidate], selected: FittedCandidate
) -> None:
    """Print final top-candidate selection basis after validation tuning."""

    ranked = sorted(candidates, key=fitted_selection_key, reverse=True)
    columns = [
        ("rank", 4),
        ("candidate", 22),
        ("ngram", 7),
        ("min_df", 6),
        ("subtf", 5),
        ("maxfeat", 7),
        ("weight", 8),
        ("C", 5),
        ("val_score", 9),
        ("val_bal", 7),
        ("val_ood", 7),
    ]
    print("\nTop candidate selection leaderboard")
    print(format_selection_group_header(columns))
    print(format_table_row([name for name, _ in columns], columns))
    print(format_table_row(["-" * width for _, width in columns], columns))
    for rank, candidate in enumerate(ranked, 1):
        print(format_table_row(selection_leaderboard_values(rank, candidate), columns))


def selection_leaderboard_values(
    rank: int, candidate: FittedCandidate
) -> list[str]:
    """Return one formatted row for the final selection leaderboard."""

    candidate_info = candidate.cv_result["candidate"]
    tfidf = candidate_info["tfidf_params"]
    classifier = candidate_info["classifier_params"]
    val_metrics = candidate.threshold_tuning["selected"]["metrics"]
    return [
        str(rank),
        candidate_info["candidate_id"],
        "-".join(str(value) for value in tfidf["ngram_range"]),
        str(tfidf["min_df"]),
        str(tfidf["sublinear_tf"]),
        str(tfidf["max_features"] or "all"),
        str(classifier["class_weight"]),
        f"{classifier['C']:.3g}",
        f"{val_metrics['threshold_selection_score']:.4f}",
        f"{val_metrics['known_balanced_accuracy']:.4f}",
        f"{val_metrics['ood_accuracy']:.4f}",
    ]


def format_table_row(values: list[str], columns: list[tuple[str, int]]) -> str:
    """Return one fixed-width table row."""

    terminal_width = shutil.get_terminal_size((160, 20)).columns
    row = "  ".join(
        value[:width].ljust(width) for value, (_, width) in zip(values, columns)
    )
    return row[:terminal_width]


def format_selection_group_header(columns: list[tuple[str, int]]) -> str:
    """Return grouped header for final candidate-selection table."""

    widths = dict(columns)
    groups = [
        ("", widths["rank"] + 2 + widths["candidate"]),
        ("TF-IDF params", sum(widths[name] for name in ("ngram", "min_df", "subtf", "maxfeat")) + 6),
        ("LogReg params", sum(widths[name] for name in ("weight", "C")) + 2),
        (
            "Evaluation results",
            sum(widths[name] for name in ("val_score", "val_bal", "val_ood")) + 4,
        ),
    ]
    terminal_width = shutil.get_terminal_size((160, 20)).columns
    return "  ".join(label.center(width) for label, width in groups)[:terminal_width]


def evaluate_selected(selected: FittedCandidate, splits, known_labels: list[str]) -> dict[str, dict[str, Any]]:
    """Evaluate selected model and policy on final splits."""

    policy = selected.threshold_tuning["selected"]["policy"]
    evaluator = ModelEvaluator()
    evaluations = {}
    for split_name, records in split_records(splits).items():
        evaluations[split_name] = evaluator.evaluate(
            selected.model, records, policy, known_labels
        ).to_dict()
    return evaluations


def split_records(splits) -> dict[str, list[Any]]:
    """Return evaluation records by split name."""

    return {
        "validation": splits.validation,
        "test": splits.test,
        "provided_other": splits.provided_other,
    }


def print_selection_summary(selected: FittedCandidate, evaluations: dict[str, Any]) -> None:
    """Print selected candidate and metrics before artifact writing."""

    for line in selection_summary_lines(selected, evaluations):
        print(line)


def print_artifact_summary(run_dir: Path, model_path: Path) -> None:
    """Print artifact locations after run files are written."""

    for line in artifact_summary_lines(run_dir, model_path):
        print(line)


def artifact_summary_lines(run_dir: Path, model_path: Path) -> list[str]:
    """Return compact artifact-location summary lines."""

    return [
        f"Experiment runs directory: {run_dir}",
        f"Model pipeline is saved as: {model_path.name}",
    ]


def configure_logging(level_name: str) -> None:
    """Configure console logging for experiment progress."""

    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run the selected training workflow."""

    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    run_training(build_config(args))


if __name__ == "__main__":
    main()
