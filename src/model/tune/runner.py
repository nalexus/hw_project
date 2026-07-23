from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.model.data_preparation.loader import DatasetLoader
from src.model.data_preparation.splits import SplitBuilder
from src.model.data_preparation.text import TextProcessor
from src.model.evaluate.evaluator import KnownEvaluation, ModelEvaluator, OtherEvaluation
from src.model.evaluate.metrics import summarize_gate
from src.model.predict.predictor import TfidfOODPolicy
from src.model.train.trainer import ModelFactory
from src.model.train.validators import TrainingConfigModel
from src.model.tune.tuner import (
    CandidateCatalog,
    CrossValidationTuner,
    OutOfFoldSignalCollector,
    SupportAwareConfidenceTuner,
)
from src.model.tune.validators import TuneConfigModel


def build_parser() -> ArgumentParser:
    """Build CLI options for the full plain-TFIDF model workflow."""

    parser = ArgumentParser(description="Tune and persist the TF-IDF OOD model.")
    parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("config/model/train.yaml"),
    )
    parser.add_argument(
        "--tune-config",
        type=Path,
        default=Path("config/model/tune.yaml"),
    )
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--test-frac", type=float)
    parser.add_argument("--random-state", type=int)
    parser.add_argument("--cv-folds", type=int)
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--promote-selected", action="store_true", default=None)
    parser.add_argument("--support-tail-rate", type=float)
    parser.add_argument("--target-accepted-error", type=float)
    parser.add_argument("--minimum-correct-coverage", type=float)
    return parser


@dataclass(frozen=True)
class TuningResult:
    """Selected model configuration, OOD policy, and persisted run location."""

    run_dir: Path
    best_config: dict[str, Any]
    policy: TfidfOODPolicy


class ConsoleReporter:
    """Render tuning progress, evaluation evidence, and the selected policy."""

    def report_stage(self, message: str) -> None:
        """Print one completed workflow stage heading."""

        print(message, flush=True)

    def report_progress(self, stage: str, completed: int, total: int) -> None:
        """Render one compact in-place progress bar for a fitting stage."""

        width = 24
        filled = round(width * completed / total) if total else width
        bar = "#" * filled + "-" * (width - filled)
        end = "\n" if completed == total else ""
        print(f"\r{stage}: [{bar}] {completed}/{total}", end=end, flush=True)

    def report_evaluation(
        self,
        oof_diagnostics: pd.Series,
        test_evaluation: KnownEvaluation,
        other_evaluation: OtherEvaluation,
    ) -> None:
        """Print OOF diagnostics plus held-out known and OOD evaluation evidence."""

        print("\nEVALUATION STATS IN DETAIL")
        print("OOF signal diagnostics for known-class errors")
        print(oof_diagnostics.to_string())
        print("\nTest data known-class evaluation results overall")
        print(test_evaluation.overall.to_string(index=False))
        print("\nTest data known-class evaluation results by class")
        print(test_evaluation.by_class.to_string(index=False))
        print("\nTest data known-class evaluation results by length bucket")
        print(test_evaluation.by_bucket.to_string(index=False))
        print("\nOther data OOD-policy signals")
        columns = ["file_name", "raw_label", "max_probability", "oov_ratio", "accepted"]
        print(other_evaluation.results[columns].to_string(index=False))
        print(f"OOD rejection rate: {other_evaluation.rejection_rate:.6f}")
        print(
            "Other accuracy: "
            f"{other_evaluation.correct_predictions}/{other_evaluation.num_examples} "
            f"({other_evaluation.accuracy:.2%})"
        )
        print()
        self._report_evaluation_summary(test_evaluation, other_evaluation)
        print()

    def report_selection(self, result: TuningResult) -> None:
        """Print the selected model and persisted OOD thresholds."""

        print(f"Selected pipeline: {result.best_config['pipeline_id']}")
        print(
            "Minimal probability threshold for accepting predicted class "
            f"(otherwise, the class is '{result.policy.other_label}'): "
            f"{result.policy.probability_threshold:.6f}"
        )
        print(f"Maximum out-of-vocabulary ratio: {result.policy.max_oov_ratio:.6f}")
        print(f"Saved run: {result.run_dir}")

    @staticmethod
    def _report_evaluation_summary(
        test_evaluation: KnownEvaluation,
        other_evaluation: OtherEvaluation,
    ) -> None:
        """Print the three final metrics used to review one selected run."""

        overall = test_evaluation.overall.iloc[0]
        print("EVALUATION SUMMARY")
        print(
            "Test data known-class balanced accuracy: "
            f"{test_evaluation.raw_metrics['balanced_accuracy']:.2%}"
        )
        print(
            "Test data known-class accepted: "
            f"{int(overall['accepted'])}/{int(overall['num_examples'])} "
            f"({overall['coverage']:.2%})"
        )
        print(
            "Other accuracy (OOD): "
            f"{other_evaluation.correct_predictions}/{other_evaluation.num_examples} "
            f"({other_evaluation.accuracy:.2%})"
        )


class SelectedRunStore:
    """Persist selected run artifacts and maintain the single PROD run marker."""

    def __init__(self, config: TrainingConfigModel) -> None:
        """Store the run naming and promotion configuration."""

        self.config = config

    def save(
        self,
        model: Any,
        policy: TfidfOODPolicy,
        cv_results: pd.DataFrame,
        oof_results: pd.DataFrame,
        test_evaluation: KnownEvaluation,
        other_evaluation: OtherEvaluation,
        split_ids: dict[str, list[str]],
    ) -> Path:
        """Write the selected pipeline, policy, metrics, and split identifiers."""

        run_dir = self._prepare_run_dir()
        run_dir.mkdir(parents=True)
        joblib.dump(model, run_dir / "model.joblib")
        self._write_json(
            run_dir / "runtime_config.json",
            policy.to_runtime_config("model.joblib"),
        )
        self._write_json(run_dir / "candidate_cv_results.json", cv_results.to_dict("records"))
        self._write_json(
            run_dir / "oof_policy_metrics.json",
            summarize_gate(policy.apply(oof_results)).to_dict("records"),
        )
        self._write_json(run_dir / "metrics_test.json", test_evaluation.to_dict())
        self._write_json(run_dir / "metrics_other.json", other_evaluation.to_dict())
        self._write_json(run_dir / "splits.json", split_ids)
        return run_dir

    def _prepare_run_dir(self) -> Path:
        """Create a timestamped name and transfer PROD status when requested."""

        self.config.runs_dir.mkdir(parents=True, exist_ok=True)
        run_name = (self.config.run_name or f"run_{self._utc_run_id()}").replace(
            "_PROD",
            "",
        )
        run_name = f"{run_name}_PROD" if self.config.promote_selected else run_name
        run_dir = self.config.runs_dir / run_name
        if run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        if self.config.promote_selected:
            self._demote_existing_prod_runs()
        return run_dir

    def _demote_existing_prod_runs(self) -> None:
        """Remove the PROD suffix from older served runs without altering artifacts."""

        for run_dir in self.config.runs_dir.iterdir():
            if not run_dir.is_dir() or "_PROD" not in run_dir.name:
                continue
            demoted_dir = run_dir.with_name(run_dir.name.replace("_PROD", ""))
            if demoted_dir.exists():
                raise FileExistsError(f"Cannot demote {run_dir}; {demoted_dir} exists.")
            run_dir.rename(demoted_dir)

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        """Write JSON while converting pandas and NumPy scalars when required."""

        path.write_text(
            json.dumps(payload, indent=2, default=SelectedRunStore._json_default),
            encoding="utf-8",
        )

    @staticmethod
    def _json_default(value: Any) -> Any:
        """Convert common scientific-Python values into JSON-compatible values."""

        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "item"):
            return value.item()
        raise TypeError(f"Cannot serialize {type(value).__name__}")

    @staticmethod
    def _utc_run_id() -> str:
        """Return a UTC timestamp suitable for a reproducible run directory."""

        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class TuningWorkflow:
    """Coordinate splitting, tuning, evaluation, reporting, and persistence."""

    def __init__(
        self,
        training_config: TrainingConfigModel,
        tune_config: TuneConfigModel,
        reporter: ConsoleReporter | None = None,
        run_store: SelectedRunStore | None = None,
    ) -> None:
        """Store configuration and optional adapters for terminal output and storage."""

        self.training_config = training_config
        self.tune_config = tune_config
        self.reporter = reporter or ConsoleReporter()
        self.run_store = run_store or SelectedRunStore(training_config)

    def run(self) -> TuningResult:
        """Select, evaluate, and persist one TF-IDF OOD model run."""

        splits = self._build_splits()
        train_df = self._records_frame(splits.train)
        best_config, cv_results = self._select_candidate(train_df)
        model = self._fit_model(best_config, train_df)
        oof_results = self._collect_oof_signals(best_config, train_df)
        self.reporter.report_stage("Tuning OOD policy...")
        policy = SupportAwareConfidenceTuner(**self.tune_config.model_dump()).tune(
            oof_results,
            self.training_config.other_label,
        )
        self.reporter.report_stage("Evaluating test and other data...")
        evaluator = ModelEvaluator()
        test_evaluation = evaluator.evaluate_known(model, splits.test, policy)
        other_evaluation = evaluator.evaluate_other(model, splits.other, policy)
        self.reporter.report_evaluation(
            evaluator.oof_signal_auroc(oof_results),
            test_evaluation,
            other_evaluation,
        )
        self.reporter.report_stage("Saving selected run...")
        run_dir = self.run_store.save(
            model,
            policy,
            cv_results,
            oof_results,
            test_evaluation,
            other_evaluation,
            splits.split_ids(),
        )
        return TuningResult(run_dir, best_config, policy)

    def _build_splits(self):
        """Load configured documents and create deterministic known-class splits."""

        text_processor = TextProcessor.from_config(self.training_config.text_config_path)
        bundle = DatasetLoader(
            self.training_config.dataset_dir,
            other_label=self.training_config.other_label,
            exclusions_config_path=self.training_config.exclusions_config_path,
            text_processor=text_processor,
        ).load()
        return SplitBuilder(
            test_frac=self.training_config.test_frac,
            random_state=self.training_config.random_state,
        ).build(bundle)

    def _select_candidate(
        self,
        train_df: pd.DataFrame,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        """Select the strongest configured pipeline through stratified CV."""

        configs = CandidateCatalog(self.training_config.candidates_config_path).build(
            self.training_config.candidate_limit
        )
        return CrossValidationTuner(
            configs,
            cv_folds=self.training_config.cv_folds,
            random_state=self.training_config.random_state,
            progress=self.reporter.report_progress,
        ).tune(train_df)

    def _fit_model(self, config: dict[str, Any], train_df: pd.DataFrame) -> Any:
        """Fit the selected TF-IDF and logistic-regression pipeline on train rows."""

        self.reporter.report_stage("Fitting selected best pipeline...")
        model = ModelFactory(self.training_config.random_state).build_tfidf_logreg(config)
        model.fit(train_df["text"], train_df["class"])
        return model

    def _collect_oof_signals(
        self,
        config: dict[str, Any],
        train_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Collect unbiased train-row signals for OOD policy selection."""

        self.reporter.report_stage("Collecting out-of-fold signals...")
        return OutOfFoldSignalCollector(
            config,
            cv_folds=self.training_config.cv_folds,
            random_state=self.training_config.random_state,
            progress=self.reporter.report_progress,
        ).collect(train_df)

    @staticmethod
    def _records_frame(records) -> pd.DataFrame:
        """Convert production records into the notebook-compatible DataFrame shape."""

        return pd.DataFrame(record.as_row() for record in records)


def main(argv: list[str] | None = None) -> None:
    """Load configuration, run the workflow, and print the selected policy."""

    args = build_parser().parse_args(argv)
    training_config = TrainingConfigModel.from_yaml(args.train_config, vars(args))
    tune_config = TuneConfigModel.from_yaml(args.tune_config, vars(args))
    reporter = ConsoleReporter()
    result = TuningWorkflow(training_config, tune_config, reporter).run()
    reporter.report_selection(result)


if __name__ == "__main__":
    main()
