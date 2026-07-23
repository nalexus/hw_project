"""Aggregate behavior test for known classes across runtime length buckets."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest
import yaml

from src.model.predict.predictor import PredictorMultiClass


FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "i1a_known_class_by_length"
    / "fixtures.jsonl"
)
GATE_CONFIG_PATH = FIXTURE_PATH.parent / "gates.yaml"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = PROJECT_ROOT / "best_pipeline_search_runs"


def load_fixture_rows() -> list[dict[str, Any]]:
    """Load JSONL fixture rows from the committed test data file."""

    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_production_predictor() -> tuple[str, PredictorMultiClass]:
    """Load the single production-marked TF-IDF run for acceptance testing."""

    prod_runs = sorted(path for path in RUNS_DIR.iterdir() if "_PROD" in path.name)
    if len(prod_runs) != 1:
        raise ValueError("Expected exactly one *_PROD run for acceptance testing.")
    run_dir = prod_runs[0]
    return run_dir.name, load_runtime_predictor(run_dir / "runtime_config.json")


def load_runtime_predictor(runtime_config_path: Path) -> PredictorMultiClass:
    """Load the selected plain TF-IDF pipeline and its direct OOD policy."""

    runtime_config = read_json(runtime_config_path)
    model_path = resolved_model_path(runtime_config_path.parent, runtime_config["model_path"])
    pipeline = joblib.load(model_path)
    return PredictorMultiClass(pipeline=pipeline, threshold_policy=runtime_config)


def resolved_model_path(run_dir: Path, configured_model_path: str) -> Path:
    """Resolve runtime-config model paths relative to their run directory."""

    path = Path(configured_model_path)
    return path if path.is_absolute() else run_dir / path


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_golden_known_class_behavior():
    """Evaluate all I.1.A known-class fixtures as one golden behavior gate."""

    rows = load_fixture_rows()
    pipeline_name, predictor = load_production_predictor()
    gate_config = load_gate_config()
    predictions = predictor.predict(
        np.array([row["text"] for row in rows], dtype=object)
    )["predicted_label"]
    results = [
        {
            **row,
            "predicted_label": str(prediction),
            "passed": str(prediction) == row["expected_label"],
        }
        for row, prediction in zip(rows, predictions)
    ]
    report = build_golden_report(results, gate_config)
    report["summary"].insert(1, f"production_run: {pipeline_name}")

    if report["gate_failures"]:
        pytest.fail("\n".join(report["summary"]), pytrace=False)


def load_gate_config() -> dict[str, float]:
    """Load committed golden behavior acceptance thresholds."""

    with GATE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw = raw.get("golden_behavior_gate", raw)
    return {
        "minimum_golden_accuracy": float(raw.get("minimum_golden_accuracy", 1.0)),
        "minimum_golden_bal_acc": float(raw.get("minimum_golden_bal_acc", 1.0)),
        "minimum_category_accuracy": float(raw.get("minimum_category_accuracy", 1.0)),
        "minimum_length_accuracy": float(raw.get("minimum_length_accuracy", 1.0)),
    }


def build_golden_report(
    results: list[dict[str, Any]], gate_config: dict[str, float]
) -> dict[str, Any]:
    """Return aggregate metrics and compact report lines for golden fixtures."""

    labels = sorted({row["expected_label"] for row in results})
    buckets = sorted({row["length_bucket"] for row in results})
    total = len(results)
    correct = sum(row["passed"] for row in results)
    by_label = group_scores(results, "expected_label", labels)
    by_bucket = group_scores(results, "length_bucket", buckets)
    golden_bal_acc = sum(item["accuracy"] for item in by_label) / len(by_label)
    failures = [row for row in results if not row["passed"]]
    gate_failures = evaluate_gate(
        total, correct, golden_bal_acc, by_label, by_bucket, gate_config
    )
    return {
        "failures": failures,
        "gate_failures": gate_failures,
        "summary": format_report(
            total, correct, golden_bal_acc, by_label, by_bucket, failures, gate_config, gate_failures
        ),
    }


def evaluate_gate(
    total: int,
    correct: int,
    golden_bal_acc: float,
    by_label: list[dict[str, Any]],
    by_bucket: list[dict[str, Any]],
    gate_config: dict[str, float],
) -> list[str]:
    """Return configured acceptance-gate failures."""

    accuracy = correct / total if total else 0.0
    failures = []
    if accuracy < gate_config["minimum_golden_accuracy"]:
        failures.append(
            f"golden_accuracy {accuracy:.4f} < {gate_config['minimum_golden_accuracy']:.4f}"
        )
    if golden_bal_acc < gate_config["minimum_golden_bal_acc"]:
        failures.append(
            f"golden_bal_acc {golden_bal_acc:.4f} < {gate_config['minimum_golden_bal_acc']:.4f}"
        )
    failures.extend(group_gate_failures(by_label, "category", gate_config["minimum_category_accuracy"]))
    failures.extend(group_gate_failures(by_bucket, "length", gate_config["minimum_length_accuracy"]))
    return failures


def group_gate_failures(groups: list[dict[str, Any]], name: str, threshold: float) -> list[str]:
    """Return threshold failures for grouped golden metrics."""

    return [
        f"{name} {group['name']} accuracy {group['accuracy']:.4f} < {threshold:.4f}"
        for group in groups
        if group["accuracy"] < threshold
    ]


def group_scores(
    results: list[dict[str, Any]], key: str, ordered_names: list[str]
) -> list[dict[str, Any]]:
    """Return pass counts and accuracy for one result grouping key."""

    totals = Counter(str(row[key]) for row in results)
    correct = defaultdict(int)
    for row in results:
        if row["passed"]:
            correct[str(row[key])] += 1
    return [
        {
            "name": name,
            "correct": correct[name],
            "total": totals[name],
            "accuracy": correct[name] / totals[name] if totals[name] else 0.0,
        }
        for name in ordered_names
    ]


def format_report(
    total: int,
    correct: int,
    golden_bal_acc: float,
    by_label: list[dict[str, Any]],
    by_bucket: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    gate_config: dict[str, float],
    gate_failures: list[str],
) -> list[str]:
    """Format aggregate golden known-class metrics for pytest summary output."""

    accuracy = correct / total if total else 0.0
    lines = [
        "Golden known-class behavior",
        f"overall: samples={total} golden_accuracy={accuracy:.4f} golden_bal_acc={golden_bal_acc:.4f}",
        "gate:",
        *format_gate_lines(gate_config, gate_failures),
        "",
        "by category:",
        *format_group_lines(by_label),
        "",
        "by length:",
        *format_group_lines(by_bucket),
    ]
    if failures:
        lines.extend(["", "failures:", *format_failure_rows(failures)])
    return lines


def format_gate_lines(gate_config: dict[str, float], gate_failures: list[str]) -> list[str]:
    """Format configured acceptance gate status."""

    status = "PASS" if not gate_failures else "FAIL"
    return [
        f"  status={status}",
        "  thresholds: "
        f"golden_accuracy>={gate_config['minimum_golden_accuracy']:.2f} "
        f"golden_bal_acc>={gate_config['minimum_golden_bal_acc']:.2f} "
        f"category_acc>={gate_config['minimum_category_accuracy']:.2f} "
        f"length_acc>={gate_config['minimum_length_accuracy']:.2f}",
        *[f"  {failure}" for failure in gate_failures],
    ]


def format_group_lines(groups: list[dict[str, Any]]) -> list[str]:
    """Format one grouped metric table."""

    return [
        f"{group['name']:<13} {group['correct']:>2}/{group['total']:<2} {group['accuracy']:.4f}"
        for group in groups
    ]


def format_failure_rows(failures: list[dict[str, Any]]) -> list[str]:
    """Format compact failure rows for the assertion and summary."""

    return [
        "record_id                           expected       predicted      length",
        *[
            f"{row['record_id']:<35} {row['expected_label']:<14} "
            f"{row['predicted_label']:<14} {row['length_bucket']}"
            for row in failures
        ],
    ]
