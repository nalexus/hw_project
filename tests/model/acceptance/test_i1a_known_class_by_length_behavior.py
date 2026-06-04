"""Aggregate behavior test for known classes across runtime length buckets."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from src.api.model_loader import build_predictor
from src.api.settings import PROJECT_ROOT, load_runtime_config, load_settings


FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "i1a_known_class_by_length"
    / "fixtures.jsonl"
)
DEFAULT_GOLDEN_CONFIG_PATH = Path(__file__).resolve().parent / "configs.yaml"


def load_fixture_rows() -> list[dict[str, Any]]:
    """Load JSONL fixture rows from the committed test data file."""

    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.fixture(scope="module")
def predictor(request):
    """Load the requested run predictor once for fixture behavior checks."""

    return build_predictor(settings_for_pipeline_run(request.config.getoption("pipeline_run")))


def settings_for_pipeline_run(pipeline_run: str | None):
    """Return API settings for the promoted run or requested pipeline run."""

    settings = load_settings()
    if not pipeline_run:
        return settings
    runtime_config_path = resolve_runtime_config_path(pipeline_run)
    runtime_config = load_runtime_config(runtime_config_path, PROJECT_ROOT)
    return replace(
        settings,
        model_path=Path(runtime_config["model_path"]),
        threshold_policy=runtime_config,
        runtime_config_path=runtime_config_path,
        model_version=runtime_config_path.parent.name,
    )


def resolve_runtime_config_path(pipeline_run: str) -> Path:
    """Resolve a run name, run directory, or runtime config path."""

    path = Path(pipeline_run)
    candidates = candidate_runtime_config_paths(path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    rendered = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"--pipeline-run did not resolve to runtime_config.json. Checked:\n{rendered}"
    )


def candidate_runtime_config_paths(path: Path) -> list[Path]:
    """Return possible runtime config paths for one CLI value."""

    if path.name == "runtime_config.json":
        return [path if path.is_absolute() else PROJECT_ROOT / path]
    if path.is_absolute() or len(path.parts) > 1:
        candidate_dir = path if path.is_absolute() else PROJECT_ROOT / path
    else:
        candidate_dir = PROJECT_ROOT / "best_pipeline_search_runs" / path
    return [candidate_dir / "runtime_config.json"]


def test_golden_known_class_behavior(predictor, request):
    """Evaluate all I.1.A known-class fixtures as one golden behavior gate."""

    rows = load_fixture_rows()
    gate_config = load_gate_config(request.config.getoption("golden_behavior_config"))
    pipeline_run = request.config.getoption("pipeline_run") or "promoted _PROD"
    predictions = predictor.predict(np.array([row["text"] for row in rows], dtype=object))
    results = [
        {
            **row,
            "predicted_label": str(prediction),
            "passed": str(prediction) == row["expected_label"],
        }
        for row, prediction in zip(rows, predictions)
    ]
    report = build_golden_report(results, gate_config)
    report["summary"].insert(1, f"pipeline_run: {pipeline_run}")
    request.config._golden_known_report = report["summary"]

    if report["gate_failures"]:
        pytest.fail(format_gate_failures(report["gate_failures"]), pytrace=False)


def load_gate_config(config_path: str | None) -> dict[str, float]:
    """Load configured golden behavior acceptance thresholds."""

    path = Path(config_path) if config_path else DEFAULT_GOLDEN_CONFIG_PATH
    path = path if path.is_absolute() else PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as handle:
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


def format_gate_failures(failures: list[str]) -> str:
    """Return assertion message for failed golden acceptance gates."""

    if not failures:
        return ""
    return "Golden known-class gate failures:\n" + "\n".join(failures)
