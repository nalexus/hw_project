"""Runtime settings resolution for the API."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ApiSettings:
    """Typed runtime settings loaded from YAML and environment."""

    model_path: Path
    threshold: float
    threshold_policy: dict[str, Any] | None
    runtime_config_path: Path | None
    max_document_length: int
    model_version: str


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "api" / "settings.yaml"


def load_settings(settings_path: Path | None = None) -> ApiSettings:
    """Load API settings and resolve the promoted pipeline run."""

    path = settings_path or DEFAULT_SETTINGS_PATH
    raw_settings = read_settings_yaml(path)
    project_root = PROJECT_ROOT
    prod_runs_dir = resolved_path(
        raw_settings.get("prod_runs_dir", "best_pipeline_search_runs"), project_root
    )
    runtime_config_path = selected_runtime_config_path(raw_settings, prod_runs_dir, project_root)
    runtime_config = load_runtime_config(runtime_config_path, project_root)
    return ApiSettings(
        model_path=selected_model_path(raw_settings, runtime_config, prod_runs_dir, project_root),
        threshold=float(raw_settings.get("threshold", 0.16)),
        threshold_policy=runtime_config or None,
        runtime_config_path=runtime_config_path,
        max_document_length=int(raw_settings.get("max_document_length", 10000)),
        model_version=selected_model_version(raw_settings, runtime_config_path),
    )


def read_settings_yaml(path: Path) -> dict[str, Any]:
    """Read API YAML settings, returning an empty mapping when absent."""

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def selected_runtime_config_path(
    raw_settings: dict[str, Any], prod_runs_dir: Path, project_root: Path
) -> Path | None:
    """Return explicit or promoted runtime config path for serving."""

    runtime_config_path = configured_runtime_config_path(raw_settings, project_root)
    env_model_path = os.getenv("MODEL_PATH")
    raw_model_path = raw_settings.get("model_path")
    if runtime_config_path is None and env_model_path is None and raw_model_path is None:
        return promoted_runtime_config_path(prod_runs_dir)
    return runtime_config_path


def selected_model_path(
    raw_settings: dict[str, Any],
    runtime_config: dict[str, Any],
    prod_runs_dir: Path,
    project_root: Path,
) -> Path:
    """Return the model path selected by env, runtime config, or promoted fallback."""

    configured_model_path = os.getenv("MODEL_PATH")
    if configured_model_path is None:
        configured_model_path = runtime_config.get("model_path")
    if configured_model_path is None:
        configured_model_path = raw_settings.get("model_path")
    if configured_model_path is None:
        configured_model_path = raw_settings.get(
            "model_path", missing_promoted_model_path(prod_runs_dir)
        )
    return resolved_path(configured_model_path, project_root)


def selected_model_version(
    raw_settings: dict[str, Any], runtime_config_path: Path | None
) -> str:
    """Return the model version exposed in API responses."""

    return os.getenv(
        "MODEL_VERSION",
        str(raw_settings.get("model_version", inferred_model_version(runtime_config_path))),
    )


def configured_runtime_config_path(
    raw_settings: dict[str, Any], project_root: Path
) -> Path | None:
    """Return an explicit runtime config override if one is configured."""

    return resolved_optional_path(
        os.getenv("RUNTIME_CONFIG_PATH", raw_settings.get("runtime_config_path")),
        project_root,
    )


def promoted_runtime_config_path(prod_runs_dir: Path) -> Path | None:
    """Find the single promoted run's runtime config."""

    prod_runs = promoted_run_dirs(prod_runs_dir)
    if len(prod_runs) != 1:
        return None
    runtime_config_path = prod_runs[0] / "runtime_config.json"
    if not runtime_config_path.exists():
        return None
    return runtime_config_path


def promoted_run_dirs(prod_runs_dir: Path) -> list[Path]:
    """Return run directories marked as API-serving PROD runs."""

    if not prod_runs_dir.exists():
        return []
    return sorted(
        path
        for path in prod_runs_dir.iterdir()
        if path.is_dir() and "_PROD" in path.name
    )


def missing_promoted_model_path(prod_runs_dir: Path) -> Path:
    """Return a descriptive missing path when no promoted run is available."""

    prod_runs = promoted_run_dirs(prod_runs_dir)
    if not prod_runs:
        return prod_runs_dir / "NO_PROD_RUN_FOUND.joblib"
    if len(prod_runs) > 1:
        return prod_runs_dir / "MULTIPLE_PROD_RUNS_FOUND.joblib"
    return prod_runs[0] / "NO_RUNTIME_CONFIG_FOUND.joblib"


def inferred_model_version(runtime_config_path: Path | None) -> str:
    """Use the promoted run folder name as the default model version."""

    if runtime_config_path is None:
        return "unpromoted"
    return runtime_config_path.parent.name


def load_runtime_config(
    runtime_config_path: Path | None, project_root: Path
) -> dict[str, Any]:
    """Read a selected training runtime config when one is configured."""

    if runtime_config_path is None:
        return {}
    with runtime_config_path.open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)
    if "model_path" in config:
        config["model_path"] = str(
            resolved_runtime_model_path(
                config["model_path"], project_root, runtime_config_path.parent
            )
        )
    return config


def resolved_runtime_model_path(value: str | Path, root: Path, config_dir: Path) -> Path:
    """Resolve runtime model paths from repo root, then config directory."""

    path = Path(value)
    if path.is_absolute():
        return path
    root_path = root / path
    if root_path.exists():
        return root_path
    return config_dir / path


def resolved_optional_path(value: str | Path | None, root: Path) -> Path | None:
    """Resolve an optional path relative to the project root."""

    if value in (None, ""):
        return None
    return resolved_path(value, root)


def resolved_path(value: str | Path, root: Path) -> Path:
    """Resolve a configured path relative to the project root."""

    path = Path(value)
    if path.is_absolute():
        return path
    return root / path
