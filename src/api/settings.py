"""Compatibility exports for API settings."""

from src.api.config.settings import (
    DEFAULT_SETTINGS_PATH,
    PROJECT_ROOT,
    ApiSettings,
    configured_runtime_config_path,
    inferred_model_version,
    load_runtime_config,
    load_settings,
    missing_promoted_model_path,
    promoted_run_dirs,
    promoted_runtime_config_path,
    resolved_optional_path,
    resolved_path,
    resolved_runtime_model_path,
    selected_model_path,
    selected_model_version,
    selected_runtime_config_path,
)


__all__ = [
    "ApiSettings",
    "DEFAULT_SETTINGS_PATH",
    "PROJECT_ROOT",
    "configured_runtime_config_path",
    "inferred_model_version",
    "load_runtime_config",
    "load_settings",
    "missing_promoted_model_path",
    "promoted_run_dirs",
    "promoted_runtime_config_path",
    "resolved_optional_path",
    "resolved_path",
    "resolved_runtime_model_path",
    "selected_model_path",
    "selected_model_version",
    "selected_runtime_config_path",
]
