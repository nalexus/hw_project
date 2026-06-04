"""Pytest reporting hooks for project-specific model evaluation summaries."""

from __future__ import annotations


def pytest_addoption(parser):
    """Register project-specific pytest CLI options."""

    parser.addoption(
        "--pipeline-run",
        "--pipeline_run",
        action="store",
        default=None,
        dest="pipeline_run",
        help="Run folder name, run folder path, or runtime_config.json path for golden model checks.",
    )
    parser.addoption(
        "--golden-behavior-config",
        action="store",
        default=None,
        dest="golden_behavior_config",
        help="YAML config with acceptance thresholds for golden behavior checks.",
    )


def pytest_terminal_summary(terminalreporter):
    """Print aggregate golden behavior metrics when a test run produced them."""

    report = getattr(terminalreporter.config, "_golden_known_report", None)
    if not report:
        return
    terminalreporter.write_sep("-", "golden known-class behavior")
    for line in report:
        terminalreporter.write_line(line)
