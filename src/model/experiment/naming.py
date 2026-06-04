"""Naming helpers for experiment runs."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_run_id() -> str:
    """Return a UTC timestamp suitable for run directory names."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
