"""Project-relative path resolution without personal absolute paths."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    override = os.environ.get("EFFICIENT_LLM_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root() / path).resolve()


def data_root() -> Path:
    return resolve_project_path(os.environ.get("EFFICIENT_LLM_DATA_ROOT", "data"))


def artifact_root() -> Path:
    return resolve_project_path(os.environ.get("EFFICIENT_LLM_ARTIFACT_ROOT", "artifacts"))


def result_root() -> Path:
    return resolve_project_path(os.environ.get("EFFICIENT_LLM_RESULT_ROOT", "results"))

