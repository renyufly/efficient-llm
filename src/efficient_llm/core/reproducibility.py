"""Reproducibility primitives with overwrite protection."""

from __future__ import annotations

import importlib
import json
import os
import random
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class OutputExistsError(FileExistsError):
    """An append-only output would be overwritten."""


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    for module_name in ("numpy", "torch"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        manual_seed = getattr(module, "manual_seed", None)
        if callable(manual_seed):
            manual_seed(seed)
        random_module = getattr(module, "random", None)
        random_seed = getattr(random_module, "seed", None)
        if callable(random_seed):
            random_seed(seed)


def make_run_id(name: str, seed: int, config_hash: str, now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    safe_name = "".join(char if char.isalnum() or char in "-_" else "-" for char in name)
    return f"{timestamp}-{safe_name}-s{seed}-{config_hash[:8]}"


def reserve_output_directory(path: Path, *, resume: bool = False) -> Path:
    path = path.resolve()
    if path.exists() and any(path.iterdir()) and not resume:
        raise OutputExistsError(f"refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
