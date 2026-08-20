from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from efficient_llm.core.reproducibility import (
    OutputExistsError,
    atomic_write_json,
    make_run_id,
    reserve_output_directory,
)


def test_run_id_is_deterministic_for_fixed_time() -> None:
    now = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)
    assert make_run_id("LoRA r8", 42, "abcdef012345", now) == (
        "20260820T120000Z-LoRA-r8-s42-abcdef01"
    )


def test_non_empty_output_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "evidence.json").write_text("first", encoding="utf-8")
    with pytest.raises(OutputExistsError, match="refusing to overwrite"):
        reserve_output_directory(output)
    assert (output / "evidence.json").read_text(encoding="utf-8") == "first"


def test_atomic_json_write(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "value.json"
    atomic_write_json(destination, {"value": 7})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"value": 7}

