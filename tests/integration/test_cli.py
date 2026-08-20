from __future__ import annotations

import json
from pathlib import Path

import pytest

from efficient_llm.cli import main


@pytest.mark.integration
def test_config_command_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["train", "--config", "configs/train/lora_r8.yaml", "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "dry-run"
    assert payload["resolved_config"]["lora"]["rank"] == 8


@pytest.mark.integration
def test_placeholder_cannot_report_false_success() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["train", "--config", "configs/train/lora_r8.yaml"])
    assert exc.value.code == 4


@pytest.mark.integration
def test_doctor_writes_once(tmp_path: Path) -> None:
    output = tmp_path / "environment.json"
    assert main(["doctor", "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(report["gpu"]["available"], bool)
    with pytest.raises(SystemExit) as exc:
        main(["doctor", "--output", str(output)])
    assert exc.value.code == 2

