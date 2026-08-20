from __future__ import annotations

import pytest

from efficient_llm.core import hardware


def test_doctor_handles_machine_without_nvidia(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware.shutil, "which", lambda _name: None)
    report = hardware.collect_environment()
    assert report["gpu"]["available"] is False
    assert report["python"]["version"]
