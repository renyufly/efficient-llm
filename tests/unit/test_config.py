from __future__ import annotations

from pathlib import Path

import pytest

from efficient_llm.core.config import ConfigError, load_config, validate_fair_training_pair


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_all_checked_in_yaml_configs_validate() -> None:
    paths = sorted(Path("configs").rglob("*.yaml"))
    assert paths
    for path in paths:
        loaded = load_config(path)
        assert len(loaded.config_hash) == 64
        assert "${" not in str(loaded.resolved)


@pytest.mark.parametrize(
    ("fragment", "message"),
    [
        ("rank: 32\n    alpha: 64", "lora.rank"),
        ("rank: 8\n    alpha: 8", "alpha / lora.rank"),
    ],
)
def test_invalid_lora_config_fails(tmp_path: Path, fragment: str, message: str) -> None:
    path = _write(
        tmp_path,
        f"""schema_version: 1
run: {{name: invalid, seed: 42, output_dir: artifacts/invalid, overwrite: false}}
model: {{id: model, revision: commit}}
training:
  method: lora
  micro_batch_size: 1
  gradient_accumulation_steps: 32
  effective_batch_size: 32
lora:
    {fragment}
""",
    )
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_invalid_effective_batch_fails(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """schema_version: 1
run: {name: invalid, seed: 42, output_dir: artifacts/invalid, overwrite: false}
training:
  method: lora
  micro_batch_size: 2
  gradient_accumulation_steps: 8
  effective_batch_size: 32
lora: {rank: 8, alpha: 16}
""",
    )
    with pytest.raises(ConfigError, match="effective_batch_size"):
        load_config(path)


def test_registry_model_requires_revision(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """schema_version: 1
run: {name: invalid, seed: 42, output_dir: artifacts/invalid, overwrite: false}
model: {id: Qwen/example}
""",
    )
    with pytest.raises(ConfigError, match=r"model\.revision"):
        load_config(path)


def test_quality_and_benchmark_eos_are_distinct(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """schema_version: 1
run: {name: invalid, seed: 42, output_dir: artifacts/invalid, overwrite: false}
generation: {ignore_eos: true}
""",
    )
    with pytest.raises(ConfigError, match=r"generation\.ignore_eos"):
        load_config(path)


def test_lora_and_qlora_rank_matched_configs_are_fair() -> None:
    validate_fair_training_pair(
        load_config("configs/train/lora_r8.yaml"),
        load_config("configs/train/qlora_r8.yaml"),
    )


def test_fairness_check_rejects_unapproved_difference(tmp_path: Path) -> None:
    qlora_text = Path("configs/train/qlora_r8.yaml").read_text(encoding="utf-8")
    modified = tmp_path / "qlora.yaml"
    modified.write_text(
        qlora_text.replace("warmup_ratio: 0.03", "warmup_ratio: 0.1"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="unauthorized"):
        validate_fair_training_pair(
            load_config("configs/train/lora_r8.yaml"), load_config(modified)
        )
