"""YAML loading, deterministic interpolation, validation, and hashing."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from efficient_llm.core.paths import project_root
from efficient_llm.core.schemas import ProjectConfig

_REFERENCE = re.compile(r"\$\{([a-zA-Z0-9_.-]+)\}")


class ConfigError(ValueError):
    """Configuration is unreadable or violates the frozen protocol."""


@dataclass(frozen=True)
class LoadedConfig:
    source: Path
    model: ProjectConfig
    resolved: dict[str, Any]
    config_hash: str


def _comparison_view(config: LoadedConfig) -> dict[str, Any]:
    """Remove only differences explicitly allowed for LoRA/QLoRA fairness."""
    view = copy.deepcopy(config.resolved)
    training = view.get("training", {})
    training.pop("method", None)
    view.pop("quantization", None)
    run = view.get("run", {})
    run.pop("name", None)
    run.pop("output_dir", None)
    model = view.get("model", {})
    for field in ("load_in_4bit", "quantization_config"):
        model.pop(field, None)
    return view


def validate_fair_training_pair(left: LoadedConfig, right: LoadedConfig) -> None:
    methods = {left.model.training.method, right.model.training.method} if (
        left.model.training is not None and right.model.training is not None
    ) else set()
    if methods != {"lora", "qlora"}:
        raise ConfigError("fairness comparison requires one lora and one qlora config")
    left_view = _comparison_view(left)
    right_view = _comparison_view(right)
    if left_view != right_view:
        differing = sorted(
            key for key in set(left_view) | set(right_view)
            if left_view.get(key) != right_view.get(key)
        )
        raise ConfigError(f"unauthorized LoRA/QLoRA config differences: {differing}")


def _lookup(data: dict[str, Any], dotted: str) -> Any:
    current: Any = data
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ConfigError(f"unknown interpolation reference: ${{{dotted}}}")
        current = current[part]
    return current


def _resolve(value: Any, root: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _resolve(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item, root) for item in value]
    if isinstance(value, str):
        return _REFERENCE.sub(lambda match: str(_lookup(root, match.group(1))), value)
    return value


def load_config(path: str | Path) -> LoadedConfig:
    source = Path(path)
    if not source.is_absolute():
        source = project_root() / source
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot load config {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping: {source}")
    resolved = _resolve(raw, raw)
    try:
        model = ProjectConfig.model_validate(resolved)
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc
    canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return LoadedConfig(source=source.resolve(), model=model, resolved=resolved, config_hash=digest)


def config_as_yaml(config: LoadedConfig) -> str:
    return yaml.safe_dump(config.resolved, sort_keys=True, allow_unicode=True)
