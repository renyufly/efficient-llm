"""Strong validation for the frozen experiment invariants."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExtensibleModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class RunConfig(ExtensibleModel):
    name: str = Field(min_length=1)
    seed: int
    output_dir: str = Field(min_length=1)
    overwrite: Literal[False] = False


class ModelConfig(ExtensibleModel):
    id: str | None = None
    revision: str | None = None

    @model_validator(mode="after")
    def require_revision_for_registry_model(self) -> ModelConfig:
        if self.id is not None and not self.revision:
            raise ValueError("model.revision is required when model.id is set")
        return self


class TrainingConfig(ExtensibleModel):
    method: Literal["lora", "qlora"]
    micro_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    effective_batch_size: int = Field(gt=0)
    world_size: int = Field(default=1, gt=0)

    @model_validator(mode="after")
    def effective_batch_is_exact(self) -> TrainingConfig:
        actual = self.micro_batch_size * self.gradient_accumulation_steps * self.world_size
        if self.effective_batch_size != actual:
            raise ValueError(
                "training.effective_batch_size must equal micro_batch_size * "
                f"gradient_accumulation_steps * world_size ({actual})"
            )
        return self


class LoraConfig(ExtensibleModel):
    rank: Literal[8, 16]
    alpha: int = Field(gt=0)

    @model_validator(mode="after")
    def alpha_ratio_is_frozen(self) -> LoraConfig:
        if self.alpha != self.rank * 2:
            raise ValueError("lora.alpha / lora.rank must equal 2")
        return self


class GenerationConfig(ExtensibleModel):
    ignore_eos: Literal[False] = False


class WorkloadConfig(ExtensibleModel):
    input_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=0)
    ignore_eos: Literal[True]


class ProjectConfig(ExtensibleModel):
    schema_version: Literal[1]
    run: RunConfig
    model: ModelConfig | None = None
    training: TrainingConfig | None = None
    lora: LoraConfig | None = None
    generation: GenerationConfig | None = None
    workload: WorkloadConfig | None = None

    @model_validator(mode="after")
    def training_sections_are_consistent(self) -> ProjectConfig:
        if (self.training is None) != (self.lora is None):
            raise ValueError("training and lora sections must be provided together")
        extra: dict[str, Any] = self.model_extra or {}
        data = extra.get("data")
        if isinstance(data, dict) and data.get("packing") is True:
            if not data.get("packing_correctness_report"):
                raise ValueError(
                    "data.packing=true requires data.packing_correctness_report"
                )
        return self

