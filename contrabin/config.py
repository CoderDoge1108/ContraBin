"""Configuration objects for ContraBin.

Configs are defined as Pydantic models so they can be:

* loaded from YAML/JSON files (see :func:`load_config`),
* validated at construction time,
* and serialized to disk for reproducible experiments.

Design note
-----------
The original Zenodo release used a single monolithic ``Config`` class with class
variables. We split the configuration into four orthogonal groups so that, e.g.,
data-only changes do not force re-specification of optimizer hyper-parameters
when reusing a config for a new task.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

InterpolationStage = Literal["naive", "linear", "nonlinear"]
HeadType = Literal["linear", "nonlinear"]
AnchorModality = Literal["source", "comment", "binary"]


class _BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class DataConfig(_BaseConfig):
    """Configuration for triplet data construction and loading."""

    dataset_name: str = "anghabench-sample"
    """Name of the dataset as registered in :mod:`contrabin.data.datasets`."""

    train_path: Path = Path("data/processed/train.jsonl")
    val_path: Path = Path("data/processed/val.jsonl")
    test_path: Path | None = None

    source_max_length: int = 512
    binary_max_length: int = 512
    comment_max_length: int = 64

    use_llm_comments: bool = True
    """Paper finding: LLM-generated comments help, human-written ones hurt."""

    num_workers: int = 4
    shuffle_buffer: int = 1024


class ModelConfig(_BaseConfig):
    """Configuration for encoders, projection heads, and interpolation."""

    encoder_name: str = "microsoft/graphcodebert-base"
    """HuggingFace identifier of the backbone used for all three encoders."""

    binary_encoder_name: str | None = None
    """Optional separate backbone for the binary/IR encoder (defaults to ``encoder_name``)."""

    hidden_dim: int = 768
    projection_dim: int = 256
    head_type: HeadType = "linear"
    dropout: float = 0.1

    temperature: float = 0.1
    """Softmax temperature for the InfoNCE-style intermediate loss."""

    stop_gradient_on_anchor: bool = True
    """Whether to detach gradients on the anchored (source/comment) encoder."""

    @field_validator("dropout")
    @classmethod
    def _check_dropout(cls, v: float) -> float:
        if not 0.0 <= v < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        return v


class OptimConfig(_BaseConfig):
    """Optimizer and scheduler configuration."""

    lr: float = 1e-5
    head_lr: float = 1e-3
    weight_decay: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    scheduler: Literal["cosine", "linear", "constant"] = "cosine"


class CurriculumConfig(_BaseConfig):
    """Curriculum scheduler for the two-stage contrastive training.

    The paper uses:

    * ``naive`` primary contrastive learning for the first ``primary_epochs`` epochs
      (roughly 10 in the original implementation),
    * followed by ``linear`` simplex interpolation, and
    * then ``nonlinear`` simplex interpolation for the remaining epochs.
    """

    primary_epochs: int = 10
    linear_epochs: int = 5
    nonlinear_epochs: int = 5

    def total_epochs(self) -> int:
        return self.primary_epochs + self.linear_epochs + self.nonlinear_epochs

    def stage_for_epoch(self, epoch: int) -> InterpolationStage:
        """Return the interpolation stage for a 0-indexed epoch."""
        if epoch < self.primary_epochs:
            return "naive"
        if epoch < self.primary_epochs + self.linear_epochs:
            return "linear"
        return "nonlinear"


class TrainingConfig(_BaseConfig):
    """Top-level training loop configuration."""

    batch_size: int = 32
    eval_batch_size: int = 64
    seed: int = 42
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    mixed_precision: Literal["no", "fp16", "bf16"] = "no"
    grad_accumulation_steps: int = 1
    log_every_n_steps: int = 50
    eval_every_n_steps: int = 500
    save_every_n_steps: int = 1000
    output_dir: Path = Path("outputs/contrabin")
    resume_from: Path | None = None
    early_stopping_patience: int | None = None
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)


class ContraBinConfig(_BaseConfig):
    """Composite configuration used by the Trainer."""

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    experiment_name: str = "contrabin"

    # ------------------------------------------------------------------
    # serialization helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def save_yaml(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path | str) -> ContraBinConfig:
        with Path(path).open() as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)


def load_config(path: Path | str) -> ContraBinConfig:
    """Load a :class:`ContraBinConfig` from a YAML (``.yaml``/``.yml``) or JSON file."""
    return ContraBinConfig.from_yaml(path)
