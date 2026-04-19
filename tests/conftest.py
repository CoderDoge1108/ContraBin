"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from contrabin.config import (
    ContraBinConfig,
    CurriculumConfig,
    DataConfig,
    ModelConfig,
    OptimConfig,
    TrainingConfig,
)
from contrabin.data.datasets import build_synthetic_triplets


@pytest.fixture
def tiny_config() -> ContraBinConfig:
    """Config that uses the offline ``contrabin-tiny`` backbone."""
    return ContraBinConfig(
        data=DataConfig(
            dataset_name="synthetic",
            source_max_length=32,
            binary_max_length=32,
            comment_max_length=12,
            num_workers=0,
        ),
        model=ModelConfig(
            encoder_name="contrabin-tiny",
            hidden_dim=32,
            projection_dim=16,
            dropout=0.0,
            temperature=0.1,
        ),
        optim=OptimConfig(lr=1e-3, head_lr=1e-3, warmup_steps=0, scheduler="constant"),
        training=TrainingConfig(
            batch_size=4,
            eval_batch_size=4,
            device="cpu",
            log_every_n_steps=100,
            curriculum=CurriculumConfig(primary_epochs=1, linear_epochs=1, nonlinear_epochs=1),
        ),
    )


@pytest.fixture
def tiny_triplets(tmp_path: Path) -> tuple[Path, Path]:
    """Generate synthetic train / val JSONL files."""
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    build_synthetic_triplets(train, n=16, seed=0)
    build_synthetic_triplets(val, n=8, seed=1)
    return train, val
