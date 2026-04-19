"""Curriculum and learning-rate schedulers."""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch

from contrabin.config import CurriculumConfig, OptimConfig


class CurriculumScheduler:
    """Maps the current epoch to an interpolation stage.

    See :class:`~contrabin.config.CurriculumConfig`.
    """

    def __init__(self, cfg: CurriculumConfig) -> None:
        self.cfg = cfg
        self._epoch = 0

    def step(self) -> None:
        self._epoch += 1

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def stage(self) -> str:
        return self.cfg.stage_for_epoch(self._epoch)

    def __iter__(self) -> Iterator[str]:
        for e in range(self.cfg.total_epochs()):
            yield self.cfg.stage_for_epoch(e)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer, cfg: OptimConfig, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Return a warmup + (cosine|linear|constant) scheduler."""
    warmup = cfg.warmup_steps

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        if cfg.scheduler == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if cfg.scheduler == "linear":
            return max(0.0, 1.0 - progress)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
