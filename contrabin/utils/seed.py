"""Deterministic seeding utilities."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Parameters
    ----------
    seed:
        Integer seed.
    deterministic:
        If True, force PyTorch to use deterministic algorithms. This can hurt
        throughput and should typically only be enabled when strict bit-for-bit
        reproducibility is required.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
