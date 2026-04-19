"""ContraBin: contrastive pre-training of binary code representations.

Reference
---------
Zhang, Y., Huang, C., Zhang, Y., Shao, H., Leach, K., & Huang, Y. (2025).
Pre-Training Representations of Binary Code Using Contrastive Learning.
Transactions on Machine Learning Research. arXiv:2210.05102.

This package exposes the building blocks of the ContraBin framework:

* :mod:`contrabin.config`        - Pydantic/dataclass configs.
* :mod:`contrabin.data`          - Triplet data pipeline (source / IR / comment).
* :mod:`contrabin.models`        - Anchored/trainable encoders, projection heads,
                                   simplex interpolation, and the composite model.
* :mod:`contrabin.losses`        - Primary (CLIP-style) and intermediate (InfoNCE)
                                   contrastive objectives.
* :mod:`contrabin.training`      - Curriculum-aware trainer (naive -> linear -> non-linear).
* :mod:`contrabin.tasks`         - Four downstream tasks from the paper.
* :mod:`contrabin.evaluation`    - Metric implementations and an evaluator.
"""

from __future__ import annotations

from contrabin._version import __version__
from contrabin.config import (
    ContraBinConfig,
    DataConfig,
    ModelConfig,
    OptimConfig,
    TrainingConfig,
)
from contrabin.models.contrabin import ContraBinModel

__all__ = [
    "ContraBinConfig",
    "ContraBinModel",
    "DataConfig",
    "ModelConfig",
    "OptimConfig",
    "TrainingConfig",
    "__version__",
]
