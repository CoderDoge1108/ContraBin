"""Contrastive objectives used by ContraBin."""

from contrabin.losses.contrastive import PrimaryContrastiveLoss, clip_style_loss
from contrabin.losses.intermediate import IntermediateContrastiveLoss, info_nce_loss

__all__ = [
    "IntermediateContrastiveLoss",
    "PrimaryContrastiveLoss",
    "clip_style_loss",
    "info_nce_loss",
]
