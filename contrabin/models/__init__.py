"""Model components for ContraBin."""

from contrabin.models.contrabin import ContraBinModel, ContraBinOutput
from contrabin.models.encoders import AnchoredEncoder, TrainableEncoder, build_encoder
from contrabin.models.heads import LinearProjectionHead, NonLinearProjectionHead, build_head
from contrabin.models.interpolation import (
    LinearSimplexInterpolator,
    NonLinearSimplexInterpolator,
    simplex_interpolate,
)

__all__ = [
    "AnchoredEncoder",
    "ContraBinModel",
    "ContraBinOutput",
    "LinearProjectionHead",
    "LinearSimplexInterpolator",
    "NonLinearProjectionHead",
    "NonLinearSimplexInterpolator",
    "TrainableEncoder",
    "build_encoder",
    "build_head",
    "simplex_interpolate",
]
