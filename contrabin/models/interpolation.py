r"""Simplex interpolation modules.

The paper defines a generic interpolation operator
:math:`\Gamma(A, B; \lambda) = \lambda \cdot A + (1 - \lambda) \cdot B`
used in three regimes:

* **naive**      : no interpolation, pick either ``A`` or ``B``. (Primary stage.)
* **linear**     : :math:`\lambda_l \in [0, 1]` is a *scalar* predicted from
  ``H1 + H2`` by a small neural network (:class:`LinearSimplexInterpolator`).
* **non-linear** : :math:`\lambda_n` has the *same shape* as ``H1`` and ``H2``
  (element-wise interpolation), predicted by a deeper network
  (:class:`NonLinearSimplexInterpolator`).

See Equations 7-11 of Zhang et al. (TMLR 2025).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from contrabin.models.heads import LinearProjectionHead, NonLinearProjectionHead


def simplex_interpolate(a: torch.Tensor, b: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    r""":math:`\Gamma(A, B; \lambda) = \lambda \cdot A + (1 - \lambda) \cdot B`.

    ``lam`` may be a scalar (broadcasts) or a tensor with the same shape as
    ``a`` / ``b`` for element-wise interpolation.
    """
    return lam * a + (1.0 - lam) * b


class LinearSimplexInterpolator(nn.Module):
    """Learns a *scalar* interpolation index :math:`\\lambda_l \\in [0, 1]`.

    Input: ``H1 + H2`` with shape ``(B, D)``.
    Output: per-example scalar ``lambda`` with shape ``(B, 1)``.
    """

    def __init__(self, projection_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.head = LinearProjectionHead(projection_dim, projection_dim, dropout)
        self.gate = nn.Linear(projection_dim, 1)

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        h = self.head(h1 + h2)
        lam = torch.sigmoid(self.gate(h))
        return simplex_interpolate(h1, h2, lam)


class NonLinearSimplexInterpolator(nn.Module):
    """Learns an *element-wise* interpolation mask :math:`\\lambda_n`.

    Input: ``H1 + H2`` with shape ``(B, D)``.
    Output: per-feature ``lambda`` with shape ``(B, D)``.
    """

    def __init__(self, projection_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.head = NonLinearProjectionHead(projection_dim, projection_dim, dropout)
        self.gate = nn.Linear(projection_dim, projection_dim)

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        h = self.head(h1 + h2)
        lam = torch.sigmoid(self.gate(h))
        return simplex_interpolate(h1, h2, lam)


InterpolationStage = Literal["naive", "linear", "nonlinear"]


class SimplexInterpolationModule(nn.Module):
    """Container that holds both linear and non-linear interpolators.

    The ``stage`` argument in :meth:`forward` selects which behavior to apply.
    """

    def __init__(self, projection_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear = LinearSimplexInterpolator(projection_dim, dropout)
        self.nonlinear = NonLinearSimplexInterpolator(projection_dim, dropout)

    def forward(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
        stage: InterpolationStage,
    ) -> torch.Tensor:
        if stage == "naive":
            # For the naive stage we simply return h1 (caller is expected to
            # have picked the modality); this keeps the API uniform.
            return h1
        if stage == "linear":
            return self.linear(h1, h2)
        if stage == "nonlinear":
            return self.nonlinear(h1, h2)
        raise ValueError(f"Unknown interpolation stage: {stage!r}")
