r"""Intermediate contrastive loss (InfoNCE) from Section 2.2.

.. math::
   Z_i = \sum_{h' \in B'} \exp(\mathrm{sim}(h_i, h') / \tau)

   \mathcal{L}_{\text{inter}} = -\frac{1}{N}\sum_{i=1}^{N}
      \log\!\frac{\exp(\mathrm{sim}(h_i, h_b^{(i)}) / \tau)}
                 {\exp(\mathrm{sim}(h_i, h_b^{(i)}) / \tau) + Z_i}

where :math:`h_i` is the interpolated intermediate representation, :math:`h_b^{(i)}`
is the matched binary embedding, :math:`B'` is the set of mismatched in-batch
negatives, and :math:`\tau` is a temperature.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def info_nce_loss(
    intermediate: torch.Tensor,
    binary: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE between interpolated intermediate and binary embeddings.

    Both ``intermediate`` and ``binary`` have shape ``(B, D)``. This formulation
    uses in-batch negatives: row ``i`` of ``intermediate`` is aligned with row
    ``i`` of ``binary``; all other binary rows in the batch act as negatives.
    """
    inter = F.normalize(intermediate, dim=-1)
    bin_ = F.normalize(binary, dim=-1)
    logits = inter @ bin_.t() / temperature
    targets = torch.arange(inter.size(0), device=inter.device)
    # Symmetric InfoNCE (CLIP-style) for robustness.
    loss_i2b = F.cross_entropy(logits, targets)
    loss_b2i = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_i2b + loss_b2i)


class IntermediateContrastiveLoss(nn.Module):
    """Module wrapper around :func:`info_nce_loss`."""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, intermediate: torch.Tensor, binary: torch.Tensor) -> torch.Tensor:
        return info_nce_loss(intermediate, binary, temperature=self.temperature)
