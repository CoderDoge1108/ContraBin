r"""Primary (CLIP-style) contrastive loss used in Section 2.1 of the paper.

Given a batch of anchored embeddings :math:`H_{c/s} \in \mathbb{R}^{n \times d}`
(source or comment) and binary embeddings :math:`H_b \in \mathbb{R}^{n \times d}`:

.. math::
   \text{logits} = H_{c/s} H_b^{T}, \qquad P = \mathrm{softmax}(\text{logits}, -1)

   \mathrm{sim}_{c/s} = H_{c/s} H_{c/s}^{T}, \qquad \mathrm{sim}_b = H_b H_b^{T}

   \text{targets} = \mathrm{softmax}\!\left(\tfrac{\mathrm{sim}_{c/s} + \mathrm{sim}_b}{2}, -1\right)

   \mathcal{L} = \mathrm{CE}(P, \text{targets})

This is effectively a soft-label cross-entropy (Eq. 6).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def clip_style_loss(
    anchor: torch.Tensor, target: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Eqs. 2-6 of the paper, implemented symmetrically.

    Parameters
    ----------
    anchor, target:
        ``(B, D)`` tensors; cosine similarity is *not* used directly (the paper
        uses plain matrix multiplication of non-normalized projections), but the
        ``temperature`` argument scales the logits for numerical stability.
    """
    logits = anchor @ target.t() / temperature
    p = F.log_softmax(logits, dim=-1)

    sim_a = anchor @ anchor.t() / temperature
    sim_t = target @ target.t() / temperature
    targets = F.softmax((sim_a + sim_t) / 2.0, dim=-1)

    loss_a = -(targets * p).sum(dim=-1).mean()
    # Symmetric term (swap roles) stabilizes training.
    logits_t = logits.t()
    p_t = F.log_softmax(logits_t, dim=-1)
    loss_t = -(targets.t() * p_t).sum(dim=-1).mean()
    return 0.5 * (loss_a + loss_t)


class PrimaryContrastiveLoss(nn.Module):
    """Wrap :func:`clip_style_loss` as an ``nn.Module`` for composition."""

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return clip_style_loss(anchor, target, temperature=self.temperature)
