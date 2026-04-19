"""Composite ContraBin model.

The :class:`ContraBinModel` wraps:

* an :class:`~contrabin.models.encoders.AnchoredEncoder` for source code and
  comments (frozen),
* a :class:`~contrabin.models.encoders.TrainableEncoder` for binary / IR
  (updated during training),
* three :class:`~contrabin.models.heads.LinearProjectionHead` /
  :class:`~contrabin.models.heads.NonLinearProjectionHead` projection heads,
  one per modality, and
* a :class:`~contrabin.models.interpolation.SimplexInterpolationModule`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from contrabin.config import ModelConfig
from contrabin.models.encoders import build_encoder
from contrabin.models.heads import build_head
from contrabin.models.interpolation import SimplexInterpolationModule

AnchorModality = Literal["source", "comment", "binary"]
InterpolationStage = Literal["naive", "linear", "nonlinear"]


@dataclass
class ContraBinOutput:
    """Container of intermediate tensors produced by :meth:`ContraBinModel.forward`."""

    source: torch.Tensor
    binary: torch.Tensor
    comment: torch.Tensor
    intermediate: torch.Tensor | None = None


class ContraBinModel(nn.Module):
    """End-to-end ContraBin model used for pre-training.

    Parameters
    ----------
    config:
        A :class:`~contrabin.config.ModelConfig` instance.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        anchor_name = config.encoder_name
        binary_name = config.binary_encoder_name or config.encoder_name

        self.anchored_encoder = build_encoder(
            anchor_name, trainable=False, hidden_dim=config.hidden_dim
        )
        self.binary_encoder = build_encoder(
            binary_name, trainable=True, hidden_dim=config.hidden_dim
        )

        self.source_head = build_head(
            config.head_type, config.hidden_dim, config.projection_dim, config.dropout
        )
        self.comment_head = build_head(
            config.head_type, config.hidden_dim, config.projection_dim, config.dropout
        )
        self.binary_head = build_head(
            config.head_type, config.hidden_dim, config.projection_dim, config.dropout
        )
        self.interpolation = SimplexInterpolationModule(
            projection_dim=config.projection_dim, dropout=config.dropout
        )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode_source(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.config.stop_gradient_on_anchor:
            with torch.no_grad():
                h = self.anchored_encoder(input_ids, attention_mask)
        else:
            h = self.anchored_encoder(input_ids, attention_mask)
        return self.source_head(h)

    def encode_comment(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.config.stop_gradient_on_anchor:
            with torch.no_grad():
                h = self.anchored_encoder(input_ids, attention_mask)
        else:
            h = self.anchored_encoder(input_ids, attention_mask)
        return self.comment_head(h)

    def encode_binary(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.binary_encoder(input_ids, attention_mask)
        return self.binary_head(h)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        batch: dict[str, dict[str, torch.Tensor]],
        stage: InterpolationStage = "naive",
        anchor: AnchorModality = "source",
    ) -> ContraBinOutput:
        """Encode a triplet batch.

        Parameters
        ----------
        batch:
            Mapping with keys ``source``, ``binary``, ``comment`` mapping to
            dicts of ``input_ids`` / ``attention_mask``.
        stage:
            Interpolation stage: ``naive`` | ``linear`` | ``nonlinear``.
        anchor:
            Which modality plays the role of the positive anchor during
            intermediate contrastive learning. Binary is always the "target"
            representation being refined.
        """
        source = self.encode_source(**batch["source"])
        comment = self.encode_comment(**batch["comment"])
        binary = self.encode_binary(**batch["binary"])

        intermediate: torch.Tensor | None = None
        if stage != "naive":
            # Interpolate the two anchored modalities (source+comment) to form
            # an "intermediate view" of the same program, then align it with
            # the binary embedding via the intermediate contrastive loss.
            if anchor == "source":
                intermediate = self.interpolation(source, comment, stage)
            elif anchor == "comment":
                intermediate = self.interpolation(comment, source, stage)
            else:  # anchor == "binary"
                intermediate = self.interpolation(source, comment, stage)

        return ContraBinOutput(
            source=source, binary=binary, comment=comment, intermediate=intermediate
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def binary_embedding(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Produce the post-pretraining binary embedding used by downstream tasks."""
        return self.encode_binary(input_ids, attention_mask)
