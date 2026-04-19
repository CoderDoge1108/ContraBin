"""Encoders used by ContraBin.

The paper uses a *single anchored encoder* :math:`f_M^a` to encode source code
and comments (parameters frozen), and a *trainable encoder* :math:`f_M^t` to
encode binary / IR (parameters updated). Both encoders are initialized from the
same HuggingFace checkpoint by default (``microsoft/graphcodebert-base``), but
the binary encoder may use a different backbone (for example ``GraphCodeBERT``
fine-tuned on LLVM IR).

This module also provides a lightweight ``TinyEncoder`` that does not require
HuggingFace weights, which is used by the test suite and the ``contrabin-tiny``
synthetic backbone name.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

try:
    from transformers import AutoModel

    _HAS_HF = True
except ImportError:  # pragma: no cover - transformers is a declared dependency
    _HAS_HF = False


# ---------------------------------------------------------------------------
# Tiny encoder for tests / offline smoke runs
# ---------------------------------------------------------------------------


class TinyEncoder(nn.Module):
    """A minimal embedding + transformer encoder used for tests.

    It avoids any network access and keeps the parameter count tiny so the full
    ContraBin training loop can be exercised on CPU in under a second.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        hidden_dim: int = 32,
        num_layers: int = 2,
        num_heads: int = 2,
        max_length: int = 512,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    @property
    def config(self) -> Any:
        class _Cfg:
            hidden_size = self.hidden_dim

        return _Cfg()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        hidden = self.encoder(x, src_key_padding_mask=key_padding_mask)

        class _Out:
            pass

        out = _Out()
        out.last_hidden_state = hidden
        return out


# ---------------------------------------------------------------------------
# Encoder wrappers
# ---------------------------------------------------------------------------


class _BaseEncoder(nn.Module):
    """Common pooling logic: CLS-token representation."""

    target_token_idx: int = 0

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # HF models return a ``BaseModelOutput`` which has ``.last_hidden_state``.
        hidden = output.last_hidden_state
        return hidden[:, self.target_token_idx, :]


class AnchoredEncoder(_BaseEncoder):
    """Frozen encoder for source code and comments.

    The paper calls this encoder *anchored* because its parameters are detached
    from the computation graph. Gradients from the binary-code branch therefore
    never update the text/source representations.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__(backbone)
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True) -> AnchoredEncoder:  # type: ignore[override]
        super().train(mode)
        self.backbone.eval()
        return self


class TrainableEncoder(_BaseEncoder):
    """Trainable encoder for binary / IR."""


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _build_backbone(name: str, hidden_dim: int) -> nn.Module:
    if name == "contrabin-tiny":
        return TinyEncoder(hidden_dim=hidden_dim)
    if not _HAS_HF:
        raise RuntimeError(
            "transformers is not installed; cannot load a HuggingFace backbone."
        )
    return AutoModel.from_pretrained(name)


def build_encoder(
    name: str,
    trainable: bool,
    hidden_dim: int = 768,
) -> _BaseEncoder:
    """Build an encoder by name.

    Parameters
    ----------
    name:
        HuggingFace model identifier, or ``"contrabin-tiny"`` for the offline
        test backbone.
    trainable:
        If ``True`` returns a :class:`TrainableEncoder`, otherwise an
        :class:`AnchoredEncoder`.
    hidden_dim:
        Used only when ``name == "contrabin-tiny"``.
    """
    backbone = _build_backbone(name, hidden_dim=hidden_dim)
    return TrainableEncoder(backbone) if trainable else AnchoredEncoder(backbone)
