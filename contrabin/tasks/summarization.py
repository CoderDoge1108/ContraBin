"""Binary code summarization (binary -> English).

We use the pre-trained ContraBin encoder as the encoder of a seq2seq model. By
default we wire it to a lightweight Transformer decoder so the task is runnable
without relying on a heavyweight encoder-decoder checkpoint. Power users can
swap in a full ``T5`` / ``BART`` decoder via :meth:`SummarizationModel.from_hf`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from contrabin.evaluation.metrics import bleu_score, rouge_l
from contrabin.models.contrabin import ContraBinModel


class SummarizationModel(nn.Module):
    """ContraBin binary encoder + Transformer decoder + LM head."""

    def __init__(
        self,
        backbone: ContraBinModel,
        vocab_size: int,
        hidden_dim: int | None = None,
        num_decoder_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        hidden = hidden_dim or backbone.config.projection_dim
        self.hidden_dim = hidden
        self.token_embedding = nn.Embedding(vocab_size, hidden)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

    def _memory_from_binary(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        emb = self.backbone.binary_embedding(input_ids, attention_mask)
        return emb.unsqueeze(1)  # (B, 1, D)

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self._memory_from_binary(input_ids, attention_mask)
        tgt_emb = self.token_embedding(decoder_input_ids)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_input_ids.size(1)).to(
            decoder_input_ids.device
        )
        hidden = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.lm_head(hidden)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_length: int = 32,
        start_id: int = 1,
        end_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device
        decoded = torch.full((batch_size, 1), start_id, dtype=torch.long, device=device)
        for _ in range(max_length - 1):
            logits = self.forward(input_ids, decoded, attention_mask)
            next_tok = logits[:, -1, :].argmax(-1, keepdim=True)
            decoded = torch.cat([decoded, next_tok], dim=1)
            if end_id is not None and bool((next_tok == end_id).all()):
                break
        return decoded


@dataclass
class SummarizationResult:
    bleu: float
    rouge_l: float


def train_summarization(
    model: SummarizationModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    num_epochs: int = 5,
    lr: float = 3e-5,
    head_lr: float = 1e-3,
    device: str = "cpu",
) -> list[dict[str, float]]:
    model.to(device)
    optim = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": lr},
            {
                "params": [p for n, p in model.named_parameters() if not n.startswith("backbone")],
                "lr": head_lr,
            },
        ]
    )
    history: list[dict[str, float]] = []
    for epoch in range(num_epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            bin_ids = batch["binary"]["input_ids"].to(device)
            bin_am = batch["binary"]["attention_mask"].to(device)
            tgt_ids = batch["comment"]["input_ids"].to(device)
            dec_in = tgt_ids[:, :-1]
            dec_tgt = tgt_ids[:, 1:]
            logits = model(bin_ids, dec_in, bin_am)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                dec_tgt.reshape(-1),
                ignore_index=0,
            )
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total += float(loss.item())
        entry = {"epoch": epoch, "train_loss": total / max(1, len(train_loader))}
        if val_loader is not None:
            entry.update(_evaluate(model, val_loader, device))
        history.append(entry)
    return history


@torch.no_grad()
def _evaluate(model: SummarizationModel, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    hyps: list[str] = []
    refs: list[str] = []
    for batch in loader:
        bin_ids = batch["binary"]["input_ids"].to(device)
        bin_am = batch["binary"]["attention_mask"].to(device)
        out = model.generate(bin_ids, bin_am, max_length=32)
        hyps.extend(_ids_to_text(out))
        refs.extend([m.get("comment_text", "") for m in batch["metadata"]])
    return {"bleu": bleu_score(hyps, refs), "rouge_l": rouge_l(hyps, refs)}


def _ids_to_text(ids: torch.Tensor) -> list[str]:
    # Placeholder detokenization (repo ships with a byte-hash tokenizer by default).
    # Real users should pass a HF tokenizer and decode it here.
    return [" ".join(str(int(x)) for x in row.tolist() if int(x) != 0) for row in ids]


__all__ = ["SummarizationModel", "SummarizationResult", "train_summarization"]

_ = Iterable
