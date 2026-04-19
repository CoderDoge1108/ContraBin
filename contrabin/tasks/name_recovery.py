"""DIRE function name recovery.

We model function-name recovery as a *multi-label classification* over a fixed
vocabulary of name sub-tokens (the paper's Table 7 details this). A linear
decoder produces logits of size ``|vocab|``, trained with BCE against the set
of sub-tokens that appear in the ground-truth name (split on underscores and
camel-case boundaries).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from contrabin.evaluation.metrics import exact_match
from contrabin.models.contrabin import ContraBinModel

_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")


def split_function_name(name: str) -> list[str]:
    """Split ``MyFancyName_v2`` -> ``['my', 'fancy', 'name', 'v2']``."""
    parts = _CAMEL_RE.split(name)
    flat: list[str] = []
    for p in parts:
        flat.extend(p.split("_"))
    return [t.lower() for t in flat if t]


class NameRecoveryModel(nn.Module):
    """Binary-embedding -> multi-label vocabulary classifier."""

    def __init__(self, backbone: ContraBinModel, subtoken_vocab: list[str]) -> None:
        super().__init__()
        self.backbone = backbone
        self.vocab = subtoken_vocab
        self.token2id = {t: i for i, t in enumerate(subtoken_vocab)}
        hidden = backbone.config.projection_dim
        self.head = nn.Linear(hidden, len(subtoken_vocab))

    def encode_labels(self, names: Iterable[str]) -> torch.Tensor:
        names = list(names)
        labels = torch.zeros(max(1, len(names)), len(self.vocab))
        for i, n in enumerate(names):
            for t in split_function_name(n):
                if t in self.token2id:
                    labels[i, self.token2id[t]] = 1.0
        return labels

    def decode(self, logits: torch.Tensor, threshold: float = 0.5) -> list[str]:
        probs = torch.sigmoid(logits)
        out: list[str] = []
        for row in probs:
            idx = (row > threshold).nonzero(as_tuple=False).flatten().tolist()
            tokens = [self.vocab[i] for i in idx]
            out.append("_".join(tokens) if tokens else "unknown")
        return out

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        emb = self.backbone.binary_embedding(input_ids, attention_mask)
        return self.head(emb)


@dataclass
class NameRecoveryResult:
    exact_match: float
    subtoken_f1: float


def train_name_recovery(
    model: NameRecoveryModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    num_epochs: int = 5,
    lr: float = 2e-5,
    head_lr: float = 1e-3,
    device: str = "cpu",
) -> list[dict[str, float]]:
    model.to(device)
    optim = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": lr},
            {"params": model.head.parameters(), "lr": head_lr},
        ]
    )
    history: list[dict[str, float]] = []
    for epoch in range(num_epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            ids = batch["binary"]["input_ids"].to(device)
            am = batch["binary"]["attention_mask"].to(device)
            names = [m["name"] for m in batch["metadata"]]
            labels = model.encode_labels(names).to(device)
            logits = model(ids, am)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
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
def _evaluate(model: NameRecoveryModel, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    hyps: list[str] = []
    refs: list[str] = []
    f1_scores: list[float] = []
    for batch in loader:
        ids = batch["binary"]["input_ids"].to(device)
        am = batch["binary"]["attention_mask"].to(device)
        logits = model(ids, am)
        pred_names = model.decode(logits)
        true_names = [m["name"] for m in batch["metadata"]]
        hyps.extend(pred_names)
        refs.extend(true_names)
        for p, t in zip(pred_names, true_names, strict=False):
            ps = set(split_function_name(p))
            ts = set(split_function_name(t))
            if not ps and not ts:
                f1_scores.append(1.0)
                continue
            if not ps or not ts:
                f1_scores.append(0.0)
                continue
            inter = ps & ts
            prec = len(inter) / len(ps)
            rec = len(inter) / len(ts)
            f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
            f1_scores.append(f1)
    return {
        "exact_match": exact_match(hyps, refs),
        "subtoken_f1": sum(f1_scores) / max(1, len(f1_scores)),
    }


__all__ = ["NameRecoveryModel", "NameRecoveryResult", "split_function_name", "train_name_recovery"]
