"""Binary functional-similarity retrieval.

Why this task
-------------
The ContraBin paper (TMLR 2025) tests "algorithmic functionality classification"
on POJ-104: a 104-way softmax over program categories. That framing confounds
two things a contrastive model actually optimizes for:

1. *Representation quality* - do semantically equivalent binaries cluster
   together in the embedding space?
2. *Supervised head capacity* - how much can a linear classifier reshape the
   embedding space at fine-tuning time?

A 104-way cross-entropy objective rewards (2) at the expense of (1), and the
numbers reported in the paper are hard to compare with embedding-only baselines
such as Asm2Vec, jTrans, or SAFE.

The retrieval reformulation used here is closer to what a practitioner actually
wants from a binary representation: *given a query binary, find its functional
twins in a large gallery of candidates*. No classifier is trained on top of the
embeddings; we evaluate the post-pretraining binary encoder directly.

Metrics
-------
* **mAP** - mean average precision over all queries (matches the original
  POJ-104 retrieval protocol).
* **Recall@k** - fraction of queries whose top-``k`` nearest neighbors contain
  at least one positive (``k in {1, 5, 10}``).
* **MRR** - mean reciprocal rank of the first positive.

Optional linear probe
---------------------
For comparison with the paper's numbers, an optional
:class:`LinearProbeClassifier` trains a frozen-backbone linear classifier on
the gallery labels. This mirrors the ``linear probe`` protocol commonly used
when evaluating self-supervised vision models and is *cheaper and fairer* than
fully fine-tuning the backbone.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from contrabin.evaluation.metrics import (
    mean_average_precision,
    mean_reciprocal_rank,
    recall_at_k,
)
from contrabin.models.contrabin import ContraBinModel

# ---------------------------------------------------------------------------
# Gallery embedding extraction
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    mAP: float  # noqa: N815 - mAP is the canonical metric name
    mrr: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float


@torch.no_grad()
def extract_binary_embeddings(
    model: ContraBinModel,
    loader: DataLoader,
    device: str = "cpu",
    label_key: str = "label",
) -> tuple[torch.Tensor, list[int]]:
    """Run the binary encoder over a DataLoader and return (embeddings, labels).

    The dataloader is expected to yield batches produced by
    :class:`~contrabin.data.loaders.TripletCollator`; labels are read from the
    ``metadata`` field of each record.
    """
    model.eval().to(device)
    embs: list[torch.Tensor] = []
    labels: list[int] = []
    for batch in loader:
        ids = batch["binary"]["input_ids"].to(device)
        am = batch["binary"]["attention_mask"].to(device)
        emb = model.binary_embedding(ids, am)
        embs.append(emb.detach().cpu())
        labels.extend(int(m[label_key]) for m in batch["metadata"])
    return torch.cat(embs, dim=0), labels


def evaluate_retrieval(embeddings: torch.Tensor, labels: list[int]) -> RetrievalResult:
    """Compute mAP / MRR / Recall@k from the embedding matrix and labels."""
    if embeddings.numel() == 0:
        return RetrievalResult(0.0, 0.0, 0.0, 0.0, 0.0)
    emb = F.normalize(embeddings, dim=-1)
    sim = emb @ emb.t()
    sim.fill_diagonal_(-float("inf"))
    return RetrievalResult(
        mAP=mean_average_precision(embeddings, labels),
        mrr=mean_reciprocal_rank(sim, labels),
        recall_at_1=recall_at_k(sim, labels, k=1),
        recall_at_5=recall_at_k(sim, labels, k=5),
        recall_at_10=recall_at_k(sim, labels, k=10),
    )


# ---------------------------------------------------------------------------
# Optional: linear probe (for head-to-head comparison with the paper)
# ---------------------------------------------------------------------------


class LinearProbeClassifier(nn.Module):
    """Linear classifier on *frozen* ContraBin binary embeddings.

    Only the single linear layer is learned; the backbone is kept in
    :meth:`torch.nn.Module.eval` mode with ``requires_grad=False``.
    """

    def __init__(self, backbone: ContraBinModel, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        hidden = backbone.config.projection_dim
        self.head = nn.Linear(hidden, num_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        self.backbone.eval()
        with torch.no_grad():
            emb = self.backbone.binary_embedding(input_ids, attention_mask)
        return self.head(emb)


def train_linear_probe(
    model: LinearProbeClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    *,
    num_epochs: int = 10,
    head_lr: float = 1e-2,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    label_key: str = "label",
) -> list[dict[str, float]]:
    """Fit the linear probe on top of frozen embeddings."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=head_lr, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    for epoch in range(num_epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            ids = batch["binary"]["input_ids"].to(device)
            am = batch["binary"]["attention_mask"].to(device)
            labels = torch.tensor([int(m[label_key]) for m in batch["metadata"]], device=device)
            logits = model(ids, am)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        entry = {"epoch": epoch, "train_loss": total / max(1, len(train_loader))}
        if val_loader is not None:
            entry.update(_eval_probe(model, val_loader, device, label_key))
        history.append(entry)
    return history


@torch.no_grad()
def _eval_probe(
    model: LinearProbeClassifier, loader: DataLoader, device: str, label_key: str
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        ids = batch["binary"]["input_ids"].to(device)
        am = batch["binary"]["attention_mask"].to(device)
        labels = torch.tensor([int(m[label_key]) for m in batch["metadata"]], device=device)
        logits = model(ids, am)
        preds = logits.argmax(-1)
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())
    return {"val_accuracy": correct / max(1, total)}


__all__ = [
    "LinearProbeClassifier",
    "RetrievalResult",
    "evaluate_retrieval",
    "extract_binary_embeddings",
    "train_linear_probe",
]
