"""Compiler provenance recovery.

Why this task
-------------
The ContraBin paper (TMLR 2025) reports a "reverse engineering" benchmark
framed as binary-to-C source generation, scored with BLEU against an (often
compiler-normalized) ground-truth C file. That framing is problematic:

* BLEU over source code is a weak signal - two decompilations that are
  semantically identical can have near-zero BLEU (renamed locals, different
  loop constructs, etc.), while stylistically similar but semantically wrong
  output can score highly.
* True binary-to-source lifting is a research field on its own (BinSync,
  Ghidra + ML, DIRE, neural decompilers), and evaluating it well requires
  test-execution infrastructure.

Instead we ship **compiler provenance recovery**: given a binary / IR blob,
predict metadata about how it was produced. This is a well-posed classification
problem with unambiguous ground truth, directly measures how much "production
context" a self-supervised binary encoder has internalised, and is used by
several downstream security and forensics pipelines (e.g., libc fingerprinting,
malware attribution, compiler identification for vulnerability triage).

Three heads
-----------
* ``compiler``  : multi-class (``gcc``, ``clang``, ``msvc``, ``icc``, ``unknown``).
* ``opt_level`` : multi-class (``O0``, ``O1``, ``O2``, ``O3``, ``Os``, ``Oz``).
* ``language``  : multi-class (``c``, ``cpp``, ``rust``, ``go``, ``fortran``, ``unknown``).

The class lists are configurable - see :class:`ProvenanceLabelSpace`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from contrabin.evaluation.metrics import token_accuracy
from contrabin.models.contrabin import ContraBinModel


@dataclass
class ProvenanceLabelSpace:
    """Closed-world label vocabularies used by the three provenance heads."""

    compilers: list[str] = field(
        default_factory=lambda: ["gcc", "clang", "msvc", "icc", "unknown"]
    )
    opt_levels: list[str] = field(
        default_factory=lambda: ["O0", "O1", "O2", "O3", "Os", "Oz"]
    )
    languages: list[str] = field(
        default_factory=lambda: ["c", "cpp", "rust", "go", "fortran", "unknown"]
    )

    def __post_init__(self) -> None:
        self._c2i = {v: i for i, v in enumerate(self.compilers)}
        self._o2i = {v: i for i, v in enumerate(self.opt_levels)}
        self._l2i = {v: i for i, v in enumerate(self.languages)}
        self._compiler_fallback = self._c2i.get("unknown", 0)
        self._language_fallback = self._l2i.get("unknown", 0)

    def encode_compiler(self, name: str) -> int:
        return self._c2i.get(name, self._compiler_fallback)

    def encode_opt(self, level: str) -> int:
        level = level.lstrip("-").capitalize() if not level.startswith("O") else level
        return self._o2i.get(level, 0)

    def encode_language(self, lang: str) -> int:
        return self._l2i.get(lang.lower(), self._language_fallback)

    @property
    def num_compilers(self) -> int:
        return len(self.compilers)

    @property
    def num_opt_levels(self) -> int:
        return len(self.opt_levels)

    @property
    def num_languages(self) -> int:
        return len(self.languages)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class CompilerProvenanceModel(nn.Module):
    """Multi-head classifier on top of the ContraBin binary encoder.

    Each head is a small MLP; the backbone is fine-tunable by default, but
    can be frozen by the caller (``requires_grad_(False)``) for a cheaper
    linear-probe style evaluation.
    """

    def __init__(
        self,
        backbone: ContraBinModel,
        label_space: ProvenanceLabelSpace | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.label_space = label_space or ProvenanceLabelSpace()
        hidden = backbone.config.projection_dim
        self.compiler_head = _head(hidden, self.label_space.num_compilers, dropout)
        self.opt_head = _head(hidden, self.label_space.num_opt_levels, dropout)
        self.language_head = _head(hidden, self.label_space.num_languages, dropout)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        emb = self.backbone.binary_embedding(input_ids, attention_mask)
        return {
            "compiler": self.compiler_head(emb),
            "opt_level": self.opt_head(emb),
            "language": self.language_head(emb),
        }


def _head(hidden: int, n_classes: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, n_classes),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class ProvenanceResult:
    compiler_accuracy: float
    opt_level_accuracy: float
    language_accuracy: float
    overall_accuracy: float


def _collect_targets(
    batch: dict, label_space: ProvenanceLabelSpace, device: str
) -> dict[str, torch.Tensor]:
    metas = batch["metadata"]
    comp = torch.tensor(
        [label_space.encode_compiler(m.get("compiler", "unknown")) for m in metas],
        device=device,
    )
    opt = torch.tensor(
        [label_space.encode_opt(m.get("opt_level", "O0")) for m in metas],
        device=device,
    )
    lang = torch.tensor(
        [label_space.encode_language(m.get("language", "c")) for m in metas],
        device=device,
    )
    return {"compiler": comp, "opt_level": opt, "language": lang}


def train_compiler_provenance(
    model: CompilerProvenanceModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    *,
    num_epochs: int = 5,
    lr: float = 2e-5,
    head_lr: float = 1e-3,
    device: str = "cpu",
    weight_decay: float = 1e-2,
    loss_weights: dict[str, float] | None = None,
) -> list[dict[str, float]]:
    """Fine-tune the provenance heads (optionally with the backbone).

    ``loss_weights`` can rebalance the three heads, e.g.
    ``{"compiler": 1.0, "opt_level": 2.0, "language": 0.5}``.
    """
    model.to(device)
    head_params: list[torch.nn.Parameter] = []
    for h in (model.compiler_head, model.opt_head, model.language_head):
        head_params += list(h.parameters())
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
    weights = loss_weights or {"compiler": 1.0, "opt_level": 1.0, "language": 1.0}

    history: list[dict[str, float]] = []
    for epoch in range(num_epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            ids = batch["binary"]["input_ids"].to(device)
            am = batch["binary"]["attention_mask"].to(device)
            targets = _collect_targets(batch, model.label_space, device)
            logits = model(ids, am)
            loss = sum(
                weights[k] * F.cross_entropy(logits[k], targets[k]) for k in weights
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        entry = {"epoch": epoch, "train_loss": total / max(1, len(train_loader))}
        if val_loader is not None:
            entry.update(_evaluate(model, val_loader, device))
        history.append(entry)
    return history


@torch.no_grad()
def _evaluate(
    model: CompilerProvenanceModel, loader: DataLoader, device: str
) -> dict[str, float]:
    model.eval()
    preds: dict[str, list[int]] = {k: [] for k in ("compiler", "opt_level", "language")}
    targs: dict[str, list[int]] = {k: [] for k in ("compiler", "opt_level", "language")}
    all_correct = 0
    total = 0
    for batch in loader:
        ids = batch["binary"]["input_ids"].to(device)
        am = batch["binary"]["attention_mask"].to(device)
        targets = _collect_targets(batch, model.label_space, device)
        logits = model(ids, am)
        batch_correct = torch.ones(ids.size(0), dtype=torch.bool, device=device)
        for k in preds:
            p = logits[k].argmax(-1)
            preds[k].extend(p.tolist())
            targs[k].extend(targets[k].tolist())
            batch_correct &= p == targets[k]
        all_correct += int(batch_correct.sum().item())
        total += ids.size(0)
    return {
        "compiler_accuracy": token_accuracy(preds["compiler"], targs["compiler"]),
        "opt_level_accuracy": token_accuracy(preds["opt_level"], targs["opt_level"]),
        "language_accuracy": token_accuracy(preds["language"], targs["language"]),
        "overall_accuracy": all_correct / max(1, total),
    }


__all__ = [
    "CompilerProvenanceModel",
    "ProvenanceLabelSpace",
    "ProvenanceResult",
    "train_compiler_provenance",
]
