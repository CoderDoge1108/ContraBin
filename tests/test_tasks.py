"""Smoke tests for the four downstream tasks."""

from __future__ import annotations

import random
from pathlib import Path

from torch.utils.data import Dataset

from contrabin.data.datasets import TripletDataset
from contrabin.data.loaders import TripletCollator, build_dataloader, build_tokenizer
from contrabin.models.contrabin import ContraBinModel
from contrabin.tasks.binary_retrieval import (
    LinearProbeClassifier,
    evaluate_retrieval,
    extract_binary_embeddings,
    train_linear_probe,
)
from contrabin.tasks.compiler_provenance import (
    CompilerProvenanceModel,
    ProvenanceLabelSpace,
    train_compiler_provenance,
)
from contrabin.tasks.name_recovery import (
    NameRecoveryModel,
    split_function_name,
    train_name_recovery,
)
from contrabin.tasks.summarization import SummarizationModel, train_summarization


class _LabeledTriplets(Dataset):
    """Wrap a TripletDataset so each record carries provenance / label metadata."""

    def __init__(self, path: Path, seed: int = 0):
        self.inner = TripletDataset(path)
        self._rng = random.Random(seed)
        compilers = ["gcc", "clang", "msvc"]
        opts = ["O0", "O2", "O3"]
        langs = ["c", "cpp", "rust"]
        self._meta: list[dict] = []
        for i in range(len(self.inner)):
            self._meta.append(
                {
                    "label": i % 3,
                    "name": f"my_func_{i % 2}_v{i}",
                    "comment_text": self.inner[i]["comment"],
                    "source_text": self.inner[i]["source"],
                    "compiler": compilers[i % len(compilers)],
                    "opt_level": opts[i % len(opts)],
                    "language": langs[i % len(langs)],
                }
            )

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        row = dict(self.inner[i])
        row["metadata"] = self._meta[i]
        return row


def _setup(tiny_config, tiny_triplets):
    train_path, val_path = tiny_triplets
    tok = build_tokenizer(tiny_config.model.encoder_name, vocab_size=64)
    collator = TripletCollator(
        tokenizer=tok,
        source_max_length=tiny_config.data.source_max_length,
        binary_max_length=tiny_config.data.binary_max_length,
        comment_max_length=tiny_config.data.comment_max_length,
    )
    train_loader = build_dataloader(_LabeledTriplets(train_path), collator, batch_size=4)
    val_loader = build_dataloader(
        _LabeledTriplets(val_path, seed=1), collator, batch_size=4, shuffle=False
    )
    model = ContraBinModel(tiny_config.model)
    return model, train_loader, val_loader


# ---------------------------------------------------------------------------
# Binary retrieval
# ---------------------------------------------------------------------------


def test_binary_retrieval_extract_and_eval(tiny_config, tiny_triplets):
    backbone, _, val_loader = _setup(tiny_config, tiny_triplets)
    embeddings, labels = extract_binary_embeddings(backbone, val_loader, device="cpu")
    assert embeddings.shape[0] == len(labels)
    result = evaluate_retrieval(embeddings, labels)
    assert 0.0 <= result.mAP <= 1.0
    assert 0.0 <= result.mrr <= 1.0
    assert 0.0 <= result.recall_at_1 <= 1.0


def test_linear_probe(tiny_config, tiny_triplets):
    backbone, train_loader, val_loader = _setup(tiny_config, tiny_triplets)
    probe = LinearProbeClassifier(backbone, num_classes=3)
    history = train_linear_probe(
        probe, train_loader, val_loader, num_epochs=1, device="cpu", head_lr=1e-2
    )
    assert "val_accuracy" in history[-1]


# ---------------------------------------------------------------------------
# Function name recovery
# ---------------------------------------------------------------------------


def test_split_function_name():
    assert split_function_name("MyFuncName_v2") == ["my", "func", "name", "v2"]
    assert split_function_name("snake_case") == ["snake", "case"]
    assert split_function_name("") == []


def test_name_recovery(tiny_config, tiny_triplets):
    backbone, train_loader, val_loader = _setup(tiny_config, tiny_triplets)
    vocab = ["my", "func", "v0", "v1", "v2", "v3"]
    model = NameRecoveryModel(backbone, vocab)
    history = train_name_recovery(model, train_loader, val_loader, num_epochs=1, device="cpu")
    assert "subtoken_f1" in history[-1]


# ---------------------------------------------------------------------------
# Binary summarization
# ---------------------------------------------------------------------------


def test_summarization(tiny_config, tiny_triplets):
    backbone, train_loader, val_loader = _setup(tiny_config, tiny_triplets)
    model = SummarizationModel(backbone, vocab_size=64, hidden_dim=16)
    history = train_summarization(model, train_loader, val_loader, num_epochs=1, device="cpu")
    assert "bleu" in history[-1]


# ---------------------------------------------------------------------------
# Compiler provenance
# ---------------------------------------------------------------------------


def test_compiler_provenance(tiny_config, tiny_triplets):
    backbone, train_loader, val_loader = _setup(tiny_config, tiny_triplets)
    label_space = ProvenanceLabelSpace(
        compilers=["gcc", "clang", "msvc"],
        opt_levels=["O0", "O2", "O3"],
        languages=["c", "cpp", "rust"],
    )
    model = CompilerProvenanceModel(backbone, label_space=label_space)
    history = train_compiler_provenance(model, train_loader, val_loader, num_epochs=1, device="cpu")
    entry = history[-1]
    assert 0.0 <= entry["compiler_accuracy"] <= 1.0
    assert 0.0 <= entry["opt_level_accuracy"] <= 1.0
    assert 0.0 <= entry["language_accuracy"] <= 1.0
