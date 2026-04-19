"""Tokenization collator and DataLoader factory."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Tokenizer protocol (a subset of the HF tokenizer interface we rely on)
# ---------------------------------------------------------------------------


class Tokenizer(Protocol):
    def __call__(
        self,
        text: list[str],
        padding: bool | str = ...,
        truncation: bool = ...,
        max_length: int | None = ...,
        return_tensors: str | None = ...,
    ) -> dict[str, torch.Tensor]: ...


class _ByteHashTokenizer:
    """Deterministic byte-level hashing tokenizer for offline tests.

    This is **not** a high-quality tokenizer. It maps bytes to pseudo-random
    ids from a fixed vocab so the full training loop can run without network
    access. The :class:`contrabin-tiny` encoder pairs with this tokenizer.
    """

    def __init__(self, vocab_size: int = 1024, pad_token_id: int = 0) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        text: list[str] | str,
        padding: bool | str = "max_length",
        truncation: bool = True,
        max_length: int | None = 64,
        return_tensors: str | None = "pt",
    ) -> dict[str, torch.Tensor]:
        if isinstance(text, str):
            text = [text]
        max_length = max_length or 64
        ids = torch.full((len(text), max_length), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(text), max_length), dtype=torch.long)
        for i, s in enumerate(text):
            b = s.encode("utf-8", errors="ignore")[: max_length * 4]
            toks = [
                (
                    int.from_bytes(hashlib.md5(b[j : j + 4]).digest()[:2], "little")
                    % (self.vocab_size - 1)
                )
                + 1
                for j in range(0, len(b), 4)
            ][:max_length]
            if truncation:
                toks = toks[:max_length]
            if len(toks) == 0:
                toks = [1]
            ids[i, : len(toks)] = torch.tensor(toks)
            mask[i, : len(toks)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def build_tokenizer(name: str, vocab_size: int = 1024) -> Tokenizer:
    """Return either a HuggingFace fast tokenizer or the byte-hash tokenizer.

    The offline tokenizer is used when ``name == 'contrabin-tiny'``.
    """
    if name == "contrabin-tiny":
        return _ByteHashTokenizer(vocab_size=vocab_size)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------


@dataclass
class TripletCollator:
    """Tokenize a batch of triplet dicts into three padded tensor dicts."""

    tokenizer: Tokenizer
    source_max_length: int = 512
    binary_max_length: int = 512
    comment_max_length: int = 64
    binary_tokenizer: Tokenizer | None = None

    def __post_init__(self) -> None:
        if self.binary_tokenizer is None:
            self.binary_tokenizer = self.tokenizer

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, dict[str, torch.Tensor]]:
        sources = [r["source"] for r in batch]
        binaries = [r["binary"] for r in batch]
        comments = [r["comment"] for r in batch]
        metadata = [r.get("metadata", {}) for r in batch]
        binary_tokenizer = self.binary_tokenizer or self.tokenizer
        return {
            "source": self.tokenizer(
                sources,
                padding="max_length",
                truncation=True,
                max_length=self.source_max_length,
                return_tensors="pt",
            ),
            "binary": binary_tokenizer(
                binaries,
                padding="max_length",
                truncation=True,
                max_length=self.binary_max_length,
                return_tensors="pt",
            ),
            "comment": self.tokenizer(
                comments,
                padding="max_length",
                truncation=True,
                max_length=self.comment_max_length,
                return_tensors="pt",
            ),
            "metadata": metadata,  # type: ignore[dict-item]
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_dataloader(
    dataset: Dataset,
    collator: TripletCollator,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


__all__ = ["Tokenizer", "TripletCollator", "build_dataloader", "build_tokenizer"]
