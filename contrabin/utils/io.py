"""Minimal JSONL helpers used throughout the project."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def _open(path: Path, mode: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Yield dictionaries from a JSONL file (``.jsonl`` or ``.jsonl.gz``)."""
    path = Path(path)
    with _open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path | str, records: Iterable[dict[str, Any]]) -> int:
    """Write ``records`` as JSONL. Returns the number of rows written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with _open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def resolve_device(device: str = "auto") -> str:
    """Resolve a device string, supporting ``auto`` for cuda/mps/cpu fallback."""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
