"""PyTorch datasets for ContraBin.

Triplet JSONL files produced by :class:`~contrabin.data.triplet_builder.TripletBuilder`
are consumed by :class:`TripletDataset`. For offline tests we also provide
:func:`build_synthetic_triplets`.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from contrabin.utils.io import read_jsonl, write_jsonl


class TripletDataset(Dataset):
    """Dataset of ``(source, binary, comment)`` triplets.

    Parameters
    ----------
    path:
        Path to a JSONL file where each row has ``source``, ``binary``,
        ``comment`` (and optional ``metadata``).
    max_records:
        Optional cap on the number of records loaded.
    """

    def __init__(self, path: str | Path, max_records: int | None = None) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Triplet file not found: {self.path}. Run `contrabin build-triplets` first."
            )
        self._records: list[dict[str, Any]] = []
        for row in read_jsonl(self.path):
            self._records.append(row)
            if max_records and len(self._records) >= max_records:
                break

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._records[idx]

    @property
    def num_records(self) -> int:
        return len(self._records)


# ---------------------------------------------------------------------------
# Synthetic dataset for tests and the ``contrabin-tiny`` backbone
# ---------------------------------------------------------------------------


def build_synthetic_triplets(
    out_path: str | Path,
    n: int = 32,
    seed: int = 0,
) -> int:
    """Write ``n`` deterministic synthetic triplets to ``out_path``.

    Each "program" is a toy permutation of token vocabulary: the source,
    binary and comment share a controlled random seed so that contrastive
    alignment is learnable even on this tiny dataset.
    """
    rng = random.Random(seed)
    records = []
    templates = [
        (
            "int add(int a, int b) {{ return a + b; }}",
            "%0 = add i32 %a, %b",
            "add: sums two integers",
        ),
        (
            "int sub(int a, int b) {{ return a - b; }}",
            "%0 = sub i32 %a, %b",
            "sub: subtracts two integers",
        ),
        (
            "int mul(int a, int b) {{ return a * b; }}",
            "%0 = mul i32 %a, %b",
            "mul: multiplies two integers",
        ),
        ("int id(int x) {{ return x; }}", "ret i32 %x", "id: returns its argument"),
    ]
    for i in range(n):
        src_tmpl, bin_tmpl, cmt_tmpl = templates[i % len(templates)]
        salt = rng.randint(0, 10_000)
        records.append(
            {
                "source": src_tmpl + f" // id={salt}",
                "binary": bin_tmpl + f" ; id={salt}",
                "comment": cmt_tmpl,
                "metadata": {"idx": i, "salt": salt},
            }
        )
    return write_jsonl(out_path, records)


__all__ = ["TripletDataset", "build_synthetic_triplets"]
