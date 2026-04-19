"""Use a ContraBin encoder to produce binary embeddings.

This example loads the offline tiny backbone and shows how to:

1. Build a ContraBinModel.
2. Encode a raw binary / IR string without going through the trainer.
3. Compute cosine similarity between two binary functions.

Usage
-----

    python examples/use_pretrained_embedding.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from contrabin.config import ModelConfig
from contrabin.data.loaders import build_tokenizer
from contrabin.models.contrabin import ContraBinModel


def main() -> None:
    cfg = ModelConfig(
        encoder_name="contrabin-tiny",
        hidden_dim=32,
        projection_dim=16,
        dropout=0.0,
    )
    model = ContraBinModel(cfg).eval()
    tokenizer = build_tokenizer(cfg.encoder_name, vocab_size=64)

    snippets = [
        "%0 = add i32 %a, %b\nret i32 %0",
        "%0 = sub i32 %a, %b\nret i32 %0",
        "%0 = add i32 %a, %b\nret i32 %0",  # duplicate of the first
    ]
    batch = tokenizer(
        snippets, padding="max_length", truncation=True, max_length=32, return_tensors="pt"
    )
    with torch.no_grad():
        embeddings = model.binary_embedding(batch["input_ids"], batch["attention_mask"])
    normed = F.normalize(embeddings, dim=-1)

    print("Cosine similarities:")
    for i, x in enumerate(snippets):
        for j, y in enumerate(snippets):
            if j <= i:
                continue
            sim = float((normed[i] @ normed[j]).item())
            print(f"  {i} vs {j}:  {sim:+.4f}  ({x!r} ↔ {y!r})")


if __name__ == "__main__":
    main()
