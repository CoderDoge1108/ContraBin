"""End-to-end pretraining demo on synthetic data.

Runs in < 10s on a laptop CPU. No network / clang / HF downloads needed.

Usage
-----

    python examples/pretrain_tiny.py

This prints the training-loss history after one full curriculum pass.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from contrabin.config import (
    ContraBinConfig,
    CurriculumConfig,
    DataConfig,
    ModelConfig,
    OptimConfig,
    TrainingConfig,
)
from contrabin.data.datasets import TripletDataset, build_synthetic_triplets
from contrabin.data.loaders import (
    TripletCollator,
    build_dataloader,
    build_tokenizer,
)
from contrabin.training.trainer import PretrainTrainer


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        train_path = td / "train.jsonl"
        val_path = td / "val.jsonl"
        build_synthetic_triplets(train_path, n=64, seed=0)
        build_synthetic_triplets(val_path, n=16, seed=1)

        cfg = ContraBinConfig(
            data=DataConfig(
                train_path=train_path,
                val_path=val_path,
                source_max_length=32,
                binary_max_length=32,
                comment_max_length=12,
                num_workers=0,
            ),
            model=ModelConfig(
                encoder_name="contrabin-tiny",
                hidden_dim=32,
                projection_dim=16,
                dropout=0.0,
                temperature=0.1,
            ),
            optim=OptimConfig(lr=1e-3, head_lr=1e-3, warmup_steps=0, scheduler="constant"),
            training=TrainingConfig(
                batch_size=8,
                eval_batch_size=8,
                device="cpu",
                output_dir=td / "outputs",
                log_every_n_steps=100,
                curriculum=CurriculumConfig(primary_epochs=1, linear_epochs=1, nonlinear_epochs=1),
            ),
        )

        tokenizer = build_tokenizer(cfg.model.encoder_name, vocab_size=64)
        collator = TripletCollator(
            tokenizer=tokenizer,
            source_max_length=cfg.data.source_max_length,
            binary_max_length=cfg.data.binary_max_length,
            comment_max_length=cfg.data.comment_max_length,
        )
        train_loader = build_dataloader(
            TripletDataset(train_path), collator, batch_size=cfg.training.batch_size
        )
        val_loader = build_dataloader(
            TripletDataset(val_path),
            collator,
            batch_size=cfg.training.eval_batch_size,
            shuffle=False,
        )

        trainer = PretrainTrainer(cfg)
        state = trainer.fit(train_loader, val_loader)
        for row in state.history:
            print(row)


if __name__ == "__main__":
    main()
