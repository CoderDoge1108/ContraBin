from pathlib import Path

import torch

from contrabin.data.datasets import TripletDataset
from contrabin.data.loaders import TripletCollator, build_dataloader, build_tokenizer
from contrabin.training.scheduler import CurriculumScheduler, build_lr_scheduler
from contrabin.training.trainer import PretrainTrainer


def test_curriculum_scheduler_stages(tiny_config):
    sched = CurriculumScheduler(tiny_config.training.curriculum)
    stages = list(sched)
    assert stages == ["naive", "linear", "nonlinear"]


def test_build_lr_scheduler_warmup():
    params = [torch.nn.Parameter(torch.randn(2))]
    opt = torch.optim.AdamW(params, lr=1.0)

    class _Opt:
        warmup_steps = 2
        scheduler = "constant"

    sched = build_lr_scheduler(opt, _Opt(), total_steps=10)
    for _ in range(4):
        opt.step()
        sched.step()
    assert opt.param_groups[0]["lr"] == 1.0


def test_pretrain_trainer_end_to_end(tiny_config, tiny_triplets: tuple[Path, Path]):
    train_path, val_path = tiny_triplets
    tok = build_tokenizer(tiny_config.model.encoder_name, vocab_size=64)
    collator = TripletCollator(
        tokenizer=tok,
        source_max_length=tiny_config.data.source_max_length,
        binary_max_length=tiny_config.data.binary_max_length,
        comment_max_length=tiny_config.data.comment_max_length,
    )
    train_loader = build_dataloader(
        TripletDataset(train_path),
        collator,
        batch_size=tiny_config.training.batch_size,
        shuffle=True,
    )
    val_loader = build_dataloader(
        TripletDataset(val_path),
        collator,
        batch_size=tiny_config.training.eval_batch_size,
        shuffle=False,
    )
    trainer = PretrainTrainer(tiny_config)
    state = trainer.fit(train_loader, val_loader)
    assert state.history, "trainer should record per-epoch metrics"
    assert {"train_loss", "val_loss"} <= set(state.history[-1].keys())
