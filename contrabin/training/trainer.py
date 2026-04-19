"""Pre-training trainer that implements Algorithm 1 of the paper."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from contrabin.config import ContraBinConfig
from contrabin.losses.contrastive import PrimaryContrastiveLoss
from contrabin.losses.intermediate import IntermediateContrastiveLoss
from contrabin.models.contrabin import ContraBinModel
from contrabin.training.callbacks import Callback, LoggingCallback
from contrabin.training.scheduler import CurriculumScheduler, build_lr_scheduler
from contrabin.utils.io import resolve_device
from contrabin.utils.logging import get_logger
from contrabin.utils.seed import seed_everything

logger = get_logger(__name__)


@dataclass
class TrainState:
    model: ContraBinModel
    epoch: int = 0
    global_step: int = 0
    stage: str = "naive"
    best_val_loss: float = float("inf")
    should_stop: bool = False
    history: list[dict] = field(default_factory=list)


class PretrainTrainer:
    """Two-stage contrastive trainer with curriculum interpolation.

    The trainer alternates between:
    1. Encoding each triplet batch in one of three stages (naive / linear /
       non-linear simplex interpolation).
    2. Computing a *primary* CLIP-style loss on the batch plus (when stage !=
       naive) an InfoNCE-style *intermediate* loss between the interpolated
       representation and the binary embedding.
    3. Back-propagating the sum with a clipped gradient step.
    """

    def __init__(
        self,
        config: ContraBinConfig,
        model: ContraBinModel | None = None,
        callbacks: Iterable[Callback] | None = None,
    ) -> None:
        self.config = config
        seed_everything(config.training.seed)
        self.device = resolve_device(config.training.device)
        self.model = model or ContraBinModel(config.model)
        self.model.to(self.device)

        self.primary_loss = PrimaryContrastiveLoss(
            temperature=config.model.temperature
        )
        self.intermediate_loss = IntermediateContrastiveLoss(
            temperature=config.model.temperature
        )

        self.curriculum = CurriculumScheduler(config.training.curriculum)
        self.callbacks: list[Callback] = list(callbacks or [LoggingCallback(
            log_every_n_steps=config.training.log_every_n_steps
        )])
        self.state = TrainState(model=self.model)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.config.optim
        # Two parameter groups: low LR for encoders, higher LR for projection heads.
        encoder_params = [p for p in self.model.binary_encoder.parameters() if p.requires_grad]
        head_params = []
        for m in (
            self.model.source_head,
            self.model.comment_head,
            self.model.binary_head,
            self.model.interpolation,
        ):
            head_params += list(m.parameters())
        groups = [
            {"params": encoder_params, "lr": cfg.lr},
            {"params": head_params, "lr": cfg.head_lr},
        ]
        return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay, betas=cfg.betas)

    # ------------------------------------------------------------------
    # Callback dispatch
    # ------------------------------------------------------------------
    def _dispatch(self, hook: str, *args, **kwargs) -> None:
        for cb in self.callbacks:
            getattr(cb, hook)(self, *args, **kwargs)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader | None = None
    ) -> TrainState:
        total_epochs = self.config.training.curriculum.total_epochs()
        total_steps = max(1, total_epochs * max(1, len(train_loader)))
        optimizer = self._build_optimizer()
        scheduler = build_lr_scheduler(optimizer, self.config.optim, total_steps)

        self._dispatch("on_train_begin")
        for epoch in range(total_epochs):
            self.state.epoch = epoch
            self.state.stage = self.curriculum.cfg.stage_for_epoch(epoch)
            self._dispatch("on_epoch_begin")

            epoch_loss = self._train_one_epoch(train_loader, optimizer, scheduler)

            metrics: dict[str, float] = {"train_loss": epoch_loss}
            if val_loader is not None:
                metrics["val_loss"] = self.evaluate(val_loader)
                self._dispatch("on_eval_end", metrics)

            self._dispatch("on_epoch_end", metrics)
            self.state.history.append({"epoch": epoch, **metrics})
            self.curriculum.step()
            if self.state.should_stop:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

        self._dispatch("on_train_end")
        return self.state

    def _train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ) -> float:
        self.model.train()
        total = 0.0
        accum = self.config.training.grad_accumulation_steps
        max_norm = self.config.optim.max_grad_norm
        stage = self.state.stage

        for step, batch in enumerate(loader):
            batch = _to_device(batch, self.device)
            out = self.model(batch, stage=stage)

            loss = self.primary_loss(out.source, out.binary) + self.primary_loss(
                out.comment, out.binary
            )
            if stage != "naive" and out.intermediate is not None:
                loss = loss + self.intermediate_loss(out.intermediate, out.binary)
            loss = loss / accum
            loss.backward()

            if (step + 1) % accum == 0:
                if max_norm:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            self.state.global_step += 1
            total += float(loss.item()) * accum
            self._dispatch("on_step_end", self.state.global_step, float(loss.item()) * accum)
        return total / max(1, len(loader))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total = 0.0
        stage = self.state.stage
        for batch in loader:
            batch = _to_device(batch, self.device)
            out = self.model(batch, stage=stage)
            loss = self.primary_loss(out.source, out.binary) + self.primary_loss(
                out.comment, out.binary
            )
            if stage != "naive" and out.intermediate is not None:
                loss = loss + self.intermediate_loss(out.intermediate, out.binary)
            total += float(loss.item())
        return total / max(1, len(loader))

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": self.config.to_dict(),
                "history": self.state.history,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.state.history = ckpt.get("history", [])


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _to_device(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = {kk: vv.to(device) if hasattr(vv, "to") else vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out
