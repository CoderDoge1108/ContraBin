"""Lightweight training callback system.

Callbacks are simple classes with optional ``on_*`` hooks. The :class:`Trainer`
calls them at well-defined points. This keeps the training loop free from
tangled I/O concerns (logging, checkpointing, early stopping).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class TrainerLike(Protocol):
    state: Any
    """Duck-typed protocol to avoid circular imports."""


class Callback:
    """Base class. Override the hooks you need."""

    def on_train_begin(self, trainer: TrainerLike) -> None: ...
    def on_train_end(self, trainer: TrainerLike) -> None: ...
    def on_epoch_begin(self, trainer: TrainerLike) -> None: ...
    def on_epoch_end(self, trainer: TrainerLike, metrics: dict[str, float]) -> None: ...
    def on_step_end(self, trainer: TrainerLike, step: int, loss: float) -> None: ...
    def on_eval_end(self, trainer: TrainerLike, metrics: dict[str, float]) -> None: ...


class LoggingCallback(Callback):
    """Pretty-print stage / epoch / step progress via ``rich`` (falls back to print)."""

    def __init__(self, log_every_n_steps: int = 50) -> None:
        self.log_every = max(1, log_every_n_steps)

    def on_train_begin(self, trainer: TrainerLike) -> None:
        from contrabin.utils.logging import get_logger

        self._log = get_logger("contrabin.trainer")
        self._log.info("Starting training")

    def on_epoch_begin(self, trainer: TrainerLike) -> None:
        self._log.info(
            "Epoch %d | stage=%s", trainer.state.epoch, trainer.state.stage
        )

    def on_step_end(self, trainer: TrainerLike, step: int, loss: float) -> None:
        if step % self.log_every == 0:
            self._log.info("  step=%d loss=%.4f", step, loss)

    def on_eval_end(self, trainer: TrainerLike, metrics: dict[str, float]) -> None:
        msg = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self._log.info("  eval %s", msg)


class EarlyStopping(Callback):
    """Stop training if ``val_loss`` fails to improve for ``patience`` evaluations."""

    def __init__(self, patience: int = 3, key: str = "val_loss", min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.key = key
        self.min_delta = min_delta
        self._best = float("inf")
        self._bad = 0

    def on_eval_end(self, trainer: TrainerLike, metrics: dict[str, float]) -> None:
        v = metrics.get(self.key)
        if v is None:
            return
        if v + self.min_delta < self._best:
            self._best = v
            self._bad = 0
        else:
            self._bad += 1
            if self._bad >= self.patience:
                trainer.state.should_stop = True


class CheckpointCallback(Callback):
    """Save ``model.state_dict()`` to disk at configured step intervals."""

    def __init__(self, output_dir: str | Path, every_n_steps: int = 1000) -> None:
        self.output_dir = Path(output_dir)
        self.every_n_steps = max(1, every_n_steps)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, trainer: TrainerLike, step: int, loss: float) -> None:
        if step > 0 and step % self.every_n_steps == 0:
            import torch

            path = self.output_dir / f"step_{step:07d}.pt"
            torch.save(trainer.state.model.state_dict(), path)

    def on_epoch_end(self, trainer: TrainerLike, metrics: dict[str, float]) -> None:
        import torch

        path = self.output_dir / f"epoch_{trainer.state.epoch:03d}.pt"
        torch.save(trainer.state.model.state_dict(), path)
