"""Optional visualization utilities.

Importing this module does not require :mod:`matplotlib`. The functions raise an
informative error if visualization dependencies are missing.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError as e:  # pragma: no cover - import guarded
        raise ImportError(
            "Visualization utilities require matplotlib. "
            "Install with: pip install 'contrabin[viz]'"
        ) from e


def plot_training_loss(
    train_losses: Sequence[float],
    val_losses: Sequence[float] | None = None,
    out_path: str | Path | None = None,
) -> None:
    """Simple training curve plot. Saves to ``out_path`` if provided."""
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(train_losses) + 1), train_losses, label="train")
    if val_losses:
        ax.plot(range(1, len(val_losses) + 1), val_losses, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("ContraBin pre-training curve")
    ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_embedding_umap(
    embeddings,
    labels: Sequence[str] | None = None,
    out_path: str | Path | None = None,
) -> None:
    """UMAP projection of ``embeddings`` into 2D. Requires :mod:`umap-learn`."""
    plt = _require_matplotlib()
    try:
        import umap
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install with: pip install 'contrabin[viz]'") from e
    import numpy as np

    embeddings = np.asarray(embeddings)
    reducer = umap.UMAP(n_components=2, random_state=0)
    proj = reducer.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(6, 6))
    if labels is None:
        ax.scatter(proj[:, 0], proj[:, 1], s=4, alpha=0.7)
    else:
        import itertools

        palette = itertools.cycle(plt.cm.tab20.colors)
        for lab in sorted(set(labels)):
            mask = [x == lab for x in labels]
            ax.scatter(proj[mask, 0], proj[mask, 1], s=6, alpha=0.8, label=str(lab), color=next(palette))
        ax.legend(fontsize=7, loc="best", markerscale=2)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    plt.close(fig)
