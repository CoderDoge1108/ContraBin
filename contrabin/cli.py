"""Unified command-line interface.

Example
-------

.. code-block:: bash

    # 1) Build triplets from a directory of C files:
    contrabin build-triplets \\
        --input data/raw/anghabench \\
        --output data/processed/triplets.jsonl

    # 2) Pre-train:
    contrabin pretrain --config configs/pretrain.yaml

    # 3) Fine-tune functionality classification on top of the pre-trained model:
    contrabin finetune-classification \\
        --config configs/finetune_poj104.yaml \\
        --pretrained outputs/contrabin-pretrain/final.pt
"""

from __future__ import annotations

from pathlib import Path

import typer

from contrabin.config import ContraBinConfig, load_config
from contrabin.utils.logging import setup_logging

app = typer.Typer(
    name="contrabin",
    help="Contrastive pre-training of binary code representations (ContraBin).",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


# ---------------------------------------------------------------------------
# utility commands
# ---------------------------------------------------------------------------


@app.callback()
def _global_opts(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Emit DEBUG logs."),
) -> None:
    setup_logging("DEBUG" if verbose else "INFO")


@app.command("version")
def version() -> None:
    """Print the installed ContraBin version."""
    from contrabin import __version__

    typer.echo(__version__)


# ---------------------------------------------------------------------------
# build-triplets
# ---------------------------------------------------------------------------


@app.command("build-triplets")
def build_triplets_cmd(
    input_dir: Path = typer.Option(..., "--input", "-i", help="Directory of C source files."),
    output: Path = typer.Option(..., "--output", "-o", help="Destination JSONL file."),
    pattern: str = typer.Option("**/*.c", help="Glob for source files."),
    optimization: str = typer.Option("-O0", help="clang optimization level."),
    comment_generator: str = typer.Option(
        "heuristic", "--comment-generator", help="'heuristic' or 'openai'."
    ),
    allow_fallback: bool = typer.Option(True, help="Synthesize dummy IR if clang is missing."),
) -> None:
    """Compile each C file and attach a generated comment to form a triplet."""
    from contrabin.data.comment_generator import (
        HeuristicCommentGenerator,
        OpenAICommentGenerator,
    )
    from contrabin.data.triplet_builder import TripletBuilder

    gen = (
        HeuristicCommentGenerator()
        if comment_generator == "heuristic"
        else OpenAICommentGenerator()
    )
    builder = TripletBuilder(
        comment_generator=gen, optimization=optimization, allow_ir_fallback=allow_fallback
    )
    n = builder.write_jsonl(builder.iter_from_directory(input_dir, pattern), output)
    typer.echo(f"Wrote {n} triplets to {output}")


# ---------------------------------------------------------------------------
# generate-synthetic-triplets (for quick smoke tests)
# ---------------------------------------------------------------------------


@app.command("make-synthetic")
def make_synthetic_cmd(
    output: Path = typer.Option(..., "--output", "-o"),
    n: int = typer.Option(64, "--num", "-n"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Generate a tiny synthetic triplet file for offline tests / demos."""
    from contrabin.data.datasets import build_synthetic_triplets

    count = build_synthetic_triplets(output, n=n, seed=seed)
    typer.echo(f"Wrote {count} synthetic triplets to {output}")


# ---------------------------------------------------------------------------
# pretrain
# ---------------------------------------------------------------------------


@app.command("pretrain")
def pretrain_cmd(
    config: Path = typer.Option(..., "--config", "-c", help="YAML config path."),
    override_output: Path | None = typer.Option(None, "--output-dir", "-o"),
    max_steps: int | None = typer.Option(None, help="Cap total steps (useful for smoke tests)."),
) -> None:
    """Run the ContraBin pre-training curriculum."""
    from contrabin.data.datasets import TripletDataset, build_synthetic_triplets
    from contrabin.data.loaders import TripletCollator, build_dataloader, build_tokenizer
    from contrabin.training.trainer import PretrainTrainer

    cfg = load_config(config)
    if override_output is not None:
        cfg.training.output_dir = override_output

    # Auto-generate synthetic data for the smoke config.
    if cfg.data.dataset_name == "synthetic":
        for p in (cfg.data.train_path, cfg.data.val_path):
            if not p.exists():
                build_synthetic_triplets(p, n=16, seed=cfg.training.seed)

    train_ds = TripletDataset(cfg.data.train_path)
    val_ds = TripletDataset(cfg.data.val_path)
    tokenizer = build_tokenizer(cfg.model.encoder_name)
    collator = TripletCollator(
        tokenizer=tokenizer,
        source_max_length=cfg.data.source_max_length,
        binary_max_length=cfg.data.binary_max_length,
        comment_max_length=cfg.data.comment_max_length,
    )
    train_loader = build_dataloader(
        train_ds, collator, batch_size=cfg.training.batch_size, num_workers=cfg.data.num_workers
    )
    val_loader = build_dataloader(
        val_ds,
        collator,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    trainer = PretrainTrainer(cfg)
    if max_steps is not None:
        # Clamp total epochs so CI smoke tests return quickly.
        cfg.training.curriculum.primary_epochs = 1
        cfg.training.curriculum.linear_epochs = 0
        cfg.training.curriculum.nonlinear_epochs = 0

    state = trainer.fit(train_loader, val_loader)
    out = Path(cfg.training.output_dir) / "final.pt"
    trainer.save(out)
    typer.echo(f"Saved {out} (history rows: {len(state.history)})")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


@app.command("evaluate")
def evaluate_cmd(
    config: Path = typer.Option(..., "--config", "-c"),
    checkpoint: Path = typer.Option(..., "--checkpoint", "-w"),
) -> None:
    """Load a checkpoint and report validation loss on the configured val split."""
    from contrabin.data.datasets import TripletDataset
    from contrabin.data.loaders import TripletCollator, build_dataloader, build_tokenizer
    from contrabin.training.trainer import PretrainTrainer

    cfg = load_config(config)
    ds = TripletDataset(cfg.data.val_path)
    tok = build_tokenizer(cfg.model.encoder_name)
    collator = TripletCollator(
        tokenizer=tok,
        source_max_length=cfg.data.source_max_length,
        binary_max_length=cfg.data.binary_max_length,
        comment_max_length=cfg.data.comment_max_length,
    )
    loader = build_dataloader(ds, collator, batch_size=cfg.training.eval_batch_size, shuffle=False)
    trainer = PretrainTrainer(cfg)
    trainer.load(checkpoint)
    loss = trainer.evaluate(loader)
    typer.echo(f"val_loss={loss:.4f}")


# ---------------------------------------------------------------------------
# write-default-config
# ---------------------------------------------------------------------------


@app.command("write-default-config")
def write_default_config(
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    """Emit a fully-filled-in YAML config with default values."""
    cfg = ContraBinConfig()
    cfg.save_yaml(output)
    typer.echo(f"Wrote default config to {output}")


# ---------------------------------------------------------------------------
# embed: extract binary embeddings to disk (for external evaluation)
# ---------------------------------------------------------------------------


@app.command("embed")
def embed_cmd(
    config: Path = typer.Option(..., "--config", "-c"),
    checkpoint: Path = typer.Option(..., "--checkpoint", "-w"),
    input_jsonl: Path = typer.Option(..., "--input", "-i"),
    output: Path = typer.Option(..., "--output", "-o", help="NPZ destination."),
    device: str = typer.Option("cpu"),
) -> None:
    """Extract binary embeddings from a trained ContraBin checkpoint."""
    import numpy as np

    from contrabin.data.datasets import TripletDataset
    from contrabin.data.loaders import TripletCollator, build_dataloader, build_tokenizer
    from contrabin.models.contrabin import ContraBinModel
    from contrabin.tasks.binary_retrieval import extract_binary_embeddings

    cfg = load_config(config)
    ds = TripletDataset(input_jsonl)
    tok = build_tokenizer(cfg.model.encoder_name)
    collator = TripletCollator(
        tokenizer=tok,
        source_max_length=cfg.data.source_max_length,
        binary_max_length=cfg.data.binary_max_length,
        comment_max_length=cfg.data.comment_max_length,
    )
    loader = build_dataloader(ds, collator, batch_size=cfg.training.eval_batch_size, shuffle=False)

    import torch

    model = ContraBinModel(cfg.model)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    embeddings, labels = extract_binary_embeddings(model, loader, device=device, label_key="idx")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, embeddings=embeddings.numpy(), labels=np.asarray(labels, dtype=np.int64)
    )
    typer.echo(f"Wrote {embeddings.shape[0]} embeddings to {output}")


# ---------------------------------------------------------------------------
# task <name>: run one of the four downstream tasks
# ---------------------------------------------------------------------------


task_app = typer.Typer(help="Downstream-task entry points.")
app.add_typer(task_app, name="task")


def _load_task_backbone(config: Path, checkpoint: Path | None, device: str):
    import torch

    from contrabin.models.contrabin import ContraBinModel

    cfg = load_config(config)
    model = ContraBinModel(cfg.model)
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    return cfg, model


def _task_loaders(cfg, path: Path, batch_size: int, shuffle: bool):
    from contrabin.data.datasets import TripletDataset
    from contrabin.data.loaders import TripletCollator, build_dataloader, build_tokenizer

    ds = TripletDataset(path)
    tok = build_tokenizer(cfg.model.encoder_name)
    collator = TripletCollator(
        tokenizer=tok,
        source_max_length=cfg.data.source_max_length,
        binary_max_length=cfg.data.binary_max_length,
        comment_max_length=cfg.data.comment_max_length,
    )
    return build_dataloader(ds, collator, batch_size=batch_size, shuffle=shuffle)


@task_app.command("retrieve")
def task_retrieve(
    config: Path = typer.Option(..., "--config", "-c"),
    checkpoint: Path | None = typer.Option(None, "--checkpoint", "-w"),
    gallery: Path = typer.Option(..., "--gallery", "-g", help="Triplet JSONL gallery."),
    device: str = typer.Option("cpu"),
) -> None:
    """Evaluate binary retrieval (mAP / MRR / Recall@k) on a gallery."""
    from contrabin.tasks.binary_retrieval import evaluate_retrieval, extract_binary_embeddings

    cfg, model = _load_task_backbone(config, checkpoint, device)
    loader = _task_loaders(cfg, gallery, batch_size=cfg.training.eval_batch_size, shuffle=False)
    embs, labels = extract_binary_embeddings(model, loader, device=device, label_key="idx")
    res = evaluate_retrieval(embs, labels)
    typer.echo(
        f"mAP={res.mAP:.4f}  MRR={res.mrr:.4f}  "
        f"R@1={res.recall_at_1:.4f}  R@5={res.recall_at_5:.4f}  R@10={res.recall_at_10:.4f}"
    )


@task_app.command("name-recovery")
def task_name_recovery(
    config: Path = typer.Option(..., "--config", "-c"),
    checkpoint: Path | None = typer.Option(None, "--checkpoint", "-w"),
    train_path: Path = typer.Option(..., "--train"),
    val_path: Path = typer.Option(..., "--val"),
    vocab_size: int = typer.Option(1024, "--vocab-size"),
    num_epochs: int = typer.Option(3),
    device: str = typer.Option("cpu"),
) -> None:
    """Fine-tune a function-name recovery head on top of ContraBin."""
    from contrabin.data.datasets import TripletDataset
    from contrabin.tasks.name_recovery import (
        NameRecoveryModel,
        split_function_name,
        train_name_recovery,
    )

    cfg, backbone = _load_task_backbone(config, checkpoint, device)
    # Build subtoken vocabulary from training names.
    ds = TripletDataset(train_path)
    tokens: dict[str, int] = {}
    for r in ds._records:
        name = r.get("metadata", {}).get("name", "")
        for t in split_function_name(name):
            tokens[t] = tokens.get(t, 0) + 1
    vocab = [t for t, _ in sorted(tokens.items(), key=lambda kv: -kv[1])[:vocab_size]]
    if not vocab:
        raise typer.BadParameter("No function names found in training metadata.")

    train_loader = _task_loaders(cfg, train_path, cfg.training.batch_size, shuffle=True)
    val_loader = _task_loaders(cfg, val_path, cfg.training.eval_batch_size, shuffle=False)
    model = NameRecoveryModel(backbone, vocab)
    history = train_name_recovery(
        model, train_loader, val_loader, num_epochs=num_epochs, device=device
    )
    typer.echo(history[-1])


@task_app.command("summarize")
def task_summarize(
    config: Path = typer.Option(..., "--config", "-c"),
    checkpoint: Path | None = typer.Option(None, "--checkpoint", "-w"),
    train_path: Path = typer.Option(..., "--train"),
    val_path: Path = typer.Option(..., "--val"),
    vocab_size: int = typer.Option(1024),
    num_epochs: int = typer.Option(3),
    device: str = typer.Option("cpu"),
) -> None:
    """Fine-tune a binary -> natural-language summarization head."""
    from contrabin.tasks.summarization import SummarizationModel, train_summarization

    cfg, backbone = _load_task_backbone(config, checkpoint, device)
    train_loader = _task_loaders(cfg, train_path, cfg.training.batch_size, shuffle=True)
    val_loader = _task_loaders(cfg, val_path, cfg.training.eval_batch_size, shuffle=False)
    model = SummarizationModel(backbone, vocab_size=vocab_size)
    history = train_summarization(
        model, train_loader, val_loader, num_epochs=num_epochs, device=device
    )
    typer.echo(history[-1])


@task_app.command("provenance")
def task_provenance(
    config: Path = typer.Option(..., "--config", "-c"),
    checkpoint: Path | None = typer.Option(None, "--checkpoint", "-w"),
    train_path: Path = typer.Option(..., "--train"),
    val_path: Path = typer.Option(..., "--val"),
    num_epochs: int = typer.Option(3),
    device: str = typer.Option("cpu"),
) -> None:
    """Fine-tune compiler / opt-level / language provenance heads."""
    from contrabin.tasks.compiler_provenance import (
        CompilerProvenanceModel,
        train_compiler_provenance,
    )

    cfg, backbone = _load_task_backbone(config, checkpoint, device)
    train_loader = _task_loaders(cfg, train_path, cfg.training.batch_size, shuffle=True)
    val_loader = _task_loaders(cfg, val_path, cfg.training.eval_batch_size, shuffle=False)
    model = CompilerProvenanceModel(backbone)
    history = train_compiler_provenance(
        model, train_loader, val_loader, num_epochs=num_epochs, device=device
    )
    typer.echo(history[-1])


if __name__ == "__main__":  # pragma: no cover
    app()
