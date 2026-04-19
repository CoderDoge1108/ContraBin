# Training

## Curriculum

ContraBin uses a three-stage curriculum over the *interpolation type*:

| Stage     | Epochs (default)              | Behavior                                               |
| --------- | ----------------------------- | ------------------------------------------------------ |
| `naive`   | `primary_epochs`   (e.g. 10)  | No interpolation; only the primary CLIP-style loss.    |
| `linear`  | `linear_epochs`    (e.g. 5)   | Adds a scalar-λ intermediate InfoNCE loss.             |
| `nonlinear` | `nonlinear_epochs` (e.g. 5) | Adds an element-wise-λ intermediate InfoNCE loss.      |

Configure it in YAML:

```yaml
training:
  curriculum:
    primary_epochs: 10
    linear_epochs: 5
    nonlinear_epochs: 5
```

## Optimizer

Two parameter groups:

- Encoder LR  (`optim.lr`, default `1e-5`)    for the trainable binary encoder.
- Head LR     (`optim.head_lr`, default `1e-3`) for the three projection heads
  and the simplex interpolation module.

We use `AdamW` with `weight_decay = 1e-3` and a cosine LR schedule with warmup
by default. Switch schedulers via `optim.scheduler` (`cosine` / `linear` /
`constant`).

## Mixed precision, gradient accumulation, checkpointing

- `training.mixed_precision`      - `no` / `fp16` / `bf16`
- `training.grad_accumulation_steps`
- `training.max_grad_norm`        - global-norm clipping
- `training.save_every_n_steps`   - checkpointing cadence

`PretrainTrainer.save(path)` writes a dictionary of:

```python
{"state_dict": ..., "config": ..., "history": [...]}
```

so that restarting is a single `trainer.load(path)` call.

## Reproducibility

- `seed_everything(cfg.training.seed)` seeds `random`, `numpy`, and `torch`.
- All configs are dumped to the output directory at the start of a run.
- The YAML-first design makes it easy to diff configurations across runs.

## Logging

The default callback emits step-level training loss and epoch-level
train/val loss through the standard `logging` module. For richer metrics
(W&B / TensorBoard), install the `experiment` extras:

```bash
pip install 'contrabin[experiment]'
```

and add your own callback subclass.

## Algorithm 1 in code

The whole thing fits in ~30 lines in `contrabin/training/trainer.py`:

```python
for epoch in range(total_epochs):
    stage = curriculum.stage_for_epoch(epoch)
    for batch in train_loader:
        out = model(batch, stage=stage)
        loss = primary_loss(out.source, out.binary) + \
               primary_loss(out.comment, out.binary)
        if stage != "naive":
            loss += intermediate_loss(out.intermediate, out.binary)
        loss.backward()
        clip_grad_norm_(...); optimizer.step(); scheduler.step()
```
