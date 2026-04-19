# Method

This document maps the code in `contrabin/` back to the equations and
algorithms of the paper.

## 1. Encoders (Section 2.1)

```
contrabin/models/encoders.py
```

The paper uses two encoders initialized from the same backbone:

- **Anchored encoder** `f_M^a` (frozen) encodes source code and comments.
- **Trainable encoder** `f_M^t` updates during training and encodes binary / IR.

The CLS-token representation is pooled (`target_token_idx = 0`). A factory
helper builds either encoder by name; pass `contrabin-tiny` for the offline
test backbone.

## 2. Projection heads (Section 2.1)

```
contrabin/models/heads.py
```

Both linear and non-linear projection heads are available. The linear head
implements Eq. 1 with a residual GELU block; the non-linear head stacks two
additional FC + GELU blocks.

## 3. Primary contrastive loss (Section 2.1, Eqs. 2-6)

```
contrabin/losses/contrastive.py
```

We compute:

- logits   = `H_{c/s} · H_b^T / τ`
- targets  = `softmax((H_{c/s} H_{c/s}^T + H_b H_b^T) / 2, dim=-1)`
- loss     = CE(logits, targets) averaged symmetrically

The symmetric term stabilizes training (CLIP-style). Note that the paper's
equation uses a *single* direction; our implementation averages the two
directions, which is numerically equivalent in expectation but lower variance.

## 4. Simplex interpolation (Section 2.2, Eqs. 7-9)

```
contrabin/models/interpolation.py
```

Given two anchored projections `H1` and `H2`, we define:

```
Γ(A, B; λ) = λ A + (1 - λ) B
```

Two interpolators:

- **Linear** predicts a *scalar* λ ∈ [0, 1] from `H1 + H2`.
- **Non-linear** predicts a *per-dimension* λ of the same shape as the inputs.

Both are wrapped by `SimplexInterpolationModule` which dispatches on the
current curriculum stage.

## 5. Intermediate contrastive loss (Section 2.2, Eq. 10)

```
contrabin/losses/intermediate.py
```

Symmetric InfoNCE between the interpolated intermediate representation
`H_i` and the matched binary embedding `H_b`, with in-batch negatives.

## 6. Training loop (Algorithm 1)

```
contrabin/training/trainer.py
```

`PretrainTrainer.fit` implements the algorithm:

1. For each epoch, pick an interpolation stage via `CurriculumScheduler`
   (`naive` -> `linear` -> `nonlinear`).
2. For each batch, encode source / comment / binary, project each through
   its modality-specific head.
3. Compute the primary loss (source↔binary and comment↔binary, summed).
4. If the stage is not `naive`, compute the interpolated representation and
   add the intermediate InfoNCE loss.
5. Back-propagate, clip gradients, and step the AdamW optimizer with two
   parameter groups (encoder LR vs. head LR).

## 7. Composite forward pass

```
contrabin/models/contrabin.py
```

`ContraBinModel.forward` returns a `ContraBinOutput` dataclass:

```python
@dataclass
class ContraBinOutput:
    source: Tensor
    binary: Tensor
    comment: Tensor
    intermediate: Tensor | None  # only populated at non-naive stages
```

This keeps the trainer generic across curriculum stages.

## References

- Original paper: [arXiv:2210.05102](https://arxiv.org/abs/2210.05102)
- Zenodo archive (skeleton code): https://zenodo.org/records/15219264
