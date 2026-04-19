# ContraBin

Contrastive pre-training of binary-code representations with simplex interpolation.

[![CI](https://github.com/CoderDoge1108/ContraBin/actions/workflows/ci.yml/badge.svg)](https://github.com/CoderDoge1108/ContraBin/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-arXiv%3A2210.05102-b31b1b.svg)](https://arxiv.org/abs/2210.05102)

ContraBin learns transferable representations of binary code by contrastively
aligning three views of the same program - **source**, **binary / LLVM IR**,
and a short **natural-language comment** - through a two-stage curriculum of
simplex interpolation. This repository is an open, research-grade
reimplementation of:

> Zhang, Y., Huang, C., Zhang, Y., Shao, H., Leach, K., & Huang, Y. (2025).
> *Pre-Training Representations of Binary Code Using Contrastive Learning.*
> Transactions on Machine Learning Research. [arXiv:2210.05102](https://arxiv.org/abs/2210.05102)

> [!NOTE]
> This repository is **not a replication package**. It is a clean, extensible
> library that makes it easy to **research on top of** the ContraBin recipe:
> swap backbones, redesign downstream tasks, or replace the triplet data
> pipeline. See [docs/method.md](docs/method.md) for how the code maps to the
> paper and [docs/tasks.md](docs/tasks.md) for the rationale behind the
> revised downstream evaluation suite.

---

## Highlights

- **Modular architecture** - encoders, projection heads, simplex interpolation,
  and contrastive losses are all independent modules you can mix and match.
- **Offline-friendly** - every component has a tiny test backbone
  (`contrabin-tiny`) so the full training loop runs on a laptop CPU.
- **Revised downstream tasks** - four well-posed evaluations that are more
  research-relevant than the originals (see [Downstream tasks](#downstream-tasks)).
- **Pure CLI workflow** - `contrabin build-triplets | pretrain | embed | task ...`.
- **Modern Python packaging** - `pyproject.toml`, `pydantic` configs, `typer`
  CLI, `ruff` + `mypy` + `pytest` with 30+ tests.

---

## Architecture

```
 ┌──────────┐           ┌──────────┐           ┌──────────┐
 │  Source  │ frozen    │  Binary  │ trainable │ Comment  │ frozen
 │ encoder  │──┐        │ encoder  │──┐        │ encoder  │──┐
 └──────────┘  │        └──────────┘  │        └──────────┘  │
               ▼                      │                      ▼
        ┌────────────┐          ┌───────────┐          ┌────────────┐
        │ src head   │          │ bin head  │          │ cmt head   │
        └─────┬──────┘          └─────┬─────┘          └─────┬──────┘
              │                       │                      │
              ├────────── CLIP loss ──┤── CLIP loss ─────────┤
              │                                              │
              └──── Simplex interpolation Γ(src, cmt; λ) ────┘
                           │
                           ▼
                 InfoNCE loss against binary
                 (curriculum: naive -> linear -> nonlinear)
```

- Two **anchored** (frozen) encoders share weights to produce source and
  comment embeddings.
- A separate **trainable** encoder produces the binary / IR embedding.
- At training time, the two anchored embeddings are mixed by a learned
  simplex interpolation module to form an *intermediate view*, which is then
  aligned with the binary embedding via an InfoNCE objective.
- A scalar curriculum schedules interpolation from *naive* (no mixing) to
  *linear* (scalar λ) to *nonlinear* (element-wise λ).

See [docs/method.md](docs/method.md) and [docs/training.md](docs/training.md)
for the full mathematical formulation and Algorithm 1 of the paper.

---

## Quick start

```bash
# 1. Install (editable + dev extras).
pip install -e '.[dev]'

# 2. Smoke-test the whole stack on synthetic data (no network / no clang needed).
pytest -q

# 3. Drive end-to-end training on synthetic data with the CPU-friendly config.
contrabin make-synthetic --output data/processed/train.jsonl -n 64
contrabin make-synthetic --output data/processed/val.jsonl   -n 16 --seed 1
contrabin pretrain --config configs/smoke.yaml

# 4. Build real triplets from a directory of C files (requires clang).
contrabin build-triplets \
    --input data/raw/anghabench \
    --output data/processed/triplets.jsonl \
    --comment-generator heuristic

# 5. Pre-train with the full curriculum.
contrabin pretrain --config configs/pretrain.yaml

# 6. Run any of the four downstream tasks.
contrabin task retrieve     --config configs/pretrain.yaml --gallery data/processed/gallery.jsonl --checkpoint outputs/contrabin/final.pt
contrabin task name-recovery --config configs/pretrain.yaml --train data/.../train.jsonl --val data/.../val.jsonl
contrabin task summarize     --config configs/pretrain.yaml --train data/.../train.jsonl --val data/.../val.jsonl
contrabin task provenance    --config configs/pretrain.yaml --train data/.../train.jsonl --val data/.../val.jsonl
```

---

## Downstream tasks

The paper evaluates four tasks. Two of them (function name recovery and binary
summarization) are well-posed and we keep them. The other two are reformulated
to be easier to reproduce and to more directly measure *representation
quality* rather than *fine-tuning-head capacity*:

| Task | Paper framing | Our framing | Metrics |
| ---- | ------------- | ----------- | ------- |
| Binary functional similarity | POJ-104 104-way classification | **Retrieval** over POJ-104 labels with frozen embeddings + optional linear probe | mAP, MRR, Recall@{1,5,10} |
| Function name recovery | Multi-label subtoken classification | Same | Exact match, subtoken F1 |
| Binary summarization | Binary -> NL seq2seq | Same | BLEU, ROUGE-L |
| Reverse engineering | Binary -> C source (BLEU) | **Compiler provenance recovery** (compiler / opt-level / language) | Per-head and joint accuracy |

The full rationale is in [docs/tasks.md](docs/tasks.md).

---

## Repository layout

```
contrabin/
├── contrabin/                  # Library code
│   ├── config.py               # Pydantic configs + YAML loading
│   ├── data/                   # Triplet builder, clang frontend, datasets
│   ├── models/                 # Encoders, heads, simplex interpolation
│   ├── losses/                 # CLIP-style + InfoNCE objectives
│   ├── training/               # Trainer, curriculum scheduler, callbacks
│   ├── tasks/                  # Four downstream tasks
│   ├── evaluation/             # Metrics (mAP, MRR, BLEU, ROUGE, F1, ...)
│   ├── utils/                  # Logging, IO, seeding, visualization
│   └── cli.py                  # Typer CLI (contrabin ...)
├── configs/                    # YAML configs (pretrain / smoke / finetune)
├── tests/                      # pytest suite (CPU-only, offline)
├── docs/                       # Method, data, training, tasks, FAQ
├── scripts/                    # Shell wrappers (build_triplets.sh, ...)
├── examples/                   # Minimal runnable usage examples
├── notebooks/                  # Jupyter quickstart + analysis notebooks
├── pyproject.toml
├── CHANGELOG.md
└── README.md
```

---

## Citing

If you use ContraBin in your research, please cite:

```bibtex
@article{zhang2025contrabin,
  title   = {Pre-Training Representations of Binary Code Using Contrastive Learning},
  author  = {Zhang, Yifan and Huang, Chengzhi and Zhang, Yichi and Shao, Haoran and Leach, Kevin and Huang, Yu},
  journal = {Transactions on Machine Learning Research},
  year    = {2025},
  url     = {https://arxiv.org/abs/2210.05102}
}
```

---

## License

MIT. See [LICENSE](LICENSE).
