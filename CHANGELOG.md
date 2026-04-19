# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-18

Initial open-source release of the ContraBin research library.

### Added
- `contrabin.config` - Pydantic-based configuration system with YAML loading.
- `contrabin.data` - Triplet data pipeline:
  - Compile-time (`clang`) frontend with a pure-Python IR fallback for CI.
  - Heuristic and OpenAI-backed comment generators.
  - `TripletDataset`, `TripletCollator`, offline byte-hash tokenizer.
- `contrabin.models`:
  - Anchored (frozen) and trainable encoders, plus a lightweight `TinyEncoder`
    for offline tests (`contrabin-tiny` backbone name).
  - Linear / non-linear projection heads.
  - Simplex interpolation module (naive / linear / non-linear).
  - Composite `ContraBinModel`.
- `contrabin.losses` - CLIP-style primary contrastive loss and
  symmetric InfoNCE intermediate loss.
- `contrabin.training` - `PretrainTrainer` implementing Algorithm 1 with a
  curriculum scheduler and pluggable callbacks.
- `contrabin.tasks` - four downstream tasks:
  - Binary functional similarity **retrieval** (replaces POJ-104 classification).
  - Function name recovery (multi-label subtoken head).
  - Binary code summarization (seq2seq).
  - **Compiler provenance recovery** (replaces binary-to-C "reverse engineering").
- `contrabin.evaluation` - metrics: mAP, MRR, Recall@k, BLEU, ROUGE-L,
  subtoken F1, exact match, token accuracy.
- `contrabin.cli` - Typer CLI:
  `contrabin {version, build-triplets, make-synthetic, pretrain, evaluate, embed, task ...}`.
- `tests/` - 30+ unit and integration tests that exercise the whole stack on
  synthetic data in under 30 seconds on a laptop CPU.
- GitHub Actions CI, pre-commit (ruff), and `mypy`.

### Changed - relative to the Zenodo release
- Flat script layout (`main.py`, `dataset.py`, `configs.py`) replaced by a
  proper Python package under `contrabin/`.
- Hard-coded class-variable `Config` replaced by structured
  `DataConfig` / `ModelConfig` / `OptimConfig` / `TrainingConfig` / `CurriculumConfig`.
- Encoders decoupled from the training loop: any HuggingFace or custom backbone
  can be dropped in via `build_encoder`.
- POJ-104 "classification" head replaced by a retrieval protocol + linear probe.
- Binary-to-C "reverse engineering" replaced by compiler provenance recovery.
- Ad-hoc metric implementations replaced by `sacrebleu` / `rouge_score` /
  dedicated retrieval metrics.

### Fixed
- `NameRecoveryModel.encode_labels` dead re-initialization of the label tensor.
- Unreachable import in the old `reverse_engineering.py`.
- Stable determinism via `seed_everything` in `utils.seed`.

[0.2.0]: https://github.com/CoderDoge1108/ContraBin/releases/tag/v0.2.0
