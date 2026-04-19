# Downstream tasks

The paper reports four downstream tasks. We keep two of them and revise the
other two. The overall design goal is:

> **Make the downstream evaluation measure what the pre-training objective
> actually learns, not the post-hoc fine-tuning head.**

## Task overview

| Task                                | Module                                         | Metrics                                      |
| ----------------------------------- | ---------------------------------------------- | -------------------------------------------- |
| Binary functional similarity        | `contrabin.tasks.binary_retrieval`             | mAP, MRR, Recall@{1, 5, 10}                  |
| Function name recovery              | `contrabin.tasks.name_recovery`                | Exact match, subtoken F1                     |
| Binary summarization                | `contrabin.tasks.summarization`                | BLEU, ROUGE-L                                |
| Compiler provenance recovery        | `contrabin.tasks.compiler_provenance`          | Per-head + joint accuracy                    |

## Why revise two of the tasks?

### 1. POJ-104 classification -> functional similarity retrieval

*Paper framing.* 104-way softmax head over algorithmic categories.

*Problem.* A 104-way cross-entropy objective rewards fine-tuning-head
capacity as much as representation quality. Two models with very different
embedding geometries can both achieve high classification accuracy if the
head is powerful enough.

*Our framing.* We freeze the ContraBin binary encoder and evaluate the
**retrieval** problem: for each binary in the gallery, rank every other
binary by cosine similarity. A binary is a positive if it shares a POJ-104
label. We report:

- **mAP** - matches the original POJ-104 retrieval protocol (used by Asm2Vec,
  jTrans, and SAFE in subsequent work).
- **Recall@{1, 5, 10}** - standard for visual-retrieval-style comparisons.
- **MRR** - sensitive to whether the top-1 result is correct.

For apples-to-apples comparison with the paper, we also provide
`LinearProbeClassifier`, which freezes the backbone and trains only a single
`nn.Linear` layer. Linear-probing is the de-facto standard way to compare
self-supervised backbones (SimCLR, MoCo, CLIP, ...).

### 2. Reverse engineering -> compiler provenance recovery

*Paper framing.* Binary -> C source seq2seq, scored with BLEU.

*Problem.* BLEU over source code is a *noisy* signal: semantically
equivalent decompilations can score near-zero (renamed locals, different
loops), and stylistically similar but semantically wrong output can score
highly. Moreover, real binary-to-source lifting is a research field on its
own (Ghidra + ML, DIRE, LLM4Decompile, ...) and evaluating it well requires
test-execution infrastructure that is outside the scope of this repo.

*Our framing.* We ship **compiler provenance recovery**: given a binary /
IR blob, predict three pieces of production metadata -

- **Compiler**: `{gcc, clang, msvc, icc, unknown}`
- **Opt level**: `{O0, O1, O2, O3, Os, Oz}`
- **Source language**: `{c, cpp, rust, go, fortran, unknown}`

This is:

- **Deterministic** - no BLEU, no decoding, no LM. Ground truth is unambiguous.
- **Well-motivated** - used in downstream security and forensics pipelines
  (libc fingerprinting, malware attribution, triage of compiled artefacts).
- **Directly informative** about how much "production context" the encoder
  has absorbed, which is exactly what you want to measure for a
  self-supervised binary representation.

## Running the tasks

Each task has a CLI entry point:

```bash
contrabin task retrieve       --config ... --checkpoint ... --gallery gallery.jsonl
contrabin task name-recovery  --config ... --checkpoint ... --train  train.jsonl --val val.jsonl
contrabin task summarize      --config ... --checkpoint ... --train  train.jsonl --val val.jsonl
contrabin task provenance     --config ... --checkpoint ... --train  train.jsonl --val val.jsonl
```

And a Python API:

```python
from contrabin.tasks.binary_retrieval import (
    extract_binary_embeddings,
    evaluate_retrieval,
)
embeddings, labels = extract_binary_embeddings(model, loader)
result = evaluate_retrieval(embeddings, labels)
print(result)   # RetrievalResult(mAP=..., mrr=..., recall_at_1=..., ...)
```

## Adding a new task

Drop a new file in `contrabin/tasks/` exporting a model class and a
`train_*` function, then re-export it from `contrabin/tasks/__init__.py` and
add a CLI subcommand in `contrabin/cli.py`. `tests/test_tasks.py` shows the
minimal contract each task needs to satisfy.
