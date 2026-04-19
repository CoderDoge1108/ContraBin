# FAQ

### Is this a replication package for the paper?

No. It is a **research library** built from the same recipe. The goals are:

- Provide a clean, extensible implementation of the ContraBin method.
- Make it easy to experiment with new backbones, new triplet modalities,
  and new downstream tasks.
- Run end-to-end on a laptop CPU (tests + synthetic data) so you can iterate
  quickly before spinning up a GPU cluster.

Numbers in the paper were produced against proprietary AnghaBench / POJ-104
preprocessing. You should **not** expect bit-for-bit matches.

### What do I need to install to run the tests?

```bash
pip install -e '.[dev]'
pytest -q
```

This uses the offline `contrabin-tiny` backbone and the byte-hash tokenizer,
so there is no network / no `clang` / no HuggingFace download.

### How do I plug in a real HuggingFace backbone?

Set `model.encoder_name` in your YAML config to any causal- or masked-LM
checkpoint. `AnchoredEncoder` and `TrainableEncoder` simply wrap the output
of `AutoModel.from_pretrained(name)`; `build_encoder` handles the rest.

### Do I need `clang` to build triplets?

Only if you want real LLVM IR. For tests and demos, pass
`allow_ir_fallback=True` (or `--allow-fallback` on the CLI) to get a
deterministic fake-IR pass-through.

### How does the simplex interpolation actually help?

Empirically, the paper's Table 5 shows that adding the intermediate InfoNCE
loss on the *interpolated* representation improves every downstream metric
over the naive-only baseline, with non-linear interpolation being the
strongest configuration. Intuitively, the interpolated view forces the
binary encoder to match a *distribution of anchored views* rather than a
single modality, which regularizes the learned manifold.

### Why rename the "reverse engineering" task?

BLEU over source code is a weak signal for semantically-preserving
decompilation. Compiler-provenance recovery is well-posed, has unambiguous
labels, and directly measures how much "production context" the encoder has
internalised. See [tasks.md](tasks.md) for the full rationale.

### Can I use this for non-C binaries?

Yes. The `binary` field is just a string - swap the triplet builder to
produce whatever serialized representation you prefer (x86 disassembly,
LLVM IR, or a hybrid). The models are agnostic as long as you pass a matching
tokenizer.

### How do I log to W&B / TensorBoard?

Install the optional extras:

```bash
pip install 'contrabin[experiment]'
```

and add a callback in `contrabin/training/callbacks.py`. The `Callback`
protocol exposes `on_train_begin`, `on_epoch_begin`, `on_step_end`,
`on_epoch_end`, `on_train_end`, `on_eval_end`.
