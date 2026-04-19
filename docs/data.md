# Data pipeline

ContraBin trains on triplets of the form `(source, binary, comment)`. This
document describes how to produce them from scratch.

## Triplet schema

Every triplet is a single JSON record:

```json
{
  "source":  "int add(int a, int b) { return a + b; }",
  "binary":  "define i32 @add(i32 %a, i32 %b) { %0 = add i32 %a, %b ... }",
  "comment": "add: function with 2 argument(s). utility function.",
  "metadata": {
    "path": "anghabench/add.c",
    "label": 7,
    "compiler": "clang",
    "opt_level": "O2",
    "language": "c"
  }
}
```

- `source` and `binary` are stripped of `//` and `/* */` comments by the
  default triplet builder.
- `comment` is produced by a `CommentGenerator` (heuristic by default).
- `metadata` is free-form but is the carrier for downstream-task labels;
  e.g., POJ-104 `label` for retrieval, `name` for function-name recovery,
  `compiler` / `opt_level` / `language` for provenance recovery.

## Compilation

`contrabin/data/compilation.py` wraps `clang -S -emit-llvm`. Set the
`CONTRABIN_CLANG` environment variable to point to a specific clang binary.

For CI and unit-test environments without clang installed, pass
`allow_ir_fallback=True` to `TripletBuilder` (or
`--allow-fallback` on the CLI). A deterministic hash-based IR string is
synthesized from the source; it is **not** semantically faithful and is only
intended for shape-matching in tests.

## Comment generation

Two generators are provided:

- **Heuristic** (default): pattern-matches common C idioms (sort, hash, I/O,
  loops, pointer ops, …) and emits a one-sentence summary. Runs offline.
- **OpenAI**: calls the chat-completions endpoint of an OpenAI-compatible API
  to produce higher-quality comments. Requires `pip install 'contrabin[llm]'`
  and an `OPENAI_API_KEY` environment variable.

The paper's finding is that **LLM-generated comments improve downstream
performance, whereas human-written comments hurt** (presumably because they
leak task-specific information). Keep this in mind when mixing sources.

## Building triplets

```bash
contrabin build-triplets \
    --input  data/raw/anghabench \
    --output data/processed/triplets.jsonl \
    --pattern '**/*.c' \
    --optimization -O0 \
    --comment-generator heuristic
```

## Synthetic triplets

For CI and end-to-end tests:

```bash
contrabin make-synthetic --output data/processed/train.jsonl -n 64
contrabin make-synthetic --output data/processed/val.jsonl   -n 16 --seed 1
```

The resulting triplets pair a toy source snippet with a matching toy IR and
a short natural-language description. The byte-hash tokenizer and
`contrabin-tiny` backbone let you run the whole pipeline in under a second
on a laptop CPU.
