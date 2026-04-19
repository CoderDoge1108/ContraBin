from pathlib import Path

import torch

from contrabin.data.comment_generator import HeuristicCommentGenerator
from contrabin.data.compilation import canonicalize_ir, compile_with_fallback
from contrabin.data.datasets import TripletDataset, build_synthetic_triplets
from contrabin.data.loaders import TripletCollator, build_dataloader, build_tokenizer
from contrabin.data.triplet_builder import TripletBuilder, _strip_c_comments


def test_heuristic_comment_generator_captures_patterns():
    gen = HeuristicCommentGenerator()
    s = "void sort_arr(int* arr, int n) { qsort(arr, n, sizeof(int), cmp); }"
    comment = gen.generate(s)
    assert "sort_arr" in comment
    assert comment.endswith(".")


def test_strip_c_comments_removes_both_styles():
    src = "/* block */ int x = 1; // inline\nint y = 2;"
    assert "/*" not in _strip_c_comments(src)
    assert "//" not in _strip_c_comments(src)


def test_canonicalize_ir_strips_debug_and_comments():
    ir = "define i32 @f() {\n  ; a comment\n  ret i32 0, !dbg !12\n}\nattributes #0 = { foo }"
    out = canonicalize_ir(ir)
    assert "!dbg" not in out
    assert "attributes #" not in out
    assert "; a comment" not in out


def test_compile_with_fallback_always_returns_string():
    ir = compile_with_fallback("int f(int x){return x+1;}", dummy_on_failure=True)
    assert isinstance(ir, str) and len(ir) > 0


def test_triplet_builder_writes_records(tmp_path: Path):
    (tmp_path / "a.c").write_text("int add(int a,int b){return a+b;}")
    (tmp_path / "b.c").write_text("int mul(int a,int b){return a*b;}")
    builder = TripletBuilder(allow_ir_fallback=True)
    out = tmp_path / "triplets.jsonl"
    n = builder.write_jsonl(builder.iter_from_directory(tmp_path), out)
    assert n == 2
    ds = TripletDataset(out)
    row = ds[0]
    assert set(row.keys()) >= {"source", "binary", "comment"}


def test_build_synthetic_triplets(tmp_path: Path):
    path = tmp_path / "syn.jsonl"
    n = build_synthetic_triplets(path, n=8, seed=0)
    assert n == 8
    ds = TripletDataset(path)
    assert len(ds) == 8


def test_tokenizer_and_collator_shapes(tmp_path: Path):
    path = tmp_path / "syn.jsonl"
    build_synthetic_triplets(path, n=8, seed=0)
    ds = TripletDataset(path)
    tok = build_tokenizer("contrabin-tiny", vocab_size=64)
    collator = TripletCollator(
        tokenizer=tok,
        source_max_length=32,
        binary_max_length=32,
        comment_max_length=12,
    )
    loader = build_dataloader(ds, collator, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch["source"]["input_ids"].shape == (4, 32)
    assert batch["binary"]["input_ids"].shape == (4, 32)
    assert batch["comment"]["input_ids"].shape == (4, 12)
    assert torch.is_tensor(batch["source"]["attention_mask"])
