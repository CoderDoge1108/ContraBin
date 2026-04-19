#!/usr/bin/env bash
# End-to-end triplet construction for a directory of C source files.
#
# Usage:
#   ./scripts/build_triplets.sh <src_dir> <out.jsonl> [comment_generator=heuristic]
#
# Environment variables:
#   CONTRABIN_CLANG   path to clang (defaults to `which clang`)

set -euo pipefail

SRC_DIR=${1:?"usage: $0 <src_dir> <out.jsonl> [generator]"}
OUT_FILE=${2:?"usage: $0 <src_dir> <out.jsonl> [generator]"}
GEN=${3:-heuristic}

contrabin build-triplets \
  --input "$SRC_DIR" \
  --output "$OUT_FILE" \
  --comment-generator "$GEN"
