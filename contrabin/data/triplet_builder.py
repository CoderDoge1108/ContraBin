"""Build ``(source, binary, comment)`` triplets from a corpus of C sources.

Usage example (see ``scripts/build_triplets.sh``):

.. code-block:: bash

    contrabin build-triplets \\
        --input data/raw/anghabench \\
        --output data/processed/triplets.jsonl \\
        --comment-generator heuristic
"""

from __future__ import annotations

import re as _re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from contrabin.data.comment_generator import CommentGenerator, HeuristicCommentGenerator
from contrabin.data.compilation import IRCompilationError, compile_with_fallback
from contrabin.utils.io import write_jsonl
from contrabin.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TripletRecord:
    """A single ``(source, binary, comment)`` triplet.

    ``metadata`` is free-form and can carry, e.g., the originating file path,
    optimization level, or a label for a downstream classification task.
    """

    source: str
    binary: str
    comment: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "binary": self.binary,
            "comment": self.comment,
            "metadata": self.metadata,
        }


@dataclass
class TripletBuilder:
    """Orchestrates compilation + comment generation for a corpus of snippets."""

    comment_generator: CommentGenerator = field(default_factory=HeuristicCommentGenerator)
    optimization: str = "-O0"
    allow_ir_fallback: bool = True
    strip_source_comments: bool = True

    def iter_from_directory(
        self,
        src_dir: Path,
        pattern: str = "**/*.c",
    ) -> Iterator[TripletRecord]:
        """Yield triplets for every C source file under ``src_dir``."""
        src_dir = Path(src_dir)
        for path in src_dir.glob(pattern):
            try:
                source = path.read_text(errors="ignore")
            except OSError as e:
                logger.warning("Skipping %s: %s", path, e)
                continue
            try:
                rec = self.build_from_source(source, metadata={"path": str(path)})
            except IRCompilationError as e:
                logger.warning("Skip %s (compilation failure): %s", path, e)
                continue
            yield rec

    def build_from_source(self, source: str, metadata: dict | None = None) -> TripletRecord:
        meta = dict(metadata or {})
        cleaned = _strip_c_comments(source) if self.strip_source_comments else source
        ir = compile_with_fallback(
            cleaned,
            dummy_on_failure=self.allow_ir_fallback,
            optimization=self.optimization,
        )
        comment = self.comment_generator.generate(cleaned)
        return TripletRecord(source=cleaned, binary=ir, comment=comment, metadata=meta)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def write_jsonl(self, records: Iterator[TripletRecord], out_path: Path) -> int:
        return write_jsonl(out_path, (r.to_dict() for r in records))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_BLOCK_COMMENT_RE = _re.compile(r"/\*.*?\*/", _re.DOTALL)
_LINE_COMMENT_RE = _re.compile(r"//[^\n]*")


def _strip_c_comments(source: str) -> str:
    """Remove ``/* ... */`` and ``// ...`` comments."""
    source = _BLOCK_COMMENT_RE.sub("", source)
    source = _LINE_COMMENT_RE.sub("", source)
    return source.strip()


__all__ = ["TripletBuilder", "TripletRecord"]
