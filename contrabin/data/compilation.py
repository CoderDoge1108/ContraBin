"""Compile C source snippets into LLVM IR (used as the "binary" modality).

The paper accepts either lifted assembly or LLVM IR as the binary
representation; IR is the easier and more reproducible choice because it is
deterministic given a fixed clang version. We wrap ``clang`` and normalize the
output (removing comments, attribute groups, and debug metadata) so that
downstream tokenizers see compact, canonical IR.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from contrabin.utils.logging import get_logger

logger = get_logger(__name__)


class IRCompilationError(RuntimeError):
    """Raised when clang fails to emit LLVM IR for a snippet."""


_DEBUG_METADATA_RE = re.compile(r", ?!dbg !\d+")
_ATTR_GROUP_RE = re.compile(r"attributes #\d+ = \{[^}]*\}")
_SEMICOLON_COMMENT_RE = re.compile(r";[^\n]*")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def canonicalize_ir(ir_text: str, keep_comments: bool = False) -> str:
    """Strip optimization hints, attribute groups, and debug info from IR."""
    text = ir_text
    if not keep_comments:
        text = _SEMICOLON_COMMENT_RE.sub("", text)
    text = _DEBUG_METADATA_RE.sub("", text)
    text = _ATTR_GROUP_RE.sub("", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def _find_clang() -> str:
    path = os.environ.get("CONTRABIN_CLANG") or shutil.which("clang")
    if not path:
        raise IRCompilationError(
            "Could not find `clang`. Install LLVM/clang or set CONTRABIN_CLANG=/path/to/clang."
        )
    return path


def compile_c_to_ir(
    source: str,
    optimization: str = "-O0",
    clang_args: list[str] | None = None,
    canonicalize: bool = True,
    timeout: float = 30.0,
) -> str:
    """Compile a C source string to human-readable LLVM IR.

    Raises :class:`IRCompilationError` on compiler failure.
    """
    clang = _find_clang()
    args = [clang, optimization, "-S", "-emit-llvm", "-x", "c", "-", "-o", "-"]
    if clang_args:
        args.extend(clang_args)
    try:
        result = subprocess.run(
            args,
            input=source,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise IRCompilationError(f"clang timed out after {timeout}s") from e
    if result.returncode != 0:
        stderr = result.stderr.strip().splitlines()[-1] if result.stderr else ""
        raise IRCompilationError(f"clang failed: {stderr}")
    ir = result.stdout
    return canonicalize_ir(ir) if canonicalize else ir


def compile_c_file_to_ir(path: Path, **kwargs) -> str:
    """Convenience wrapper around :func:`compile_c_to_ir` for files."""
    return compile_c_to_ir(Path(path).read_text(), **kwargs)


def compile_directory_to_ir(
    src_dir: Path,
    out_dir: Path,
    pattern: str = "**/*.c",
    **kwargs,
) -> int:
    """Batch-compile every C file matching ``pattern`` under ``src_dir``.

    Successful outputs are written next to the source files under ``out_dir``
    with a ``.ll`` suffix. Failures are logged but do not abort the run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    for src in src_dir.glob(pattern):
        rel = src.relative_to(src_dir).with_suffix(".ll")
        target = out_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            ir = compile_c_to_ir(src.read_text(errors="ignore"), **kwargs)
        except IRCompilationError as e:
            logger.warning("Failed to compile %s: %s", src, e)
            continue
        target.write_text(ir)
        ok += 1
    return ok


def compile_with_fallback(source: str, dummy_on_failure: bool = False, **kwargs) -> str:
    """Compile or return a synthesized IR string if clang is unavailable.

    This is useful in CI / unit tests where we do not want a hard dependency on
    a local LLVM installation.
    """
    try:
        return compile_c_to_ir(source, **kwargs)
    except IRCompilationError:
        if not dummy_on_failure:
            raise
        return _synthesize_dummy_ir(source)


def _synthesize_dummy_ir(source: str) -> str:
    """Produce a deterministic fake IR string from a C source.

    NOT semantically equivalent; used purely for shape-matching in tests.
    """
    import hashlib

    h = hashlib.sha1(source.encode(), usedforsecurity=False).hexdigest()[:8]
    n = max(1, len(source.split()) // 5)
    body = "\n".join(f"  %{i} = add i32 {i}, {i + 1}" for i in range(n))
    return f"define i32 @fn_{h}() {{\n{body}\n  ret i32 0\n}}"


__all__ = [
    "IRCompilationError",
    "canonicalize_ir",
    "compile_c_file_to_ir",
    "compile_c_to_ir",
    "compile_directory_to_ir",
    "compile_with_fallback",
]

# keep TemporaryDirectory import for scripts that subclass this module
_ = TemporaryDirectory
