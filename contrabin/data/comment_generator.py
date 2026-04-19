"""Generate concise, LLM-style comments for source code snippets.

One of the paper's main empirical findings is that **LLM-generated comments
outperform human-written comments** for binary representation learning. We
therefore provide two implementations:

1. :class:`HeuristicCommentGenerator`
   Extracts the function signature, identifies common patterns (loops, hash
   calls, recursion, pointer use) and emits a short English summary. This runs
   offline with no API key and is good enough for unit tests and small-scale
   experiments.

2. :class:`OpenAICommentGenerator` (optional)
   Calls an OpenAI-compatible API (via the ``openai`` package) to produce
   richer summaries. Import lazily to avoid a hard dependency.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass


class CommentGenerator(ABC):
    """Abstract base class for comment generators."""

    @abstractmethod
    def generate(self, source: str) -> str:  # pragma: no cover - abstract
        ...

    def batch_generate(self, sources: Iterable[str]) -> list[str]:
        return [self.generate(s) for s in sources]


# ---------------------------------------------------------------------------
# Heuristic (offline) generator
# ---------------------------------------------------------------------------


_PATTERNS = [
    (re.compile(r"\bqsort\b|\bsort\b"), "sorts an array"),
    (re.compile(r"\bhash\b|\bmd5\b|\bsha1\b|\bsha256\b"), "computes a cryptographic hash"),
    (re.compile(r"\bstrcpy\b|\bstrncpy\b|\bstrcat\b"), "copies or concatenates strings"),
    (re.compile(r"\bstrlen\b"), "computes a string length"),
    (re.compile(r"\bfopen\b|\bfread\b|\bfwrite\b|\bfclose\b"), "performs file I/O"),
    (re.compile(r"\bmalloc\b|\bcalloc\b|\bfree\b"), "manages dynamic memory"),
    (re.compile(r"\bprintf\b|\bfprintf\b"), "prints formatted output"),
    (re.compile(r"\bsocket\b|\brecv\b|\bsend\b|\bconnect\b"), "handles network I/O"),
    (re.compile(r"\bfor\s*\("), "iterates over a range"),
    (re.compile(r"\bwhile\s*\("), "loops until a condition holds"),
    (re.compile(r"\bif\s*\(.*\)\s*return"), "performs early-exit validation"),
    (re.compile(r"\brecursion\b|\brecursive\b"), "computes a value recursively"),
]

_SIGNATURE_RE = re.compile(
    r"^\s*(?:static\s+|inline\s+|extern\s+)*"
    r"(?P<ret>[\w\s\*]+?)\s+(?P<name>[A-Za-z_]\w*)\s*\((?P<args>[^)]*)\)",
    re.MULTILINE,
)


@dataclass
class HeuristicCommentGenerator(CommentGenerator):
    """Offline heuristic generator.

    Produces comments of the form:
    ``"<fn_name>: <behaviour>. Takes <n> argument(s)."``
    """

    max_length: int = 96
    include_signature: bool = True

    def generate(self, source: str) -> str:
        m = _SIGNATURE_RE.search(source)
        parts: list[str] = []
        if m and self.include_signature:
            name = m.group("name").strip()
            args = [a.strip() for a in m.group("args").split(",") if a.strip() not in {"", "void"}]
            parts.append(f"{name}: function with {len(args)} argument(s)")
        behaviors = []
        for pattern, description in _PATTERNS:
            if pattern.search(source):
                behaviors.append(description)
        if behaviors:
            parts.append("; ".join(dict.fromkeys(behaviors[:3])))
        if not parts:
            parts.append("utility function")
        comment = ". ".join(parts)
        return comment[: self.max_length].rstrip(",. ") + "."


# ---------------------------------------------------------------------------
# OpenAI-backed generator (optional)
# ---------------------------------------------------------------------------


class OpenAICommentGenerator(CommentGenerator):
    """Calls an OpenAI-compatible API to generate concise code comments.

    Requires ``pip install 'contrabin[llm]'`` and an ``OPENAI_API_KEY``
    environment variable.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 48,
        temperature: float = 0.2,
        system_prompt: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "OpenAICommentGenerator requires the `openai` package. "
                "Install with: pip install 'contrabin[llm]'"
            ) from e
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt or (
            "You are a senior C developer. Summarize the following C function "
            "in one concise sentence (<= 20 words). Focus on behavior, not "
            "syntax. Do not prepend 'This function'."
        )

    def generate(self, source: str) -> str:  # pragma: no cover - network
        from openai import OpenAI

        client = OpenAI()
        rsp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": source},
            ],
        )
        return rsp.choices[0].message.content.strip()


__all__ = ["CommentGenerator", "HeuristicCommentGenerator", "OpenAICommentGenerator"]
