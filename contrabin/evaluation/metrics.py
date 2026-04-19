"""Metric implementations covering the four downstream tasks.

* :func:`token_accuracy` - classification accuracy.
* :func:`retrieval_accuracy` - top-k accuracy for retrieval / clustering tasks.
* :func:`mean_average_precision` - MAP over a query/result gallery.
* :func:`bleu_score` - corpus-level BLEU (sacrebleu).
* :func:`rouge_l` - ROUGE-L F1 (rouge_score).
* :func:`exact_match` - token-level exact-match score (for name recovery).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def token_accuracy(
    predictions: torch.Tensor | Sequence[int], targets: torch.Tensor | Sequence[int]
) -> float:
    """Standard multi-class accuracy."""
    p = torch.as_tensor(predictions).flatten()
    t = torch.as_tensor(targets).flatten()
    if p.numel() == 0:
        return 0.0
    return float((p == t).float().mean().item())


def retrieval_accuracy(
    embeddings: torch.Tensor, labels: Sequence[int], k: int = 1
) -> float:
    """Top-k retrieval accuracy (leave-one-out nearest neighbors).

    For each sample, check whether at least one of its top-k most similar
    embeddings (excluding itself) shares the same label.
    """
    emb = torch.nn.functional.normalize(embeddings, dim=-1)
    sim = emb @ emb.t()
    sim.fill_diagonal_(-float("inf"))
    _, idx = sim.topk(k=k, dim=-1)
    lab = torch.as_tensor(labels)
    neighbors = lab[idx]
    hits = (neighbors == lab.unsqueeze(-1)).any(dim=-1)
    return float(hits.float().mean().item())


def recall_at_k(sim: torch.Tensor, labels: Sequence[int], k: int = 1) -> float:
    """Recall@k over a *precomputed* pairwise similarity matrix.

    ``sim`` is expected to have its diagonal set to ``-inf`` so a query does not
    retrieve itself. ``labels[i]`` is the ground-truth class of row ``i``.
    """
    if sim.numel() == 0:
        return 0.0
    n = sim.size(0)
    k = min(k, sim.size(1))
    _, idx = sim.topk(k=k, dim=-1)
    lab = torch.as_tensor(labels)
    neighbors = lab[idx]
    matches = neighbors == lab.unsqueeze(-1)
    # Mask self from top-k in case the diagonal was not masked.
    self_mask = idx == torch.arange(n, device=idx.device).unsqueeze(-1)
    matches = matches & ~self_mask
    return float(matches.any(dim=-1).float().mean().item())


def mean_reciprocal_rank(sim: torch.Tensor, labels: Sequence[int]) -> float:
    """Mean reciprocal rank of the first same-label neighbor.

    ``sim`` is expected to have its diagonal set to ``-inf`` so self-matches
    are ranked last; we additionally mask any self-indices from the ranked
    neighbor list to be robust when the gallery contains exact duplicates.
    """
    if sim.numel() == 0:
        return 0.0
    n = sim.size(0)
    order = sim.argsort(dim=-1, descending=True)
    lab = torch.as_tensor(labels)
    neighbors = lab[order]
    matches = neighbors == lab.unsqueeze(-1)
    # Suppress the self-match (diagonal was -inf, so self ends up last).
    self_mask = order == torch.arange(n, device=order.device).unsqueeze(-1)
    matches = matches & ~self_mask
    any_match = matches.any(dim=-1)
    first = matches.float().argmax(dim=-1)
    rr = torch.where(
        any_match,
        1.0 / (first.float() + 1.0),
        torch.zeros_like(first, dtype=torch.float32),
    )
    return float(rr.mean().item())


def subtoken_f1(hypotheses: Iterable[Iterable[str]], references: Iterable[Iterable[str]]) -> float:
    """Average F1 over lists of subtoken sets (for function name recovery)."""
    scores: list[float] = []
    for h, r in zip(hypotheses, references, strict=False):
        hs = set(h)
        rs = set(r)
        if not hs and not rs:
            scores.append(1.0)
            continue
        if not hs or not rs:
            scores.append(0.0)
            continue
        inter = hs & rs
        prec = len(inter) / len(hs)
        rec = len(inter) / len(rs)
        scores.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(scores)) if scores else 0.0


def mean_average_precision(
    embeddings: torch.Tensor, labels: Sequence[int]
) -> float:
    """MAP@R: mean average precision using labels as the ground truth.

    Follows the POJ-104 retrieval protocol: for each query we rank all other
    samples by cosine similarity and compute AP using same-label neighbors as
    positives.
    """
    emb = torch.nn.functional.normalize(embeddings, dim=-1)
    sim = (emb @ emb.t()).cpu().numpy()
    lab = np.asarray(labels)
    n = sim.shape[0]
    aps: list[float] = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        scores = sim[i][mask]
        same = (lab[mask] == lab[i]).astype(np.float32)
        if same.sum() == 0:
            continue
        order = np.argsort(-scores)
        ranked = same[order]
        cum_hits = np.cumsum(ranked)
        precision_at_k = cum_hits / (np.arange(ranked.size) + 1)
        ap = (precision_at_k * ranked).sum() / same.sum()
        aps.append(float(ap))
    return float(np.mean(aps)) if aps else 0.0


# ---------------------------------------------------------------------------
# Text generation metrics
# ---------------------------------------------------------------------------


def bleu_score(hypotheses: Iterable[str], references: Iterable[str]) -> float:
    """Corpus BLEU via sacrebleu. Returns a score in [0, 1]."""
    try:
        import sacrebleu

        hyp = list(hypotheses)
        ref = [list(references)]
        return float(sacrebleu.corpus_bleu(hyp, ref).score) / 100.0
    except ImportError:  # pragma: no cover
        return _fallback_bleu(hypotheses, references)


def _fallback_bleu(hypotheses: Iterable[str], references: Iterable[str]) -> float:
    """Trivial unigram-overlap proxy, used if sacrebleu is unavailable."""
    scores = []
    for h, r in zip(hypotheses, references, strict=False):
        hw = h.split()
        rw = set(r.split())
        if not hw:
            scores.append(0.0)
            continue
        scores.append(sum(1 for w in hw if w in rw) / len(hw))
    return float(np.mean(scores)) if scores else 0.0


def rouge_l(hypotheses: Iterable[str], references: Iterable[str]) -> float:
    """ROUGE-L F1 averaged over the corpus."""
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        fs = [
            scorer.score(r, h)["rougeL"].fmeasure
            for h, r in zip(hypotheses, references, strict=False)
        ]
        return float(np.mean(fs)) if fs else 0.0
    except ImportError:  # pragma: no cover
        return _fallback_bleu(hypotheses, references)


def exact_match(hypotheses: Iterable[str], references: Iterable[str]) -> float:
    """Fraction of hypotheses that exactly match (after stripping) the reference."""
    hs = [h.strip() for h in hypotheses]
    rs = [r.strip() for r in references]
    if not hs:
        return 0.0
    return sum(h == r for h, r in zip(hs, rs, strict=False)) / len(hs)
