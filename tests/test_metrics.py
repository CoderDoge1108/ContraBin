import torch

from contrabin.evaluation.metrics import (
    bleu_score,
    exact_match,
    mean_average_precision,
    mean_reciprocal_rank,
    recall_at_k,
    retrieval_accuracy,
    rouge_l,
    subtoken_f1,
    token_accuracy,
)


def test_token_accuracy():
    assert token_accuracy([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0
    assert abs(token_accuracy([0, 0, 0], [0, 1, 2]) - 1 / 3) < 1e-6


def test_retrieval_accuracy_is_perfect_for_clones():
    emb = torch.tensor(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
        ]
    )
    labels = [0, 0, 1, 1]
    assert retrieval_accuracy(emb, labels, k=1) == 1.0


def test_map_at_r():
    # two tight clusters => MAP should be 1.0
    emb = torch.tensor([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.0, 1.1]])
    assert mean_average_precision(emb, [0, 0, 1, 1]) == 1.0


def test_exact_match():
    assert exact_match(["hello", "world"], ["hello", "there"]) == 0.5


def test_bleu_and_rouge_nonnegative():
    hyps = ["the cat sat on the mat"]
    refs = ["the cat is on the mat"]
    assert 0 <= bleu_score(hyps, refs) <= 1
    assert 0 <= rouge_l(hyps, refs) <= 1


def test_mrr_and_recall_at_k_perfect_for_tight_clusters():
    # Two tight 2D clusters, so every query has its top-1 as a same-label neighbor.
    emb = torch.tensor([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.0, 1.1]])
    normed = torch.nn.functional.normalize(emb, dim=-1)
    sim = normed @ normed.t()
    sim.fill_diagonal_(-float("inf"))
    labels = [0, 0, 1, 1]
    assert mean_reciprocal_rank(sim, labels) == 1.0
    assert recall_at_k(sim, labels, k=1) == 1.0
    assert recall_at_k(sim, labels, k=2) == 1.0


def test_mrr_with_no_matches_returns_zero():
    emb = torch.eye(3)
    normed = torch.nn.functional.normalize(emb, dim=-1)
    sim = normed @ normed.t()
    sim.fill_diagonal_(-float("inf"))
    # All labels distinct -> no positives.
    assert mean_reciprocal_rank(sim, [0, 1, 2]) == 0.0


def test_subtoken_f1():
    assert subtoken_f1([["a", "b"]], [["a", "b"]]) == 1.0
    assert subtoken_f1([["a"]], [["b"]]) == 0.0
    # Partial overlap: hyp={a,b}, ref={b,c} -> P=0.5, R=0.5, F1=0.5
    assert abs(subtoken_f1([["a", "b"]], [["b", "c"]]) - 0.5) < 1e-6
