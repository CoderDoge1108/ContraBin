import torch

from contrabin.losses.contrastive import PrimaryContrastiveLoss, clip_style_loss
from contrabin.losses.intermediate import IntermediateContrastiveLoss, info_nce_loss


def test_clip_style_loss_positive():
    a = torch.randn(8, 16)
    b = torch.randn(8, 16)
    loss = clip_style_loss(a, b, temperature=1.0)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_clip_style_loss_module():
    loss_fn = PrimaryContrastiveLoss(temperature=0.5)
    a = torch.randn(4, 8)
    b = torch.randn(4, 8)
    assert loss_fn(a, b).ndim == 0


def test_info_nce_is_small_when_aligned():
    torch.manual_seed(0)
    x = torch.randn(16, 32)
    aligned = x + 0.01 * torch.randn_like(x)
    mismatched = torch.randn(16, 32)
    aligned_loss = info_nce_loss(x, aligned, temperature=0.1).item()
    mismatched_loss = info_nce_loss(x, mismatched, temperature=0.1).item()
    assert aligned_loss < mismatched_loss


def test_intermediate_loss_module_backprop():
    loss_fn = IntermediateContrastiveLoss(temperature=0.1)
    a = torch.randn(4, 16, requires_grad=True)
    b = torch.randn(4, 16)
    loss = loss_fn(a, b)
    loss.backward()
    assert a.grad is not None
    assert a.grad.shape == a.shape
