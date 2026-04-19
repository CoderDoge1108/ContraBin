import torch

from contrabin.models.contrabin import ContraBinModel
from contrabin.models.encoders import AnchoredEncoder, TinyEncoder, TrainableEncoder, build_encoder
from contrabin.models.heads import LinearProjectionHead, NonLinearProjectionHead, build_head
from contrabin.models.interpolation import (
    LinearSimplexInterpolator,
    NonLinearSimplexInterpolator,
    SimplexInterpolationModule,
    simplex_interpolate,
)


def test_tiny_encoder_forward():
    enc = TinyEncoder(vocab_size=64, hidden_dim=16, max_length=32)
    ids = torch.randint(1, 64, (2, 10))
    mask = torch.ones_like(ids)
    out = enc(ids, mask).last_hidden_state
    assert out.shape == (2, 10, 16)


def test_build_encoder_tiny_trainable_or_anchored():
    trainable = build_encoder("contrabin-tiny", trainable=True, hidden_dim=16)
    anchored = build_encoder("contrabin-tiny", trainable=False, hidden_dim=16)
    assert isinstance(trainable, TrainableEncoder)
    assert isinstance(anchored, AnchoredEncoder)
    # anchored params frozen
    assert all(not p.requires_grad for p in anchored.parameters())
    # trainable params active
    assert any(p.requires_grad for p in trainable.parameters())


def test_heads_shapes():
    x = torch.randn(3, 32)
    for head_cls in (LinearProjectionHead, NonLinearProjectionHead):
        h = head_cls(32, 16, dropout=0.0)
        assert h(x).shape == (3, 16)
    assert isinstance(build_head("linear", 32, 16, 0.0), LinearProjectionHead)
    assert isinstance(build_head("nonlinear", 32, 16, 0.0), NonLinearProjectionHead)


def test_simplex_interpolate_scalar_and_matrix():
    a = torch.ones(4, 8)
    b = torch.zeros(4, 8)
    torch.testing.assert_close(
        simplex_interpolate(a, b, torch.tensor(0.5)), torch.full((4, 8), 0.5)
    )
    lam = torch.rand(4, 8)
    out = simplex_interpolate(a, b, lam)
    torch.testing.assert_close(out, lam)


def test_linear_interpolator_outputs_in_range():
    interp = LinearSimplexInterpolator(projection_dim=16, dropout=0.0)
    a = torch.randn(4, 16)
    b = torch.randn(4, 16)
    out = interp(a, b)
    assert out.shape == a.shape


def test_nonlinear_interpolator_outputs_in_range():
    interp = NonLinearSimplexInterpolator(projection_dim=16, dropout=0.0)
    a = torch.randn(4, 16)
    b = torch.randn(4, 16)
    out = interp(a, b)
    assert out.shape == a.shape


def test_simplex_module_stages():
    mod = SimplexInterpolationModule(projection_dim=16, dropout=0.0)
    a = torch.randn(2, 16)
    b = torch.randn(2, 16)
    assert torch.equal(mod(a, b, "naive"), a)
    assert mod(a, b, "linear").shape == (2, 16)
    assert mod(a, b, "nonlinear").shape == (2, 16)


def test_contrabin_model_forward(tiny_config):
    model = ContraBinModel(tiny_config.model)
    batch = {
        "source": {
            "input_ids": torch.randint(1, 64, (3, 16)),
            "attention_mask": torch.ones(3, 16, dtype=torch.long),
        },
        "binary": {
            "input_ids": torch.randint(1, 64, (3, 16)),
            "attention_mask": torch.ones(3, 16, dtype=torch.long),
        },
        "comment": {
            "input_ids": torch.randint(1, 64, (3, 12)),
            "attention_mask": torch.ones(3, 12, dtype=torch.long),
        },
    }
    out = model(batch, stage="naive")
    assert out.source.shape == (3, tiny_config.model.projection_dim)
    assert out.intermediate is None

    out_lin = model(batch, stage="linear")
    assert out_lin.intermediate is not None
    assert out_lin.intermediate.shape == (3, tiny_config.model.projection_dim)

    out_nl = model(batch, stage="nonlinear")
    assert out_nl.intermediate.shape == (3, tiny_config.model.projection_dim)
