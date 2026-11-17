import pytest
import torch

from vae.models.ae import AE
from vae.models.vae import VAE
from vae.models.vqvae import VQVAE

MODELS = [AE, VAE, VQVAE]


@pytest.fixture
def x() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(8, 1, 28, 28)


@pytest.mark.parametrize("model_cls", MODELS)
def test_output_shape(model_cls, x):
    model = model_cls()
    out = model(x)
    assert out["x_logits"].shape == x.shape


@pytest.mark.parametrize("model_cls", MODELS)
def test_loss_is_finite_scalar(model_cls, x):
    model = model_cls()
    out = model(x)
    losses = model.loss(x, out)
    assert "loss" in losses
    for name, value in losses.items():
        assert value.dim() == 0, f"{name} is not a scalar"
        assert torch.isfinite(value), f"{name} is not finite"


@pytest.mark.parametrize("model_cls", MODELS)
def test_gradients_reach_encoder(model_cls, x):
    model = model_cls()
    out = model(x)
    model.loss(x, out)["loss"].backward()
    grad = model.encoder.net[0].weight.grad
    assert grad is not None and grad.abs().sum() > 0


@pytest.mark.parametrize("model_cls", MODELS)
def test_bigger_backbone(model_cls, x):
    model = model_cls(hidden_channels=(64, 128), num_res_blocks=1)
    out = model(x)
    assert out["x_logits"].shape == x.shape


def test_invalid_image_size_raises():
    with pytest.raises(ValueError):
        AE(image_size=30)
