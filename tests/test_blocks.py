import pytest
import torch

from vae.models.blocks import ConvEncoder, ConvDecoder, ResidualBlock


def test_residual_block_preserves_shape():
    x = torch.rand(4, 64, 7, 7)
    res = ResidualBlock(64)
    assert res(x).shape == x.shape


@pytest.mark.parametrize("num_res_blocks", [0, 1, 2])
@pytest.mark.parametrize("hidden_channels", [(32, 64), (64, 128)])
def test_encoder_decoder_shapes(hidden_channels, num_res_blocks):
    x = torch.rand(4, 1, 28, 28)
    enc = ConvEncoder(hidden_channels=hidden_channels, num_res_blocks=num_res_blocks)
    dec = ConvDecoder(hidden_channels=hidden_channels, num_res_blocks=num_res_blocks)

    h = enc(x)
    assert h.shape == (4, hidden_channels[-1], 7, 7)
    assert dec(h).shape == x.shape
