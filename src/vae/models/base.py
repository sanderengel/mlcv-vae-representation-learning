import torch
import torch.nn as nn
from collections.abc import Sequence
from abc import ABC, abstractmethod

from vae.models.blocks import ConvEncoder, ConvDecoder


class BaseAutoencoder(nn.Module, ABC):
    def __init__(self, in_channels: int = 1, hidden_channels: Sequence[int] = (32, 64), image_size: int = 28):
        super().__init__()
        self.encoder = ConvEncoder(in_channels=in_channels, hidden_channels=hidden_channels)
        self.decoder = ConvDecoder(out_channels=in_channels, hidden_channels=hidden_channels)
        
        size = image_size // 2 ** len(hidden_channels)   # each layer halves H and W
        self.feat_shape = (self.encoder.out_channels, size, size)   # (64, 7, 7)

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Encode images (B, C, H, W) into the model-specific latent representation."""

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents into image logits (B, C, H, W)."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> dict:
        """Run encode-decode; return a dict with at least 'x_logits' plus extras needed by 'loss'."""

    @abstractmethod
    def loss(self, x: torch.Tensor, out: dict) -> dict:
        """Compute losses from 'forward' output; return a dict with 'loss' plus components for logging."""
