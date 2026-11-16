import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Sequence

from vae.models.base import BaseAutoencoder


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 64):
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantise z_e (B, D, H, W); return z_q (B, D, H, W), indices (B, H, W), codebook_loss, commitment_loss."""
        B, D, H, W = z_e.shape

        # (B, D, H, W) -> (B, H, W, D) -> (B*H*W, D), i.e., one row per spatial position
        z_flat = z_e.permute(0, 2, 3, 1).reshape(-1, D)

        # Squared distances to all codebook vectors
        e = self.codebook.weight                  # (K, D)
        z_sq = z_flat.pow(2).sum(1, keepdim=True) # ||z||^2, (B*H*W, 1)
        e_sq = e.pow(2).sum(1)                    # ||e||^2, (K,)
        dist = z_sq - 2 * z_flat @ e.T + e_sq

        # Nearest code per position
        indices = dist.argmin(1) # (B*H*W,)
        z_q = self.codebook(indices).view(B, H, W, D).permute(0, 3, 1, 2) # Back to (B, D, H, W)

        # VQ losses (sum over elements, mean over batches)
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="sum") / B
        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="sum") / B

        # Straight-through estimator (forward uses z_q, backward copies gradients to z_e)
        z_q = z_e + (z_q - z_e).detach()

        return z_q, indices.view(B, H, W), codebook_loss, commitment_loss


class VQVAE(BaseAutoencoder):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: Sequence[int] = (32, 64),
        image_size: int = 28,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
    ):
        super().__init__(in_channels=in_channels, hidden_channels=hidden_channels, image_size=image_size)
        self.commitment_cost = commitment_cost
        feat_channels = self.feat_shape[0]

        self.pre_quant = nn.Conv2d(feat_channels, embedding_dim, kernel_size=1)
        self.quantizer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.post_quant = nn.Conv2d(embedding_dim, feat_channels, kernel_size=1)

