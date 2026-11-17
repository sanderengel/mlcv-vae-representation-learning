import torch
import torch.nn as nn
from collections.abc import Sequence

from vae.models.base import BaseAutoencoder
from vae.loss import recon_loss, mse_loss


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 64):
        super().__init__()
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
        indices = indices.view(B, H, W)

        # VQ losses (sum over elements, mean over batches)
        codebook_loss = mse_loss(z_q, z_e.detach())
        commitment_loss = mse_loss(z_e, z_q.detach())

        # Straight-through estimator (forward uses z_q, backward copies gradients to z_e)
        z_q = z_e + (z_q - z_e).detach()

        return z_q, indices, codebook_loss, commitment_loss


class VQVAE(BaseAutoencoder):
    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 64, commitment_cost: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.commitment_cost = commitment_cost
        feat_channels = self.feat_shape[0]

        self.pre_quant = nn.Conv2d(feat_channels, embedding_dim, kernel_size=1)
        self.quantizer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.post_quant = nn.Conv2d(embedding_dim, feat_channels, kernel_size=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        z_e = self.pre_quant(h)
        return z_e

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        h = self.post_quant(z_q)
        x_logits = self.decoder(h)
        return x_logits

    def forward(self, x: torch.Tensor) -> dict:
        z_e = self.encode(x)
        z_q, indices, codebook_loss, commitment_loss = self.quantizer(z_e)
        x_logits = self.decode(z_q)
        return {
            "x_logits": x_logits,
            "z_q": z_q,
            "indices": indices,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss
        }

    def loss(self, x: torch.Tensor, out: dict) -> dict:
        recon = recon_loss(x, out["x_logits"])
        loss = recon + out["codebook_loss"] + self.commitment_cost * out["commitment_loss"]
        return {
            "loss": loss,
            "recon": recon,
            "codebook": out["codebook_loss"],
            "commitment": out["commitment_loss"]
        }
