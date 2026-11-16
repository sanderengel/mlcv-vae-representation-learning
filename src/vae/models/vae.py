import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Sequence

from vae.models.base import BaseAutoencoder


class VAE(BaseAutoencoder):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: Sequence[int] = (32, 64),
        image_size: int = 28,
        latent_dim: int = 16,
        beta: float = 1.0
    ):
        super().__init__(in_channels=in_channels, hidden_channels=hidden_channels, image_size=image_size)
        self.beta = beta
        self.fc_mu = nn.Linear(self.feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.feat_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, *self.feat_shape)
        x_logits = self.decoder(h)
        return x_logits

    def forward(self, x: torch.Tensor) -> dict:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_logits = self.decode(z)
        return {"x_logits": x_logits, "z": z, "mu": mu, "logvar": logvar}

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def loss(self, x: torch.Tensor, out: dict) -> dict:
        B = x.size(0)
        recon = F.binary_cross_entropy_with_logits(out["x_logits"], x, reduction="sum") / B
        kl = -0.5 * torch.sum(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp()) / B
        loss = recon + self.beta * kl
        return {"loss": loss, "recon": recon, "kl": kl}
