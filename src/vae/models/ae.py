import torch
import torch.nn as nn

from vae.models.base import BaseAutoencoder
from vae.loss import recon_loss


class AE(BaseAutoencoder):
    def __init__(self, latent_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.fc_enc = nn.Linear(self.feat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.feat_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x).flatten(1)
        z = self.fc_enc(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, *self.feat_shape)
        x_logits = self.decoder(h)
        return x_logits

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_logits = self.decode(z)
        return x_logits

    def forward(self, x: torch.Tensor) -> dict:
        z = self.encode(x)
        x_logits = self.decode(z)
        return {"x_logits": x_logits, "z": z}

    def loss(self, x: torch.Tensor, out: dict) -> dict:
        recon = recon_loss(x, out["x_logits"])
        return {"loss": recon, "recon": recon}
