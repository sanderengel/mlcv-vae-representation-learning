import torch
import torch.nn.functional as F


def recon_loss(x: torch.Tensor, x_logits: torch.Tensor) -> torch.Tensor:
    """BCE reconstruction loss, summed over pixels and averaged over the batch."""
    return F.binary_cross_entropy_with_logits(x_logits, x, reduction="sum") / x.size(0)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence, summed over latent dimensions and averaged over the batch."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE summed over elements and averaged over the batch."""
    return F.mse_loss(pred, target, reduction="sum") / pred.size(0)
