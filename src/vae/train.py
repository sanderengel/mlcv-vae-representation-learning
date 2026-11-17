import json
import torch
from collections import defaultdict
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from pathlib import Path

from vae.utils import get_device
from vae.models.base import BaseAutoencoder


def run_epoch(
    model: BaseAutoencoder,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optimizer | None = None,
) -> dict[str, float]:
    """Train or evaluate the model for a single epoch."""
    training = optimizer is not None
    model.train(training)

    totals = {}
    n_samples = 0

    with torch.set_grad_enabled(training):
        for x, _ in loader:
            x = x.to(device)
            B = x.size(0)

            out = model(x)
            losses = model.loss(x, out)

            if training:
                optimizer.zero_grad()
                losses["loss"].backward()
                optimizer.step()

            for component, loss in losses.items():
                totals[component] = totals.get(component, 0.0) + loss.item() * B

            n_samples += B

    # Average loss over all samples in the batch
    averages = {component: total / n_samples for component, total in totals.items()}
    return averages


def fit(
    model: BaseAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    run_dir: Path,
    lr: float = 1e-3,
    n_epochs: int = 10,
    verbose: bool = True
) -> tuple[dict, int]:
    if verbose:
        print(f"Training model {type(model).__name__} with lr={lr} over {n_epochs} epochs...")

    run_dir.mkdir(parents=True, exist_ok=False) # Raises error if already ran
    if verbose:
        print(f"Created directory '{run_dir}'")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    device = get_device(verbose=verbose)
    model.to(device)

    modes = ("train", "val")
    history = {mode: defaultdict(list) for mode in modes}
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, n_epochs + 1):
        train_losses = run_epoch(model, train_loader, device, optimizer)
        val_losses = run_epoch(model, val_loader, device) # No optimizer for evaluation

        for mode, losses in zip(modes, (train_losses, val_losses)):
            for component, loss in losses.items():
                history[mode][component].append(loss)

        if verbose:
            progress_str = f"Epoch: {epoch}/{n_epochs}, train loss: {train_losses['loss']}, val loss: {val_losses['loss']}"
            if "kl" in train_losses and "kl" in val_losses:
                progress_str += f", train kl: {train_losses['kl']}, val kl: {val_losses['kl']}"
            print(progress_str)

        if val_losses["loss"] < best_val_loss:
            best_val_loss = val_losses["loss"]
            best_epoch = epoch

            # Checkpoint
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_loss": best_val_loss},
                run_dir / "model.pt"
            )

    if verbose:
        print("Finished training.")

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history, best_epoch
