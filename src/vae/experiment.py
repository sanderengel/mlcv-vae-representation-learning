import json
import torch
from pathlib import Path

from vae import config
from vae.utils import set_seed
from vae.data import get_dataloaders
from vae.models.ae import AE
from vae.models.vae import VAE
from vae.models.vqvae import VQVAE
from vae.train import fit

MODELS = {"ae": AE, "vae": VAE, "vqvae": VQVAE}


def run_experiment(
    model_name: str,
    run_name: str,
    dataset: str = "fashion_mnist",
    runs_dir: str | Path = "runs",
    seed: int = config.SEED,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LR,
    n_epochs: int = config.N_EPOCHS,
    overwrite: bool = False,
    verbose: bool = True,
    **model_kwargs
) -> tuple[dict, int]:
    """Train one model and save config, checkpoint, and history to runs_dir/run_name."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Supported models are {list(MODELS)}")

    run_dir = Path(runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=overwrite) # If not set to overwrite, raise error if ran before
    if verbose:
        print(f"Created directory '{run_dir}'")

    set_seed(seed)
    train_loader, val_loader, _, info = get_dataloaders(dataset, batch_size=batch_size, seed=seed)

    # Backbone defaults from config, dataset shape from info, explicit overrides win
    model_kwargs = {
        "hidden_channels": config.HIDDEN_CHANNELS,
        "num_res_blocks": config.NUM_RES_BLOCKS,
        **info,
        **model_kwargs
    }

    run_config = {
        "model": model_name,
        "dataset": dataset,
        "seed": seed,
        "batch_size": batch_size,
        "lr": lr,
        "n_epochs": n_epochs,
        "model_kwargs": model_kwargs
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    model = MODELS[model_name](**model_kwargs)
    return fit(model, train_loader, val_loader, run_dir=run_dir, lr=lr, n_epochs=n_epochs, verbose=verbose)


def load_run(run_dir: str | Path, device: str = "cpu"):
    """Rebuild a trained model and return it with its config and history."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    history = json.loads((run_dir / "history.json").read_text())

    model = MODELS[cfg["model"]](**cfg["model_kwargs"])
    ckpt = torch.load(run_dir / "model.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    return model, cfg, history
