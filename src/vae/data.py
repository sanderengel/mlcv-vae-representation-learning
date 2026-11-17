import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DATASETS = {
    "fashion_mnist": {"cls": datasets.FashionMNIST, "in_channels": 1, "image_size": 28},
    "cifar10": {"cls": datasets.CIFAR10, "in_channels": 3, "image_size": 32}
}


def get_dataloaders(
    name: str = "fashion_mnist",
    batch_size: int = 128,
    val_size: int = 5000,
    data_dir: str | Path = "data",
    num_workers: int = 0,
    seed: int = 0
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Return loaders for train, val, and test and dataset info (in_channels, image_size)."""
    if name not in DATASETS:
        raise ValueError(f"Dataset {name} is not supported. Supported datasets are {list(DATASETS)}")
    spec = DATASETS[name]
    dataset = spec["cls"]

    transform = transforms.ToTensor() # (C, H, W) float32 in [0, 1]
    train_full = dataset(data_dir, train=True, download=True, transform=transform)
    test = dataset(data_dir, train=False, download=True, transform=transform)

    # Seeded split for identical val sets across runs
    generator = torch.Generator().manual_seed(seed)
    train, val = random_split(train_full, [len(train_full) - val_size, val_size], generator=generator)

    loader_kwargs = {"batch_size": batch_size, "num_workers": num_workers}
    train_loader = DataLoader(train, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test, shuffle=False, **loader_kwargs)

    info = {"in_channels": spec["in_channels"], "image_size": spec["image_size"]}
    return train_loader, val_loader, test_loader, info
