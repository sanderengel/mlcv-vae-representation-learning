from vae.data import get_dataloaders


def test_fashion_mnist_loaders():
    train, val, test, info = get_dataloaders("fashion_mnist", batch_size=16)
    x, y = next(iter(train))
    assert x.shape == (16, 1, 28, 28)
    assert x.min() >= 0.0 and x.max() <= 1.0
    assert len(val.dataset) == 5000
    assert info == {"in_channels": 1, "image_size": 28}
