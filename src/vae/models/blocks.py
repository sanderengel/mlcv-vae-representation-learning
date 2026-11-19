import torch
import torch.nn as nn
from collections.abc import Sequence


class ResidualBlock(nn.Module):
    """Conv-BN-ReLU-Conv-BN with a skip connection. Preserves shape."""
    def __init__(self, channels: int):
        super().__init__()
        # Two convolutions with ReLU between them so the residual branch is nonlinear
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class ConvEncoder(nn.Module):
    """Map images to a spatial feature map. Defaults: (B, 1, 28, 28) -> (B, 64, 7, 7)."""
    def __init__(self, in_channels: int = 1, hidden_channels: Sequence[int] = (32, 64), num_res_blocks: int = 0):
        super().__init__()

        chs = tuple(hidden_channels)
        layers = []
        prev = in_channels
        for ch in chs:
            layers += [
                nn.Conv2d(prev, ch, kernel_size=4, stride=2, padding=1), # Halves H and W
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True)
            ]
            layers += [ResidualBlock(ch) for _ in range(num_res_blocks)]
            prev = ch

        self.net = nn.Sequential(*layers)
        self.out_channels = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        return h


class ConvDecoder(nn.Module):
    """Map spatial feature map to images. Defaults: (B, 64, 7, 7) -> (B, 1, 28, 28)."""
    def __init__(self, out_channels: int = 1, hidden_channels: Sequence[int] = (32, 64), num_res_blocks: int = 0):
        super().__init__()

        chs = tuple(hidden_channels)[::-1] + (out_channels,)
        layers = []
        for i, (c_in, c_out) in enumerate(zip(chs[:-1], chs[1:])):
            layers += [ResidualBlock(c_in) for _ in range(num_res_blocks)]
            layers.append(nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1))  # Doubles H and W
            if i != len(chs) - 2: # No BatchNorm or ReLU on the output layer
                layers += [nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]

        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x_logits = self.net(h)
        return x_logits
