import torch
import torch.nn as nn

from torch import Tensor
from typing import Tuple

from .base_encoder import BaseEncoder


class ConvEncoder(BaseEncoder):
    def __init__(self, z_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=2)

        self.fc = nn.Linear(32 * 6 * 6, 16)

        self.mu = nn.Linear(16, z_dim)
        self.sigma = nn.Linear(16, z_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # B, H, W (28, 28)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.reshape(x.shape[0], -1)
        x = torch.relu(self.fc(x))
        mu = self.mu(x)
        log_sigma = self.sigma(x)

        return mu, log_sigma
