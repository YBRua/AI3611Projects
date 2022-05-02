import torch
import torch.nn as nn

from torch import Tensor

from .base_decoder import BaseDecoder


class ConvDecoder(BaseDecoder):
    def __init__(self, z_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, 32 * 6 * 6)

        self.conv1 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=2)

        self.conv2 = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=1,
            kernel_size=4,
            stride=2)

    def forward(self, z: Tensor) -> Tensor:
        z = torch.relu(self.fc1(z))
        z = z.reshape(z.shape[0], 32, 6, 6)

        z = torch.relu(self.conv1(z))
        z = torch.sigmoid(self.conv2(z))

        return z
