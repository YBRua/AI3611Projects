import torch
import torch.nn as nn

from torch import Tensor

from .base_decoder import BaseDecoder


class MLPDecoder(BaseDecoder):
    """2-layer MLP Decoder.
    """
    def __init__(self, z_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: Tensor) -> Tensor:
        z = torch.relu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z))
        z = z.reshape(-1, 28, 28)
        return z


class MLP3Decoder(BaseDecoder):
    def __init__(self, z_dim: int, h1_dim: int, h2_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h2_dim, output_dim)

    def forward(self, z: Tensor) -> Tensor:
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        z = torch.sigmoid(self.fc3(z))
        z = z.reshape(-1, 28, 28)
        return z
