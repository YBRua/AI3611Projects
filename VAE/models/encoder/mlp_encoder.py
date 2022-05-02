import torch
import torch.nn as nn

from torch import Tensor
from typing import Tuple, Union

from .base_encoder import BaseEncoder


class MLPEncoder(BaseEncoder):
    """2-layer Multi-Level Perceptron Encoder.
    """
    def __init__(self, input_dim: int, hidden_dim: int, z_dim: int,):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_log_sigma = nn.Linear(hidden_dim, z_dim)

    def forward(self, x: Tensor) -> Union[Tuple[Tensor, Tensor], Tensor]:
        """Forward pass of the model nya.

        Args:
            x (`Tensor`): Input MNIST samples of shape (B, H, W)
                Will be reshaped to (B, H*W) internally during forward pass

        Returns:
            mu, log_sigma: torch `Tensor`s for modeling hidden distribution.
        """
        x = x.reshape(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        mu = self.fc_mu(x)
        log_sigma = self.fc_log_sigma(x)
        return mu, log_sigma


class MLP3Encoder(BaseEncoder):
    def __init__(self, input_dim: int, h1_dim: int, h2_dim: int, z_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.fc_mu = nn.Linear(h2_dim, z_dim)
        self.fc_sigma = nn.Linear(h2_dim, z_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape(x.shape[0], -1)
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.tanh(self.fc2(x)))
        mu = self.fc_mu(x)
        log_sigma = self.fc_sigma(x)
        return mu, log_sigma
