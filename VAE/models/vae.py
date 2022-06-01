import torch
import torch.nn as nn

from torch import Tensor

from .decoder import BaseDecoder
from .encoder import BaseEncoder


class VAE(nn.Module):
    """Universal framework for Variational AutoEncoder

    Args:
        encoder (BaseEncoder): Encoder block.
        decoder (BaseDecoder): Decoder block.
    """
    def __init__(
            self, encoder: BaseEncoder, decoder: BaseDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def sample(self, mean: Tensor, log_std: Tensor) -> Tensor:
        """Samples a latent space vector according to given mean and log_std.

        Args:
            mean (`Tensor`): Mean of the latent space distribution.
            log_std (`Tensor`): Log variance of the distribution.

        Returns:
            `Tensor`: Latent space vector
        """
        z = mean + torch.randn_like(log_std) * torch.exp(log_std)
        return z

    def encode(self, x: Tensor) -> Tensor:
        """Encodes a sample into a latent space vector.

        Args:
            x (`Tensor`): Sample.

        Returns:
            `Tensor`: Latent space vector.
        """
        mu, sigma = self.encoder(x)
        return self.sample(mu, sigma)

    def decode(self, z: Tensor) -> Tensor:
        """Decodes a latent space vector into a sample.

        Args:
            z (`Tensor`): Latent space vector.

        Returns:
            `Tensor`: Decoded sample.
        """
        return self.decoder(z)

    def forward(self, x: Tensor):
        mean, log_std = self.encoder(x)
        z = self.sample(mean, log_std)
        return self.decode(z), mean, log_std
