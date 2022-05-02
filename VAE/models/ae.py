import torch.nn as nn

from torch import Tensor

from .decoder import BaseDecoder
from .encoder import BaseEncoder


class Buggy(Exception):
    pass


class AE(nn.Module):
    def __init__(
            self, encoder: BaseEncoder, decoder: BaseDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def sample(self, mean: Tensor, log_std: Tensor) -> Tensor:
        raise Buggy('AutoEncoders cannot be sampleda.')

    def encode(self, x: Tensor) -> Tensor:
        """Encodes a sample into a latent space vector.

        Args:
            x (`Tensor`): Sample.

        Returns:
            `Tensor`: Latent space vector.
        """
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z: Tensor) -> Tensor:
        """Decodes a latent space vector into a sample.

        Args:
            z (`Tensor`): Latent space vector.

        Returns:
            `Tensor`: Decoded sample.
        """
        return self.decoder(z)

    def forward(self, x: Tensor):
        z, _ = self.encoder(x)
        return self.decode(z), z
