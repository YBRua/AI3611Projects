import torch.nn as nn

from torch import Tensor


class BaseDecoder(nn.Module):
    """Base class for the decoder layer.
    This class should not be instantiated.
    """

    def __init__(self):
        super().__init__()

    def forward(self, z: Tensor) -> Tensor:
        pass
