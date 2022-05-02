import torch.nn as nn

from torch import Tensor
from typing import Tuple, Union


class BaseEncoder(nn.Module):
    """Base class for the encoder layer.
    This class should not be instantiated.
    """

    def __init__(self, is_vae: bool = True):
        super().__init__()
        self.is_vae = is_vae

    def forward(self, x: Tensor) -> Union[Tuple[Tensor, Tensor], Tensor]:
        pass
