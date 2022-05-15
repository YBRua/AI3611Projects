import torch
import torch.nn as nn

from typing import Optional


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            pooling_size: Optional[int] = 2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

        self.pooling = None
        if pooling_size is not None:
            self.pooling = nn.MaxPool2d((1, pooling_size))

    def forward(self, x: torch.Tensor):
        """Forward function of the convolution block

        Args:
            x (Tensor): Input Tensor of shape [B, C_in, T, F_in]

        Returns:
            Tensor: Output Tensor of shape [B, C_out, T, F_out]
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)

        if self.pooling is not None:
            x = self.pooling(x)
        return x


class ResConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            pooling_size: int = 2):
        super().__init__()

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False)

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2)

        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.pooling = nn.MaxPool2d((1, pooling_size))

    def forward(self, x: torch.Tensor):
        identity = x
        out = self.conv(x)
        out = self.bn(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)
        out = self.pooling(out)
        return out
