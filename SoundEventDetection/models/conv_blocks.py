import torch
import torch.nn as nn

from typing import Optional


def get_conv_block(key: str):
    if key == 'conv':
        return ConvBlock
    elif key == 'res':
        return ResConvBlock
    elif key == 'glu':
        return GLUConvBlock
    else:
        raise ValueError(f'Unknown conv block type: {key}')


class GLUConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            pooling_size: Optional[int] = 2):
        super().__init__()
        padding_size = kernel_size // 2
        self.conv1 = nn.Conv2d(
            in_channels, out_channels * 2, kernel_size,
            padding=padding_size, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels * 2)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels * 2, kernel_size,
            padding=padding_size, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * 2)

        self.pooling = None
        if pooling_size is not None:
            self.pooling = nn.MaxPool2d((1, pooling_size))
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        out = x[:, :self.out_channels, :, :]
        mask = x[:, self.out_channels:, :, :]
        out = out * torch.sigmoid(mask)

        x = self.conv2(out)
        x = self.bn2(x)
        out = x[:, :self.out_channels, :, :]
        mask = x[:, self.out_channels:, :, :]
        out = out * torch.sigmoid(mask)

        if self.pooling is not None:
            out = self.pooling(out)

        return out


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

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, padding=kernel_size // 2)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

        self.pooling = nn.MaxPool2d((1, pooling_size))

    def forward(self, x: torch.Tensor):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act2(out)
        out = self.pooling(out)
        return out
