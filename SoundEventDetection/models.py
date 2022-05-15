import math

import torch
import torch.nn as nn


def linear_softmax_pooling(x):
    return (x ** 2).sum(1) / x.sum(1)


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            pooling_size: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
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
        x = self.pooling(x)
        return x


class Crnn(nn.Module):
    def __init__(
            self,
            num_freq: int,
            class_num: int):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     num_freq: int, mel frequency bins
        #     class_num: int, the number of output classes
        ##############################
        self.bn = nn.BatchNorm2d(num_freq)
        self.conv1 = ConvBlock(1, 16)
        self.conv2 = ConvBlock(16, 32)
        self.conv3 = ConvBlock(32, 64)

        self.gru = nn.GRU(64, 64, 2, batch_first=True, bidirectional=True)

        self.linear = nn.Linear(64 * 2, class_num)

    def detection(self, x):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_wise_prob: [batch_size, time_steps, class_num]
        ##############################
        x = self.bn(x)
        x = self.conv1(x.unqueeze(1))
        x = self.conv2(x)
        x = self.conv3(x)  # B, C, T, F
        x = x.permute(0, 2, 1, 3)  # B, T, C, F
        x = x.flatten(start_dim=2)  # B, T, C * F
        x = self.gru(x)[0]  # B, T, C * F
        x = self.linear(x)  # B, T, class_num
        return x
        
    def forward(self, x): 
        frame_wise_prob = self.detection(x)
        clip_prob = linear_softmax_pooling(frame_wise_prob)
        '''(samples_num, feature_maps)'''
        return {
            'clip_probs': clip_prob,
            'time_probs': frame_wise_prob
        }
