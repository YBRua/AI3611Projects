import torch
import torch.nn as nn

from .common import linear_softmax_pooling
from .conv_blocks import ConvBlock

from typing import List, Optional


class Crnn(nn.Module):
    def __init__(
            self,
            num_freq: int,
            class_num: int,
            n_channels: List[int] = [16, 32, 64],
            pooling_sizes: List[int] = [4, 4, 2],
            gru_layers: int = 2,
            dropout: float = 0.0,
            gru_hidden: Optional[int] = None):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     num_freq: int, mel frequency bins
        #     class_num: int, the number of output classes
        ##############################
        super().__init__()

        if len(pooling_sizes) != 3:
            raise ValueError('pooling_sizes should be a list of 3 ints')
        if len(n_channels) != 3:
            raise ValueError('n_channels should be a list of 3 ints')

        self.bn = nn.BatchNorm2d(1)
        self.conv1 = ConvBlock(
            1, n_channels[0], pooling_size=pooling_sizes[0])
        self.conv2 = ConvBlock(
            n_channels[0], n_channels[1], pooling_size=pooling_sizes[1])
        self.conv3 = ConvBlock(
            n_channels[1], n_channels[2], pooling_size=pooling_sizes[2])
        # self.conv1 = ResConvBlock(1, 16, pooling_size=pooling_sizes[0])
        # self.conv2 = ResConvBlock(16, 32, pooling_size=pooling_sizes[1])
        # self.conv3 = ResConvBlock(32, 64, pooling_size=pooling_sizes[2])

        h_factor = num_freq
        for psz in pooling_sizes:
            h_factor //= psz

        hid_size = n_channels[-1] * h_factor

        if gru_hidden is None:
            gru_hidden = hid_size

        self.gru = nn.GRU(
            hid_size, gru_hidden, gru_layers, dropout=dropout,
            batch_first=True, bidirectional=True)

        self.linear = nn.Linear(gru_hidden * 2, class_num)

    def detection(self, x):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_wise_prob: [batch_size, time_steps, class_num]
        ##############################
        x = self.bn(x.unsqueeze(1))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # B, C, T, F
        x = x.permute(0, 2, 1, 3)  # B, T, C, F
        x = x.flatten(start_dim=2)  # B, T, C * F
        x = self.gru(x)[0]  # B, T, C * F
        x = self.linear(x)  # B, T, class_num
        return torch.sigmoid(x)

    def forward(self, x):
        frame_wise_prob = self.detection(x)  # B, T, ncls
        clip_prob = linear_softmax_pooling(frame_wise_prob)  # B, ncls
        '''(samples_num, feature_maps)'''
        return {
            'clip_probs': clip_prob,
            'time_probs': frame_wise_prob
        }
