import torch
import torch.nn as nn

from .common import linear_softmax_pooling, weighted_sum_pooling
from .conv_blocks import get_conv_block
from .feed_forward import LinearFeedFwd, LocalizationFeedFwd

from typing import List, Optional


class Crnn(nn.Module):
    def __init__(
            self,
            num_freq: int,
            class_num: int,
            conv_block: str = 'conv',
            n_channels: List[int] = [16, 32, 64],
            pooling_sizes: List[int] = [4, 4, 2],
            gru_layers: int = 2,
            dropout: float = 0.0,
            gru_hidden: Optional[int] = None):
        ##############################
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

        conv_block = get_conv_block(conv_block)
        self.conv1 = conv_block(
            1, n_channels[0], pooling_size=pooling_sizes[0])
        self.conv2 = conv_block(
            n_channels[0], n_channels[1], pooling_size=pooling_sizes[1])
        self.conv3 = conv_block(
            n_channels[1], n_channels[2], pooling_size=pooling_sizes[2])

        h_factor = num_freq
        for psz in pooling_sizes:
            h_factor //= psz

        hid_size = n_channels[-1] * h_factor

        if gru_hidden is None:
            gru_hidden = hid_size

        self.gru = nn.GRU(
            hid_size, gru_hidden, gru_layers, dropout=dropout,
            batch_first=True, bidirectional=True)

        self.ffwd = LinearFeedFwd(gru_hidden * 2, class_num)

    def detection(self, x):
        ##############################
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
        x = self.ffwd(x)  # B, T, class_num
        return torch.sigmoid(x)

    def forward(self, x):
        frame_wise_prob = self.detection(x)  # B, T, ncls
        clip_prob = linear_softmax_pooling(frame_wise_prob)  # B, ncls
        '''(samples_num, feature_maps)'''
        return {
            'clip_probs': clip_prob,
            'time_probs': frame_wise_prob
        }


class GatedCrnn(nn.Module):
    def __init__(
            self,
            num_freq: int,
            class_num: int,
            conv_block: str = 'conv',
            n_channels: List[int] = [64, 64, 64],
            pooling_sizes: List[int] = [4, 2, 2],
            gru_layers: int = 1,
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

        conv_block = get_conv_block(conv_block)
        self.conv1 = conv_block(
            1, n_channels[0], pooling_size=pooling_sizes[0])
        self.conv2 = conv_block(
            n_channels[0], n_channels[1], pooling_size=pooling_sizes[1])
        self.conv3 = conv_block(
            n_channels[1], n_channels[2], pooling_size=pooling_sizes[2])

        h_factor = num_freq
        for psz in pooling_sizes:
            h_factor //= psz

        hid_size = n_channels[-1] * h_factor

        if gru_hidden is None:
            gru_hidden = hid_size

        self.gru = nn.GRU(
            hid_size, gru_hidden, gru_layers, dropout=dropout,
            batch_first=True, bidirectional=True)

        self.ffwd = LocalizationFeedFwd(gru_hidden * 2, class_num)

    def detection(self, x: torch.Tensor):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_wise_prob: [batch_size, time_steps, class_num]
        ##############################

        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 1, 3)  # B, T, C, F
        x = x.flatten(start_dim=2)  # B, T, C * F
        x = self.gru(x)[0]  # B, T, C * F
        x, loc_mask = self.ffwd(x)  # B, T, class_num
        return x, loc_mask

    def forward(self, x):
        frame_wise_prob, loc_mask = self.detection(x)  # B, T, ncls
        clip_prob = weighted_sum_pooling(frame_wise_prob, loc_mask)  # B, ncls
        return {
            'clip_probs': clip_prob,
            'time_probs': frame_wise_prob
        }
