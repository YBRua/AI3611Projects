import math
import torch
import torch.nn as nn

from typing import List, Optional


def linear_softmax_pooling(x):
    return (x ** 2).sum(1) / x.sum(1)


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


class MHAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            nhead: int):
        super().__init__()
        if embed_dim % nhead != 0:
            raise ValueError(
                'embed_dim must be divisible by nhead '
                f'embed_dim={embed_dim}, nhead={nhead}')
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        self.nheads = nhead
        self.d_k = embed_dim // nhead

    def _forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor):
        B, N, E = query.shape

        # batch, length, nhead, d_k
        q = self.q_linear(query).reshape(B, N, self.nheads, self.d_k)
        k = self.k_linear(key).reshape(B, N, self.nheads, self.d_k)
        v = self.v_linear(value).reshape(B, N, self.nheads, self.d_k)

        # batch, nhead, length, d_k
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # batch * nheads, length, d_k
        q = q.reshape(B * self.nheads, N, self.d_k)
        k = k.reshape(B * self.nheads, N, self.d_k)
        v = v.reshape(B * self.nheads, N, self.d_k)

        attn = torch.bmm(q, k.transpose(-1, -2)) / \
            (self.d_k ** 0.5)  # B * nheads, length, length
        attn = torch.softmax(attn, dim=-1)

        out = torch.bmm(attn, v)  # B * nheads, length, d_k
        out = out.reshape(
            B, self.nheads, N, self.d_k).transpose(1, 2)  # B, length, nheads, d_k
        # B, length, nheads * d_k
        out = out.reshape(B, -1, self.nheads * self.d_k)
        return out

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor):
        return self._forward(query, key, value)


class FeedFwdLayer(nn.Module):
    def __init__(self, input_dims, hidden_dims, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, input_dims)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, nhead, dropout=0.1):
        super().__init__()
        self.feedforward = FeedFwdLayer(embed_dim, hidden_dim, dropout)
        self.attention = MHAttention(embed_dim, nhead)
        self.attn_dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = x + self.attn_dropout(self.attention(x, x, x))
        x = self.norm1(x)

        x = x + self.feedforward(x)
        x = self.norm2(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CMHA(nn.Module):
    def __init__(
            self,
            num_freq: int,
            class_num: int,
            pooling_sizes: List[int] = [4],
            dropout: float = 0.2):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     num_freq: int, mel frequency bins
        #     class_num: int, the number of output classes
        ##############################
        super().__init__()

        if len(pooling_sizes) != 1:
            raise ValueError('pooling_sizes should be a list of 1 ints')

        self.bn = nn.BatchNorm2d(1)
        self.conv1 = ConvBlock(1, 32, kernel_size=3, stride=1, padding=1)

        h_factor = num_freq
        for psz in pooling_sizes:
            h_factor //= psz

        hid_size = 16 * h_factor

        self.pos_encoding = PositionalEncoding(hid_size, dropout)
        self.transformer = TransformerBlock(hid_size, hid_size, 8, dropout)

        self.linear = nn.Linear(hid_size, class_num)

    def detection(self, x):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_wise_prob: [batch_size, time_steps, class_num]
        ##############################
        x = self.bn(x.unsqueeze(1))
        x = self.conv1(x)  # B, C, T, 1
        x = self.conv_bn(x)

        x = x.permute(0, 2, 1, 3)  # B, T, C, 1
        x = x.flatten(start_dim=2)  # B, T, C
        x = self.pos_encoding(x)
        x = self.transformer(x)  # B, T, C * F
        x = self.linear(x)  # B, T, class_num
        return torch.sigmoid(x)

    def forward(self, x):
        frame_wise_prob = self.detection(x)
        clip_prob = linear_softmax_pooling(frame_wise_prob)
        '''(samples_num, feature_maps)'''
        return {
            'clip_probs': clip_prob,
            'time_probs': frame_wise_prob
        }


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
