import math
import torch
import torch.nn as nn

from .conv_blocks import ConvBlock
from .common import linear_softmax_pooling

from typing import List


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
            B, self.nheads, N, self.d_k).transpose(1, 2)  # B, L, nheads, d_k
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
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim))
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
