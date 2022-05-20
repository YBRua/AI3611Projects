import math
import torch
import torch.nn as nn

from .conv_blocks import get_conv_block

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


class ConvTransformer(nn.Module):
    def __init__(
            self,
            num_freq: int,
            class_num: int,
            conv_block: str = 'conv',
            n_channels: List[int] = [16, 32, 64],
            pooling_sizes: List[int] = [4, 4, 2],
            dropout: float = 0.2):
        super().__init__()

        if len(pooling_sizes) != 3:
            raise ValueError('pooling_sizes should be a list of 3 ints')

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

        self.pos_encoding = PositionalEncoding(hid_size, dropout)
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, hid_size), requires_grad=True)
        self.transformer = TransformerBlock(
            hid_size, hid_size, nhead=8, dropout=dropout)

        self.linear = nn.Linear(hid_size, class_num)

    def detection(self, x: torch.Tensor):
        # x: [batch_size, time_steps, num_freq]
        # frame_wise_prob: [batch_size, time_steps, class_num]

        x = x.unsqueeze(1)

        x = self.conv1(x)  # B, C, T, H
        x = self.conv2(x)
        x = self.conv3(x)  # B, C, T, 1

        x = x.permute(0, 2, 1, 3)  # B, T, C, 1
        x = x.flatten(start_dim=2)  # B, T, C
        cls_token = torch.repeat_interleave(self.cls_token, x.shape[0], 0)
        x = torch.cat((cls_token, x), dim=1)

        x = self.pos_encoding(x)
        x = self.transformer(x)  # B, T+1, C * F
        x = self.linear(x)  # B, T+1, class_num
        return torch.sigmoid(x)

    def forward(self, x):
        outputs = self.detection(x)  # B, T+1, class_num
        frame_wise_prob = outputs[:, 1:, :]
        clip_prob = outputs[:, 0, :]  # [CLS] token as clip prediction
        '''(samples_num, feature_maps)'''
        return {
            'clip_probs': clip_prob,
            'time_probs': frame_wise_prob
        }
