import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(
            self,
            rnn_type: str,
            ntoken: int,
            ninp: int,
            nhid: int,
            nlayers: int,
            dropout: float = 0.5,
            tie_weights: bool = False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(
                ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError(
                    "An invalid option for `--model` was supplied,"
                    "options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']")
            self.rnn = nn.RNN(
                ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


class RNNAttention(RNNModel):
    def __init__(
            self,
            rnn_type: str,
            ntoken: int,
            ninp: int,
            nhid: int,
            nlayers: int,
            dropout: float = 0.5,
            tie_weights: bool = False):
        super().__init__(
            rnn_type, ntoken, ninp, nhid, nlayers, dropout, tie_weights)

        self.attn = nn.Linear(nhid * 2, nhid)
        nn.init.uniform_(self.attn.weight, -0.1, 0.1)

    def forward(self, x: Tensor):
        emb = self.drop(self.encoder(x))  # L, B, ninp
        output, hidden = self.rnn(emb)  # L, B, nhid

        output = output.transpose(0, 1).contiguous()  # B, L, nhid
        h_ = hidden.transpose(0, 1).contiguous()  # B, L, nhid

        attn_ws = self.attn(torch.cat((output, h_), dim=2))  # B, L, nhid
        attn_ws = torch.sum(attn_ws, dim=2, keepdim=True)  # B, L, 1
        attn_ws = F.softmax(attn_ws, dim=1)  # B, L, 1
        attn_ws = torch.transpose(attn_ws, 1, 2)  # B, 1, L
        attn_ctx = torch.bmm(attn_ws, hidden.transpose(0, 1))  # B, 1, nhid
        attn_ctx = torch.squeeze(attn_ctx, 1)  # B, nhid

        attn_ctx = self.drop(attn_ctx)
        decoded = self.decoder(attn_ctx)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden
