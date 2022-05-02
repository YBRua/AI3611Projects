import torch.nn as nn
import torch.nn.functional as F


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
            tie_weights: bool = False,
            bidirectional: bool = False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        nhid_ = nhid // 2 if bidirectional else nhid

        if rnn_type in ['LSTM', 'GRU', 'BiLSTM', 'BiGRU']:
            rnn_type_ = rnn_type.replace('Bi', '')
            self.rnn = getattr(nn, rnn_type_)(
                ninp, nhid_, nlayers,
                dropout=dropout,
                bidirectional=bidirectional)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("Invalid option for `--model`")
            self.rnn = nn.RNN(
                ninp, nhid_, nlayers,
                nonlinearity=nonlinearity, dropout=dropout,
                bidirectional=bidirectional)
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
        self.bidirectional = bidirectional

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))  # L, B, ninp

        # output: L, B, nhid
        # hidden: nlayer * bi, B, nhid
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)  # L, B, ntoken
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        nlayers = self.nlayers * 2 if self.bidirectional else self.nlayers
        nhid = self.nhid // 2 if self.bidirectional else self.nhid
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(nlayers, bsz, nhid),
                    weight.new_zeros(nlayers, bsz, nhid))
        else:
            return weight.new_zeros(nlayers, bsz, nhid)
