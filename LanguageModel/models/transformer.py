import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Transformer

from .positional_encoding import PositionalEncoding


class S2STransformer(nn.Module):
    def __init__(
            self,
            ntokens: int,
            ninput: int,
            nheads: int,
            nhidden: int,
            encoder_layers: int,
            decoder_layers: int,
            dropout: float = 0.5) -> None:
        super().__init__()
        self.embedding = nn.Embedding(ntokens, ninput)
        self.pos_encoding = PositionalEncoding(ninput, dropout)
        self.transformer = Transformer(ninput, nheads, encoder_layers, decoder_layers, nhidden, dropout)
        self.linear = nn.Linear(nhidden, ntokens)

        self.ninput = ninput
        self.mask = None

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        nn.init.zeros_(self.linear.bias)
        nn.init.uniform_(self.linear.weight, -initrange, initrange)

    def _get_attention_mask(self, size: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        assert src.size(0) == tgt.size(0)
        if self.mask is None or self.mask.size(0) != tgt.size(0):
            device = tgt.device
            self.mask = self._get_attention_mask(tgt.size(0)).to(device)

        src = self.embedding(src) * math.sqrt(self.ninput)
        tgt = self.embedding(tgt) * math.sqrt(self.ninput)

        src = self.pos_encoding(src)
        tgt = self.pos_encoding(tgt)

        out = self.transformer(src, tgt, tgt_mask=self.mask, src_mask=self.mask, memory_mask=self.mask)
        out = self.linear(out)

        pred = F.log_softmax(out, dim=-1)
        return pred


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)
