import torch.nn as nn

from .rnn import RNNModel
from .transformer import S2STransformer, TransformerModel


def get_model(
        architecture: str,
        input_size: int,
        args
) -> nn.Module:
    """Get a model of architecture `type` with specified `args`

    Args:
        - `type`: A string indicating the architecture of the model
        - `input_size`: Number of input tokens, used for the embedding layers
        - `args`: Other hyperparameters
    """
    if architecture == 'Transformer':
        model = TransformerModel(
            ntoken=input_size,
            ninp=args.emsize,
            nhead=args.nhead,
            nhid=args.nhid,
            nlayers=args.nlayers,
            dropout=args.dropout)
    elif architecture == 'S2STransformer':
        model = S2STransformer(
            ntokens=input_size,
            ninput=args.emsize,
            nheads=args.nhead,
            nhidden=args.nhid,
            encoder_layers=args.encoder_layers,
            decoder_layers=args.decoder_layers,
            dropout=args.dropout)
    elif architecture in ['RNN_RELU', 'RNN_TANH', 'LSTM', 'GRU']:
        model = RNNModel(
            rnn_type=args.model,
            ntoken=input_size,
            ninp=args.emsize,
            nhid=args.nhid,
            nlayers=args.nlayers,
            dropout=args.dropout,
            tie_weights=args.tied)
    elif architecture in ['BiLSTM', 'BiGRU']:
        model = RNNModel(
            rnn_type=args.model,
            ntoken=input_size,
            ninp=args.emsize,
            nhid=args.nhid,
            nlayers=args.nlayers,
            dropout=args.dropout,
            tie_weights=args.tied,
            bidirectional=True)
    else:
        raise ValueError(f'Architecture {architecture} does not exist')

    return model
