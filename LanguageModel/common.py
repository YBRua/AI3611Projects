import torch
import logging

from datetime import datetime
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
    parser.add_argument(
        '--data', type=str, default='./data/wikitext-2',
        help='location of the data corpus')
    parser.add_argument(
        '--model', type=str, default='LSTM',
        help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer, S2STransformer, LSTMAttn)')
    parser.add_argument(
        '--emsize', type=int, default=200,
        help='size of word embeddings')
    parser.add_argument(
        '--nhid', type=int, default=200,
        help='number of hidden units per layer')
    parser.add_argument(
        '--nlayers', type=int, default=2,
        help='number of layers')
    parser.add_argument(
        '--encoder_layers', type=int, default=2,
        help='Number of encoder layers in transformer')
    parser.add_argument(
        '--decoder_layers', type=int, default=2,
        help='Number of decoder layers in transformer')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='',
                        help='path to save the final model')
    parser.add_argument('--onnx-export', type=str, default='',
                        help='path to export the final model in onnx format')
    parser.add_argument('--nhead', type=int, default=2,
                        help='the number of heads in the encoder/decoder of the transformer model')
    parser.add_argument('--dry-run', action='store_true',
                        help='verify the code and the model')
    parser.add_argument(
        '--no_tqdm', action='store_true',
        help='Whether to disable tqdm')

    return parser.parse_args()


def batchify(
    dataset: torch.Tensor,
    n_batches: int,
    device: torch.device
) -> torch.Tensor:
    """Splits `dataset` into batches of shape (N, B)

    Args:
        - `dataset`: PyTorch Tensor of shape `(N * B + r)`
        where `r < B` is a remainder that will be dropped
        - `n_batches`: Nubmer of batches (B)
        - `device`: Torch device (CPU or GPU) on which the returned dataset will be stored

    Returns:
        Tensor of shape (N, B). Note that torch RNNs are batch-last by default.
    """
    batch_size = dataset.shape[0] // n_batches
    dataset = dataset.narrow(dim=0, start=0, length=batch_size * n_batches)
    dataset = dataset.view(n_batches, -1).t().contiguous()
    return dataset.to(device)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.
def get_batch(
    dataset: torch.Tensor,  # N, B
    offset: int,
    seq_len: int
) -> torch.Tensor:
    """Produces a batch of data for autoregressive language modeling.
    Note that torch RNNs are not batch-first by default,
    so we slice along the first dimension (dim=0)

    Args:
        - `dataset`: Original dataset of shape (N, B)
        - `offset`: Offset index to start slicing from `dataset`
        - `seq_len`: Length of minibatch

    Returns:
        - `data`: Tensor of shape (seq_len, B)
        - `target`: Ground truth Tensor of shape (seq_len * B,)
    """
    seq_len = min(seq_len, len(dataset) - 1 - offset)
    data = dataset[offset:offset+seq_len]
    target = dataset[offset+1:offset+1+seq_len].view(-1)
    return data, target


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def setup_logger(args):
    logger = logging.getLogger('LM')
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter("%(message)s")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_handler = logging.FileHandler(
        filename=f'./logs/{timestamp}-{args.model}-{args.emsize}-{args.epochs}.log',
        mode='a',
        encoding='utf-8')
    file_formatter = logging.Formatter(
        "[%(levelname)s %(asctime)s]: %(message)s")

    stream_handler.setFormatter(stream_formatter)
    file_handler.setFormatter(file_formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger
