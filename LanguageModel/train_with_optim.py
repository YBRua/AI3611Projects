import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import data
from common import get_batch, parse_args, batchify, repackage_hidden
from common import setup_logger
from models import get_model

EVAL_BATCH_SIZE = 10


def train(
        epoch: int,
        train_data: torch.Tensor,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        ntokens: int,
        args):
    model.train()
    total_loss = 0.

    if args.model != 'Transformer' and args.model != 'S2STransformer':
        hidden = model.init_hidden(args.batch_size)
    if args.no_tqdm:
        progress = range(0, train_data.size(0) - 1, args.bptt)
    else:
        progress = tqdm(range(0, train_data.size(0) - 1, args.bptt))

    if args.no_tqdm:
        logger.info(f'Epoch {epoch}')

    for bid, i in enumerate(progress):
        samples, targets = get_batch(train_data, i, args.bptt)
        optimizer.zero_grad()

        if args.model == 'Transformer':
            output = model(samples)
            output = output.view(-1, ntokens)
        elif args.model == 'S2STransformer':
            output = model(samples, samples)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(samples, hidden)

        loss = criterion(output, targets)
        loss.backward()

        # gradient clipping to avoid explosions
        # nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (bid + 1)
        ppl = math.exp(avg_loss)

        if not args.no_tqdm:
            progress.set_description(
                f'| Epoch {epoch:3d} | Loss {avg_loss:5.4f} | Ppl {ppl:8.2f}')

        if args.no_tqdm and bid % 1500 == 0:
            logger.info(
                f"| E {epoch:2d} | {bid:5d}/{len(progress)} |"
                f" Loss {avg_loss:8.4f} | Ppl {ppl:8.2f}")

    return avg_loss, ppl


def evaluate(
        eval_data: torch.Tensor,
        model: nn.Module,
        criterion: nn.Module,
        ntokens: int,
        args):
    model.eval()
    total_loss = 0.
    if args.model != 'Transformer' and args.model != 'S2STransformer':
        hidden = model.init_hidden(EVAL_BATCH_SIZE)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, args.bptt):
            samples, targets = get_batch(eval_data, i, args.bptt)
            if args.model == 'Transformer':
                output = model(samples)
                output = output.view(-1, ntokens)
            elif args.model == 'S2STransformer':
                output = model(samples, samples)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(samples, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(samples) * criterion(output, targets).item()
        avg_loss = total_loss / (eval_data.size(0) - 1)
    return avg_loss, math.exp(avg_loss)


# parse args
args = parse_args()

if args.save == '':
    SAVE_PATH = f'./ckpts/{args.model}_{args.emsize}_{args.epochs}.pt'

# set up logger
logger = setup_logger(args)
logger.info(' '.join(sys.argv))

# set up tensorboard
writer = SummaryWriter()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        logger.info("You have a CUDA device. It's suggested that you run with --cuda.")

# set device
device = torch.device("cuda" if args.cuda else "cpu")

# load data and split data into batches
corpus = data.Corpus(args.data)

logger.info(f'Training set size {len(corpus.train):,}')
train_data = batchify(corpus.train, args.batch_size, device)
val_data = batchify(corpus.valid, EVAL_BATCH_SIZE, device)
test_data = batchify(corpus.test, EVAL_BATCH_SIZE, device)

# build model
criterion = nn.NLLLoss()

ntokens = len(corpus.dictionary)
model: nn.Module = get_model(args.model, ntokens, args).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
# optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

logger.info(f"Vocabulary Size: {ntokens}")
logger.info(f"Num of Trainable Params: {num_params / 1e6:.2f}M")

# training loop
best_val_loss = None
best_epoch = None
for epoch in range(1, args.epochs + 1):
    train_loss, train_ppl = train(
        epoch, train_data, model, criterion, optimizer, ntokens, args)

    val_loss, val_ppl = evaluate(
        val_data, model, criterion, ntokens, args)

    logger.info(
        f'| End of epoch {epoch:3d} | Valid Loss {val_loss:9.4f} | Valid Ppl {val_ppl:9.4f} |')

    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Val', val_loss, epoch)
    writer.add_scalar('Perplexity/Train', train_ppl, epoch)
    writer.add_scalar('Perplexity/Val', val_ppl, epoch)

    # save the model if validation loss is the best we've seen
    if best_val_loss is None or val_loss < best_val_loss:
        with open(SAVE_PATH, 'wb') as f:
            torch.save(model, f)
            logger.info(f'Saved model to {SAVE_PATH}')
        best_val_loss = val_loss
        best_epoch = epoch

# testing
# load saved model
with open(SAVE_PATH, 'rb') as f:
    model = torch.load(f)
    logger.info(f'Loaded best model from {SAVE_PATH} (Epoch {best_epoch})')
    # makes params continuous in memory to speed up forward pass
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU', 'BiLSTM', 'BiGRU']:
        model.rnn.flatten_parameters()

# run on test data
test_loss, test_ppl = evaluate(test_data, model, criterion, ntokens, args)
logger.info(f'| Model arch {args.model:>10s} | Num of Params {num_params / 1e6:3.2f}M |')
logger.info(f'| Test loss {test_loss:6.3f} | Test Ppl {test_ppl:9.2f} |')
logger.info(f'Best model is from epoch {best_epoch}')

writer.close()
