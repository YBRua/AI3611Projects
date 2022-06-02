import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from typing import Callable
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import data_prep
import vaeplotlib as vplot
from loss_funcs import BCEKLDLoss, BCELoss
from models import get_vae, get_ae
from common import setup_logger


def train_epoch(
        e: int,
        dataloader: DataLoader,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: Callable,
        device: torch.device,
        is_ae: bool,
        file_postfix: str,
        no_tqdm: bool = False):
    model.train()
    if no_tqdm:
        progress = dataloader
    else:
        progress = tqdm(dataloader)
    tot_loss = 0.0
    tot_recon_loss = 0.0
    tot_hidden_loss = 0.0
    avg_loss = 0.0
    avg_recon_loss = 0.0
    avg_hidden_loss = 0.0
    for bid, (x, _) in enumerate(progress):
        x = x.to(device)
        if is_ae:
            remake_x, _ = model(x)
            loss_val = loss_fn(x.detach().clone(), remake_x)
            tot_loss += loss_val.item()
        else:
            remake_x, mean, log_std = model(x)
            loss_val, recon_loss, hidden_loss = loss_fn(
                x.detach().clone(), remake_x, mean, log_std)

            tot_loss += loss_val.item()
            tot_recon_loss += recon_loss.item()
            tot_hidden_loss += hidden_loss.item()

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        if is_ae:
            avg_loss = tot_loss / (bid + 1)
            if not no_tqdm:
                progress.set_description(
                    '| Training '
                    f'| Epoch: {e:2} '
                    f'| Loss: {avg_loss:9.4f} |')
        else:
            avg_loss = tot_loss / (bid + 1)
            avg_recon_loss = tot_recon_loss / (bid + 1)
            avg_hidden_loss = tot_hidden_loss / (bid + 1)

            if not no_tqdm:
                progress.set_description(
                    '| Training '
                    f'| Epoch {e:2} |'
                    f' Loss {avg_loss:9.4f} |'
                    f' Recon {avg_recon_loss:9.4f} |'
                    f' KLDiv {avg_hidden_loss:9.4f} |')

        if bid == 0 and not is_ae:
            nelem = x.size(0)
            nrow = 8
            save_image(
                x.view(nelem, 1, 28, 28),
                f'./images/image-x{file_postfix}.png', nrow=nrow)
            save_image(
                remake_x.view(nelem, 1, 28, 28),
                f'./images/image-recon{file_postfix}.png', nrow=nrow)

    return avg_loss, avg_recon_loss, avg_hidden_loss


def test_epoch(
        e: int,
        dataloader: DataLoader,
        model: nn.Module,
        loss_fn: Callable,
        device: torch.device,
        is_ae: bool):
    model.eval()
    progress = dataloader
    tot_loss = 0.0
    tot_recon_loss = 0.0
    tot_hidden_loss = 0.0
    avg_loss = 0.0
    avg_recon_loss = 0.0
    avg_hidden_loss = 0.0
    for bid, (x, _) in enumerate(progress):
        x = x.to(device)
        if is_ae:
            remake_x, _ = model(x)
            loss_val = loss_fn(x.detach().clone(), remake_x)
            tot_loss += loss_val.item() * x.shape[0]
        else:
            remake_x, mean, log_std = model(x)
            loss_val, recon_loss, hidden_loss = loss_fn(
                x, remake_x, mean, log_std)

            tot_loss += loss_val.item() * x.shape[0]
            tot_recon_loss += recon_loss.item() * x.shape[0]
            tot_hidden_loss += hidden_loss.item() * x.shape[0]

    avg_loss = tot_loss / len(dataloader.dataset)
    avg_recon_loss = tot_recon_loss / len(dataloader.dataset)
    avg_hidden_loss = tot_hidden_loss / len(dataloader.dataset)

    return avg_loss, avg_recon_loss, avg_hidden_loss


def parse_args():
    parser = ArgumentParser()

    # general training settings
    parser.add_argument(
        '--data_root', type=str, default='data',
        help='Path to dataset root')
    parser.add_argument(
        '--batch_size', type=int, default=64)
    parser.add_argument(
        '--epochs', type=int, default=50)
    parser.add_argument(
        '--lr', type=float, default=1e-3)
    parser.add_argument(
        '--device', type=str, default='cuda')
    parser.add_argument(
        '--seed', type=int, default=42)
    parser.add_argument(
        '--skip_training', action='store_true',
        help='Skip training and only sample')
    parser.add_argument(
        '--model_save', type=str, default=None)

    # model settings
    parser.add_argument(
        '--ae', action='store_true',
        help='Use AE if enabled. Otherwise a VAE is used.')
    parser.add_argument(
        '--encoder', choices=['MLP', 'MLP3', 'CONV'],
        default='MLP3',
        help='Encoder architecture')
    parser.add_argument(
        '--decoder', choices=['MLP', 'MLP3', 'CONV'],
        default='MLP3',
        help='Decoder architecture')
    parser.add_argument(
        '--alpha', type=float, default=1.0,
        help='Weight of regularization loss')
    parser.add_argument(
        '--z_dim', type=int, default=2,
        help='Size of hidden dimension z')
    parser.add_argument(
        '--no_tqdm', action='store_true',
        help='Disable tqdm, used in slurm')

    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    Z_DIM = args.z_dim
    ALPHA = args.alpha

    N_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    DO_TRAINING = not args.skip_training
    AE_LABEL = '-AE' if args.ae else ''
    FILE_POSTFIX = f'{AE_LABEL}-{args.encoder}-{args.decoder}-{Z_DIM}'

    if args.model_save is None:
        MODEL_SAVE = f'model{AE_LABEL}_{args.encoder}_{args.decoder}_{Z_DIM}'
        MODEL_SAVE = f'./ckpts/{MODEL_SAVE}.pt'
    else:
        MODEL_SAVE = args.model_save
    if DO_TRAINING:
        logger = setup_logger(args)

    device = torch.device(args.device)

    # build model
    if args.ae:
        model = get_ae(args.encoder, args.decoder, Z_DIM)
    else:
        model = get_vae(args.encoder, args.decoder, Z_DIM)
    n_params = sum(p.numel() for p in model.parameters())
    model.to(device)

    # init optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.ae:
        loss_fn = BCELoss()
    else:
        loss_fn = BCEKLDLoss(alpha=ALPHA)

    # prepare data
    train_dataset, test_dataset = data_prep.get_mnist()
    train_loader = data_prep.wrap_dataloader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = data_prep.wrap_dataloader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # training loop
    best_test_loss = 1e8
    best_epoch = -1

    if DO_TRAINING:
        logger.info(f'Encoder: {args.encoder} | Decoder: {args.decoder}')
        logger.info(f'Number of parameters: {n_params}')
        logger.info('Training...')

        losses = []
        recon_losses = []
        kldiv_losses = []

        for e in range(N_EPOCHS):
            l_train, recon_train, kl_train = train_epoch(
                e, train_loader, model, optimizer,
                loss_fn, device, args.ae, FILE_POSTFIX, args.no_tqdm)
            if args.ae:
                logger.info(
                    f'| Epoch {e:2} '
                    f'| Train Loss {l_train:.4f} |')
            else:
                logger.info(
                    f'| Epoch {e:2} '
                    f'| Train Loss {l_train:9.4f} '
                    f'| Recon Loss {recon_train:9.4f} '
                    f'| KLDiv Loss {kl_train:9.4f} |')

            l_test, recon_test, kl_test = test_epoch(
                e, test_loader, model, loss_fn, device, args.ae)
            if args.ae:
                losses.append(l_test)
                logger.info(
                    f'| Epoch {e:2} '
                    f'| Test  Loss {l_test:.4f} |')
            else:
                losses.append(l_test)
                recon_losses.append(recon_test)
                kldiv_losses.append(kl_test)

                logger.info(
                    f'| Epoch {e:2} '
                    f'| Test  Loss {l_test:9.4f} '
                    f'| Recon Loss {recon_test:9.4f} '
                    f'| KLDiv Loss {kl_test:9.4f} |')

            if l_test < best_test_loss:
                best_test_loss = l_test
                best_epoch = e
                torch.save(model.state_dict(), MODEL_SAVE)

            # sample from model
            # if not args.ae:
            vplot.sample(100, Z_DIM, model, device, FILE_POSTFIX)

        logger.info(f'Number of parameters: {n_params}')
        logger.info(f'Best epoch {best_epoch} (loss {best_test_loss:9.4f})')

        if not args.ae:
            fig, axes = vplot.loss_curve(losses, recon_losses, kldiv_losses)
            fig.savefig(f'./images/loss-curve{FILE_POSTFIX}.png')

    # if not train
    else:
        print(f'Loading model from {MODEL_SAVE}')
        model.load_state_dict(torch.load(MODEL_SAVE))
        # if not args.ae:
        vplot.sample(100, Z_DIM, model, device, FILE_POSTFIX)

    # sample from model (2d and 1d latent space only)
    if Z_DIM == 2:
        model.load_state_dict(torch.load(MODEL_SAVE))
        model.eval()
        # if not args.ae:
        vplot.plot_grid_sample(-5, 5, 0.25, model, device, FILE_POSTFIX)
        fig, ax = vplot.plot_latent_space(test_loader, model, device)
        fig.savefig(f'./images/latent{FILE_POSTFIX}.png')
    if Z_DIM == 1 and not args.ae:
        model.load_state_dict(torch.load(MODEL_SAVE))
        model.eval()
        vplot.plot_line_sample(-5, 5, 0.05, model, device, FILE_POSTFIX)


if __name__ == '__main__':
    main()
