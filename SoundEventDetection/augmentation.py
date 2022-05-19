import torch
import numpy as np

from torch import Tensor
from typing import Tuple


def mixup(
        batch: Tuple[np.ndarray, Tensor, Tensor],
        n_samples: int = 8,
        alpha: float = 0.2):
    lamb = np.random.beta(alpha, alpha)
    aids, feats, targets = batch
    idx1 = torch.randperm(
        len(feats), dtype=torch.long)[:n_samples]
    idx2 = torch.randperm(
        len(feats), dtype=torch.long)[:n_samples]
    feats1 = feats[idx1]
    feats2 = feats[idx2]
    targets1 = targets[idx1]
    targets2 = targets[idx2]

    feats_extra = lamb * feats1 + (1 - lamb) * feats2
    targets_extra = lamb * targets1 + (1 - lamb) * targets2

    return feats_extra, targets_extra


def time_warp(
        batch: Tuple[np.ndarray, Tensor, Tensor],
        n_samples: int = 8,
        mean: int = 0,
        std: int = 50):
    aids, feats, targets = batch
    shift = torch.normal(mean, std).int().item()
    feats = torch.roll(x, shift, dims=1)


def spec_aug(
        batch: Tuple[np.ndarray, Tensor, Tensor],
        n_samples: int = 8,
        nt: int = 2,
        eta_t0: int = 60,
        nf: int = 2,
        eta_f0: int = 12,):
    aids, feats, targets = batch
    B, T, F = feats.shape

    for _ in range(nf):
        eta_f = np.random.randint(0, eta_f0)
        f_limit = max(F - eta_f, 1)
        f0 = np.random.randint(0, f_limit)
        feats[:, :, f0:f0 + eta_f] = 0

    for _ in range(nt):
        eta_t = np.random.randint(0, eta_t0)
        t_limit = max(T - eta_t, 1)
        t0 = np.random.randint(0, t_limit)
        feats[:, t0:t0 + eta_t, :] = 0

    return feats, targets


def xctx_block_mixing(
        batch: Tuple[np.ndarray, Tensor, Tensor],
        n_samples: int = 8):
    aids, feats, targets = batch
    idx1 = torch.randperm(
        len(feats), dtype=torch.long)[:n_samples]
    idx2 = torch.randperm(
        len(feats), dtype=torch.long)[:n_samples]

    feats1 = feats[idx1]
    feats2 = feats[idx2]
    targets1 = targets[idx1]
    targets2 = targets[idx2]

    feats_extra = feats1 + feats2
    targets_extra = torch.logical_and(targets1, targets2)

    return feats_extra, targets_extra
