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


def block_mixing(
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
