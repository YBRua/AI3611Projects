import torch

from torch import Tensor


def linear_softmax_pooling(x: Tensor):
    return (x ** 2).sum(1) / x.sum(1)


def weighted_sum_pooling(x: Tensor, w: Tensor):
    w = torch.clip(w, 1e-7, 1.)
    return x.sum(1) / w.sum(1)
