from torch import Tensor


def linear_softmax_pooling(x: Tensor):
    return (x ** 2).sum(1) / x.sum(1)
