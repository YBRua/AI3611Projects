import math
import random
from typing import Type


class SamplingScheduler:
    def __init__(self, k: float) -> None:
        self.k = k

    def _get_prob(self, i: int) -> float:
        raise NotImplementedError('No you dont!')

    def __call__(self, i: int) -> bool:
        """Determines whether to use a sampled token
        or a ground truth token

        Args:
            i (int): current epoch

        Returns:
            bool: Returns `True` if should use sampled output,
        returns `False` if a ground truth should be used.
        """
        raise NotImplementedError('No you dont!')


class LinearScheduler(SamplingScheduler):
    def __init__(
            self,
            k: float = 1,
            c: float = 0.05,
            eps: float = 0.1) -> None:
        super().__init__(k)
        self.c = c
        self.eps = eps

    def _get_prob(self, i: int) -> float:
        return max(self.eps, self.k - self.c * i)

    def __call__(self, i: int) -> bool:
        prob = self._get_prob(i)
        return random.random() > prob


class ExpScheduler(SamplingScheduler):
    def __init__(self, k: float = 0.8) -> None:
        if k >= 1:
            raise ValueError('k must be less than 1')

        super().__init__(k)

    def _get_prob(self, i: int) -> float:
        return self.k ** i

    def __call__(self, i: int) -> bool:
        prob = self._get_prob(i)
        return random.random() > prob


class InvSigmoidScheduler(SamplingScheduler):
    def __init__(self, k: float = 10) -> None:
        if k < 1:
            raise ValueError('k must be >= 1')

        super().__init__(k)

    def _get_prob(self, i: int) -> float:
        return self.k / (self.k + math.exp(i / self.k))

    def __call__(self, i: int) -> bool:
        prob = self._get_prob(i)
        return random.random() > prob


def get_scheduler_constructor(sched: str) -> Type[SamplingScheduler]:
    """Returns a constructor for a scheduler
    specified by parameter `sched`

    Args:
        sched (str): Type of scheduler

    Returns:
        Type[SamplingScheduler]: Constructor of the corresponding scheduler
    """
    if sched == 'lin':
        return LinearScheduler
    elif sched == 'exp':
        return ExpScheduler
    elif sched == 'invsigmoid':
        return InvSigmoidScheduler
    elif sched == 'none':
        return None
    else:
        raise ValueError(f'No scheduler named {sched}')


if __name__ == "__main__":
    scheduler = InvSigmoidScheduler()
    es = []
    for e in range(45):
        bs = []
        for batch in range(10):
            print(scheduler(e))
            bs.append(scheduler(e))
        es.append(bs)

    for eid, choices in enumerate(es):
        print(eid, ' '.join(map(str, choices)))
