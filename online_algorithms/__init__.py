from typing import Any
import argparse
import numpy as np
from .offline_algorithm import OfflineAlgorithm
from .stochastic_virtual import StochasticVirtual
from .stochastic_optimistic import StochasticOptimistic
from .base import Algorithm
from torch.utils.data import Dataset

__all__ = [
    "Algorithm",
    "OfflineAlgorithm",
    "StochasticVirtual",
    "StochasticOptimistic",
]


def create_online_algorithm(arg_parse: argparse.Namespace, online_type: str, N:
                            int, K: int, **kwargs: Any) -> (Algorithm, Algorithm):
    threshold = np.floor(N / np.e)
    offline_algorithm = OfflineAlgorithm(K)
    if online_type == 'stochastic_virtual':
        online_algorithm = StochasticVirtual(N, K, threshold)
    elif online_type == 'stochastic_optimistic':
        online_algorithm = StochasticOptimistic(N, K, threshold)
    else:
        raise ValueError(f"Unknown online algo type: '{online_type}'.")
    return offline_algorithm, online_algorithm


def compute_competitive_ratio(data_stream: Dataset, online_algorithm: Algorithm, offline_algorithm: Algorithm) -> int:
    offline_algorithm.reset()
    online_algorithm.reset()
    for index, data in enumerate(data_stream):
        online_algorithm.action(data, index)
        offline_algorithm.action(data, index)
    online_indices = set([x[1] for x in online_algorithm.S])
    offline_indices = set([x[1] for x in offline_algorithm.S])
    comp_ratio = len(list(online_indices & offline_indices))
    return comp_ratio

