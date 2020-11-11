import numpy as np
from .offline_algorithm import OfflineAlgorithm
from .stochastic_virtual import StochasticVirtual
from .stochastic_optimistic import StochasticOptimistic
from .base import Algorithm
from enum import Enum
import tqdm
from dataclasses import dataclass
from typing import Iterable


class AlgorithmType(Enum):
    OFFLINE = "offline_algorithm"
    STOCHASTIC_VIRTUAL = "stochastic_virtual"
    STOCHASTIC_OPTIMISTIC = "stochastic_optimistic"


@dataclass
class OnlineParams:
    online_type: AlgorithmType = AlgorithmType.STOCHASTIC_VIRTUAL
    N: int = 5
    K: int = 1


def create_online_algorithm(params: OnlineParams = OnlineParams()) -> (Algorithm, Algorithm):
    threshold = np.floor(params.N / np.e)
    offline_algorithm = OfflineAlgorithm(params.K)
    if params.online_type == AlgorithmType.STOCHASTIC_VIRTUAL:
        online_algorithm = StochasticVirtual(params.N, params.K, threshold)
    elif params.online_type == AlgorithmType.STOCHASTIC_OPTIMISTIC:
        online_algorithm = StochasticOptimistic(params.N, params.K, threshold)
    else:
        raise ValueError(f"Unknown online algo type: '{online_type}'.")
    return offline_algorithm, online_algorithm


def compute_competitive_ratio(data_stream: Iterable, online_algorithm: Algorithm, offline_algorithm: Algorithm) -> int:
    offline_algorithm.reset()
    online_algorithm.reset()
    for index, data in enumerate(data_stream):
        if not isinstance(data, Iterable):
            data = [data]
        for value in data:
            value = float(value)
            online_algorithm.action(value, index)
            offline_algorithm.action(value, index)
            index += 1
    online_indices = set([x[1] for x in online_algorithm.S])
    offline_indices = set([x[1] for x in offline_algorithm.S])
    comp_ratio = len(list(online_indices & offline_indices))
    return comp_ratio

