import numpy as np
from .offline_algorithm import OfflineAlgorithm
from .stochastic_virtual import StochasticVirtual
from .stochastic_optimistic import StochasticOptimistic
from .base import Algorithm
from enum import Enum
from dataclasses import dataclass
from typing import Iterable, List
import tqdm


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


def compute_indices(data_stream: Iterable, algorithm_list: List[Algorithm], pbar_flag=False) -> List[Iterable]:
    for algorithm in algorithm_list:
        algorithm.reset()

    if pbar_flag:
        pbar = tqdm.tqdm(total=len(data_stream))

    index = 0
    for data in data_stream:
        if not isinstance(data, Iterable):
            data = [data]
        for value in data:
            value = float(value)
            for algorithm in algorithm_list:
                algorithm.action(value, index)
            index += 1
        
        if pbar_flag:
            pbar.update()

    if pbar_flag:
        pbar.close()
    
    indices_list = tuple(algorithm.S for algorithm in algorithm_list)
    return indices_list


def compute_competitive_ratio(online_indices: Iterable, offline_indices: Iterable) -> int:
    online_indices = set([x[1] for x in online_indices])
    offline_indices = set([x[1] for x in offline_indices])
    comp_ratio = len(list(online_indices & offline_indices))
    return comp_ratio

