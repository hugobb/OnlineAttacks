import numpy as np
from .offline_algorithm import OfflineAlgorithm
from .stochastic_virtual import StochasticVirtual
from .stochastic_optimistic import StochasticOptimistic
from .base import Algorithm
from enum import Enum
from dataclasses import dataclass
from typing import Iterable, List, Union
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
    threshold: int = 0 # This will be reset in create_online_algorithm
    exhaust: bool = False # Exhaust K


def create_algorithm(params: OnlineParams = OnlineParams()):
    if params.threshold == 0:
        threshold = np.floor(params.N / np.e)
    else:
        threshold = params.threshold

    if params.online_type == AlgorithmType.STOCHASTIC_VIRTUAL:
        online_algorithm = StochasticVirtual(params.N, params.K, threshold,
                params.exhaust)
    elif params.online_type == AlgorithmType.STOCHASTIC_OPTIMISTIC:
        online_algorithm = StochasticOptimistic(params.N, params.K, threshold,
                params.exhaust)
    else:
        raise ValueError(f"Unknown online algo type: '{online_type}'.")
    
    return online_algorithm


def create_online_algorithm(params: OnlineParams = OnlineParams()) -> (Algorithm, Algorithm):
    offline_algorithm = OfflineAlgorithm(params.K)
    online_algorithm = create_algorithm(params)
    return offline_algorithm, online_algorithm


def compute_indices(data_stream: Iterable, algorithm_list: Union[Algorithm, List[Algorithm]], pbar_flag=False) -> Union[Iterable, List[Iterable]]:
    if isinstance(algorithm_list, Algorithm):
        algorithm_list = (algorithm_list, ) 
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

