import numpy as np
from .offline_algorithm import OfflineAlgorithm
from .stochastic_virtual import StochasticVirtual
from .stochastic_optimistic import StochasticOptimistic
from .stochastic_modified_virtual import StochasticModifiedVirtual
from .base import Algorithm, RandomAlgorithm, AlgorithmType

from dataclasses import dataclass, field
from typing import Iterable, List, Union
import tqdm


@dataclass
class OnlineParams:
    online_type: List[AlgorithmType] = field(default_factory=lambda: [AlgorithmType.STOCHASTIC_VIRTUAL])
    N: int = 5
    K: int = 1
    threshold: int = 0 # This will be reset in create_online_algorithm
    exhaust: bool = False # Exhaust K


def create_algorithm(online_type: Union[AlgorithmType, List[AlgorithmType]], params: OnlineParams = OnlineParams()):
    if isinstance(online_type, AlgorithmType):
        online_type = (online_type, )

    list_algorithms = []
    for alg_type in online_type:
        if params.threshold == 0:
            threshold = np.floor(params.N / np.e)
        else:
            threshold = params.threshold

        if alg_type == AlgorithmType.STOCHASTIC_VIRTUAL:
            algorithm = StochasticVirtual(params.N, params.K, threshold,
                    params.exhaust)
        elif alg_type == AlgorithmType.STOCHASTIC_OPTIMISTIC:
            algorithm = StochasticOptimistic(params.N, params.K, threshold,
                    params.exhaust)
        elif alg_type == AlgorithmType.STOCHASTIC_MODIFIED_VIRTUAL:
            algorithm = StochasticModifiedVirtual(params.N, params.K, threshold,
                    params.exhaust)
        elif alg_type == AlgorithmType.OFFLINE:
            algorithm = OfflineAlgorithm(params.K)
        elif alg_type == AlgorithmType.RANDOM:
            algorithm = RandomAlgorithm(params.N, params.K)
        else:
            raise ValueError(f"Unknown online algo type: '{alg_type}'.")

        list_algorithms.append(algorithm)
    
    return list_algorithms


def create_online_algorithm(params: OnlineParams = OnlineParams()) -> (Algorithm, Algorithm):
    return create_algorithm([AlgorithmType.OFFLINE] + list(params.online_type), params)


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

    indices_list = dict((algorithm.name, algorithm.S) for algorithm in algorithm_list)
    return indices_list


def compute_competitive_ratio(online_indices: Iterable, offline_indices: Iterable, knapsack=False) -> int:
    if knapsack:
        online_value = compute_knapsack_online_value(online_indices)
        offline_value = compute_knapsack_online_value(offline_indices)
        comp_ratio = online_value/offline_value
    else:
        online_indices = set([x[1] for x in online_indices])
        offline_indices = set([x[1] for x in offline_indices])
        comp_ratio = len(list(online_indices & offline_indices))
    return comp_ratio


def compute_knapsack_online_value(online_indices: Iterable) -> float:
    if len(online_indices) > 0:
        online_value = sum([x[0] for x in online_indices])
    else:
        online_value = 0.0
    return online_value
