from typing import Any
import argparse
import numpy as np
from .offline_algorithm import offline
from .stochastic_virtual import stochastic_virtual
from .stochastic_optimistic import stochastic_optimistic

__all__ = [
    "offline",
    "stochastic_virtual",
    "stochastic_optimistic",
]

def create_online_algorithm(arg_parse: argparse.Namespace, online_type: str, N:
                            int, K: int, **kwargs: Any):

    threshold = np.floor( N / np.e)
    offline_algorithm = offline(K, threshold)
    if online_type == 'stochastic_virtual':
        online_algorithm = stochastic_virtual(N, K , threshold)
    elif online_type == 'stochastic_optimistic':
        online_algorithm = stochastic_virtual(N, K , threshold)
    else:
        raise ValueError(f"Unknown online algo type: '{online_type}'.")
    return offline_algorithm, online_algorithm
