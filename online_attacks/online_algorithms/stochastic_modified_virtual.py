from .base import Algorithm, AlgorithmType
from typing import Optional
import numpy as np

class StochasticModifiedVirtual(Algorithm):
    def __init__(self, N: int, k: int, threshold: Optional[int] = None, exhaust: bool = False):
        """ Construct Stochastic Virtual
        Parameters:
            N (int)           -- number of data points
            k (int)           -- number of attacks
            threshold (int)   -- threshold for t
            exhaust (bool)    -- whether to exhaust K
        """
        super().__init__(k)
        self.N = N

        if threshold is None:
            threshold = np.floor(N / np.e)
        self.threshold = threshold

        self.R = []
        self.sampling_phase = True
        self.exhaust = exhaust
        self.name = AlgorithmType.STOCHASTIC_MODIFIED_VIRTUAL.name

    def reset(self):
        super().reset()
        self.R = []
        self.sampling_phase = True

    def action(self, value: float, index: int):
        if self.sampling_phase:
            self.R.append([value, index])
            self.R.sort(key=lambda tup: tup[0], reverse=True)  # sorts in place
            self.R = self.R[:self.k]

            if index >= self.threshold:
                self.sampling_phase = False
        else:
            k_value, k_index = self.R[-1]
            num_picked = len(self.S)
            num_left_to_pick = self.k - num_picked
            num_samples_left = self.N - index
            if num_samples_left <= num_left_to_pick and self.exhaust and num_left_to_pick > 0:
                # Just Pick the last samples to exhaust K
                self.S.append([value, index])
            elif value > k_value and num_left_to_pick > 0:
                # Update and pick
                self.S.append([value, index])
                self.R.append([value, index])
                self.R.sort(key=lambda tup: tup[0], reverse=True)  # sorts in place
                self.R = self.R[:self.k]



if __name__ == "__main__":
    algorithm = StochasticVirtualRef(10, 1, 5)
    algorithm.reset()
    algorithm.action(1, 1)
