from .base import Algorithm, AlgorithmType
import numpy as np


class StochasticSingleRef(Algorithm):
    C_DEFAULT = {"default": 0.3678, 1: 0.3678, 10: 0.2159, 100:  0.1331}
    R_DEFAULT = {"default": 1, 1: 1, 10: 3, 100: 15, 1000: 150}
    
    @classmethod
    def get_default_c(cls, k: int) -> float:
        if k in cls.C_DEFAULT:
            return cls.C_DEFAULT[k]
        return cls.C_DEFAULT["default"]

    @classmethod
    def get_default_r(cls, k: int) -> int:
        if k in cls.R_DEFAULT:
            return cls.R_DEFAULT[k]
        return cls.R_DEFAULT["default"]

    def __init__(self, N: int, k : int, r: int, threshold: int, exhaust: bool):
        """ Construct Stochastic Optimistic
        Parameters:
            N (int)           -- number of data points
            k (int)           -- number of attacks
            r (int)           -- reference rank
            threshold (int)   -- threshold for t
            exhaust (bool)    -- whether to exhaust K
        """
        super().__init__(k)
        self.N = N
        self.r = self.get_default_r(k)

        if threshold is None:
            threshold = np.floor(self.get_default_c(k)*N + 1)

        self.threshold = threshold
        self.R = []
        self.sampling_phase = True
        self.exhaust = exhaust
        self.name = AlgorithmType.STOCHASTIC_SINGLE_REF.name

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
            num_picked = len(self.S)
            num_left_to_pick = self.k - num_picked
            num_samples_left = self.N - index
            if num_left_to_pick > 0:
                r_value, r_index = self.R[self.r]
                if num_samples_left <= num_left_to_pick and self.exhaust:
                    # Just Pick the last samples to exhaust K
                    self.S.append([value, index])
                elif value > r_value:
                    # Pick
                    self.S.append([value, index])


if __name__ == "__main__":
    algorithm = StochasticSingleRef(10, 1, 5)
    algorithm.reset()
    algorithm.action(1, 1)
