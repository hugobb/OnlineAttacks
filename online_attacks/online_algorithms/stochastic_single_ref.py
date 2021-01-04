from .base import Algorithm, AlgorithmType


class StochasticSingleRef(Algorithm):
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
        self.r = r
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
            if len(self.R) > 0:
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
