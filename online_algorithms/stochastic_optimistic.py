from base import Algorithm


class StochasticOptimistic(Algorithm):
    def __init__(self, N: int, k : int, threshold: int):
        """ Construct Stochastic Virtual
        Parameters:
            N (int)     -- number of data points
                k (int)     -- number of attacks
        """
        super().__init__(k)
        self.N = N
        self.threshold = threshold
        self.R = []
        self.sampling_phase = True

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
            if len(self.R) > 0:
                k_value, k_index = self.R[-1]
                if value > k_value:
                    # Update and pick
                    self.S.append([value, index])
                    self.R = self.R[:-2]


if __name__ == "__main__":
    algorithm = StochasticOptimistic(10, 1, 5)
    algorithm.reset()
    algorithm.action(1, 1)