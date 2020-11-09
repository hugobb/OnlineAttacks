from base import Algorithm


class StochasticVirtual(Algorithm):
    def __init__(self, N: int, k: int, threshold: int):
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
            k_value, k_index = self.R[-1]

            if value < k_value:
                # Don't pick or Update R
                pass
            elif value > k_value and k_index <= self.threshold:
                # Update and pick
                self.S.append([value, index])
                self.R.append([value, index])
                self.R.sort(key=lambda tup: tup[0], reverse=True)  # sorts in place
                self.R = self.R[:self.k]
            elif value > k_value and k_index > self.threshold:
                self.R.append([value, index])
                self.R.sort(key=lambda tup: tup[0], reverse=True)  # sorts in place
                self.R = self.R[:self.k]


if __name__ == "__main__":
    algorithm = StochasticVirtual(10, 1, 5)
    algorithm.reset()
    algorithm.action(1, 1)