import random


class Algorithm:
    def __init__(self, k: int):
        self.k = k
        self.S = []

    def action(self, value: float, index: int):
        raise NotImplementedError()

    def reset(self):
        self.S = []


class RandomAlgorithm(Algorithm):
    def __init__(self, N: int, k:int):
        super().__init__(k)
        self.N = N
        self.random_permutation = random.sample(range(self.N), self.k)

    def action(self, value: float, index: int):
        if index in self.random_permutation:
            self.S.append([value, index])

    def reset(self):
        self.S = []