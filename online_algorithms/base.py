class Algorithm:
    def __init__(self, k: int):
        self.k = k
        self.S = []

    def action(self, value: float, index: int):
        raise NotImplementedError()

    def reset(self):
        self.S = []