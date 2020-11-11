import numpy as np
from torch.utils.data import TensorDataset, Dataset
import torch


class ToyDatastream:
    def __init__(self, N: int, max_perms: int = 120):
        self.perms = []
        total_perms = np.math.factorial(N)
        max_perms = np.minimum(max_perms, total_perms)
        for i in range(max_perms):                        # (1) Draw N samples from permutations Universe U (#U = k!)
            while True:                             # (2) Endless loop
                perm = np.random.permutation(N)     # (3) Generate a random permutation form U
                key = tuple(perm)
                if key not in self.perms:                # (4) Check if permutation already has been drawn (hash table)
                    self.perms.append(key)               # (5) Insert into set
                    break
                pass

    def __getitem__(self, index: int) -> Dataset:
        self.perms[index]
        return self.perms[index]

    def __len__(self) -> int:
        return len(self.perms)