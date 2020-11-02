import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad, Variable
import ipdb

class offline(nn.Module):
    def __init__(self, k, threshold):
        """ Construct Offline Algorithm
        Parameters:
            N (int)     -- number of data points
            k (int)     -- number of attacks
        """
        super(offline, self).__init__()
        self.k = k
        self.S = []

    def reset(self):
        self.S = []

    def action(self, value, index):
        self.S.append([value, index])
        self.S.sort(key=lambda tup: tup[0], reverse=True)  # sorts in place
        self.S = self.S[:self.k]

