import torch
import torch.nn as nn
import torch.nn.functional as F
from operator import itemgetter
from torch.autograd import grad, Variable
import ipdb


class stochastic_virtual(nn.Module):
    def __init__(self, N, k, threshold):
        """ Construct Stochastic Virtual
        Parameters:
            N (int)     -- number of data points
                k (int)     -- number of attacks
        """
        super(stochastic_virtual, self).__init__()
        self.N = N
        self.k = k
        self.threshold = threshold
        self.R = []
        self.S = []
        self.sampling_phase = True

    def reset(self):
        self.R = []
        self.S = []
        self.sampling_phase = True

    def action(self, value, index):
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

