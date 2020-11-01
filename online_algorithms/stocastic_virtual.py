import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad, Variable
import ipdb


class stochastic_virtual(nn.Module):
	def __init__(self, N, k):
        """ Construct Stochastic Virtual
        Parameters:
            N (int)     -- number of data points
         	k (int)     -- number of attacks
        """
        super(stochastic_virtual, self).__init__()
        self.N = N
        self.k = k
        self.R = []
        self.S = []
        self.sampling_phase = True

    def action(self, value, index):
    	if self.sampling_phase:
    		
