import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Policy(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 300)
        self.mu = nn.Linear(300, num_outputs)
        self.sigma_sq = nn.Linear(300, num_outputs)
        self.mu.weight.data.normal_()
        self.sigma_sq.weight.data.normal_()
        self.train()

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        mu = self.mu(x)
        sigma_sq = F.softplus(self.sigma_sq(x))
        return mu, sigma_sq

class Value(torch.nn.Module):
    def __init__(self, num_inputs):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 300)
        self.v = nn.Linear(300,1)
        self.train()

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        v = self.v(x)
        return v
