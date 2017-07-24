import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.multiprocessing as mp

class Policy(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 300)
        self.mu = nn.Linear(300, num_outputs)
        self.sigma_sq = nn.Linear(300, num_outputs)
        self.mu.weight.data.normal_()
        self.sigma_sq.weight.data.normal_()
        for name, p in self.named_parameters():
            self.register_buffer(name+'_grad', torch.zeros(p.size()))
        self.train()

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        mu = self.mu(x)
        sigma_sq = F.softplus(self.sigma_sq(x))
        return mu, sigma_sq

    def cum_grads(self):
        for name, p in self.named_parameters():
            if p.grad is not None:
                val = self.__getattr__(name+'_grad')
                val += p.grad.data
                self.__setattr__(name+'_grad', val)

    def reset_grads(self):
        self.zero_grad()
        for name, p in self.named_parameters():
            if p.grad is not None:
                val = self.__getattr__(name+'_grad')
                val = p.grad.data
                self.__setattr__(name+'_grad', val)

    def synchronize(self):
        for name, p in self.named_parameters():
            val = self.__getattr__(name+'_grad')
            p._grad = Variable(val)


class Value(torch.nn.Module):
    def __init__(self, num_inputs):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 300)
        self.v = nn.Linear(300,1)
        for name, p in self.named_parameters():
            self.register_buffer(name+'_grad', torch.zeros(p.size()))
        self.train()

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        v = self.v(x)
        return v

    def cum_grads(self):
        for name, p in self.named_parameters():
            if p.grad is not None:
                val = self.__getattr__(name+'_grad')
                val += p.grad.data
                self.__setattr__(name+'_grad', val)

    def synchronize(self):
        for name, p in self.named_parameters():
            val = self.__getattr__(name+'_grad')
            p._grad = Variable(val)

    def reset_grads(self):
        self.zero_grad()
        for name, p in self.named_parameters():
            if p.grad is not None:
                val = self.__getattr__(name+'_grad')
                val = p.grad.data
                self.__setattr__(name+'_grad', val)
