import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.multiprocessing as mp

class Model(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Model, self).__init__()
        h_size_1 = 100
        h_size_2 = 100
        self.p_fc1 = nn.Linear(num_inputs, h_size_1)
        self.p_fc2 = nn.Linear(h_size_1, h_size_2)
        self.v_fc1 = nn.Linear(num_inputs, h_size_1)
        self.v_fc2 = nn.Linear(h_size_1, h_size_2)
        self.mu = nn.Linear(h_size_2, num_outputs)
        self.sigma_sq = nn.Linear(h_size_2, num_outputs)
        self.v = nn.Linear(h_size_2,1)
        #self.mu.weight.data.normal_()
        #self.sigma_sq.weight.data.normal_()
        for name, p in self.named_parameters():
            self.register_buffer(name+'_grad', torch.zeros(p.size()))
        self.train()

    def forward(self, inputs):
        x = F.tanh(self.p_fc1(inputs))
        x = F.tanh(self.p_fc2(x))
        mu = self.mu(x)
        sigma_sq = F.softplus(self.sigma_sq(x))
        x = F.tanh(self.v_fc1(inputs))
        x = F.tanh(self.v_fc2(x))
        v = self.v(x)
        return mu, sigma_sq, v

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
