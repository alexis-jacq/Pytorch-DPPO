import argparse
import os
import sys
import gym

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from model import Policy, Value
from train import train
from test import test
from chief import chief
from utils import TrafficLight, Counter
import my_optim

class Params():
    def __init__(self):
        self.batch_size = 10000
        self.lr = 1e-4
        self.gamma = 0.99
        self.kl_target = 1e-2
        self.alpha = 1.5
        self.ksi = 1000
        self.beta_hight = 1.3
        self.beta_low = 0.7
        self.num_processes = 4
        self.num_steps = 20
        self.update_treshold = self.num_processes - 1
        self.max_episode_length = 10000
        self.seed = 1
        self.env_name = 'Pendulum-v0'

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    params = Params()
    torch.manual_seed(params.seed)
    env = gym.make(params.env_name)
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]
    traffic_light = TrafficLight()
    counter = Counter()
    lock = mp.Lock()

    shared_p = Policy(num_inputs, num_outputs)
    shared_v = Value(num_inputs)
    shared_p.share_memory()
    shared_v.share_memory()
    optimizer_p = my_optim.SharedAdam(shared_p.parameters(), lr=params.lr)
    optimizer_v = my_optim.SharedAdam(shared_v.parameters(), lr=params.lr)

    processes = []
    p = mp.Process(target=test, args=(params.num_processes, params, shared_p))
    p.start()
    processes.append(p)
    p = mp.Process(target=chief, args=(params.num_processes, params, traffic_light, counter, shared_p, shared_v, optimizer_p, optimizer_v))
    p.start()
    processes.append(p)
    for rank in range(0, params.num_processes):
        p = mp.Process(target=train, args=(rank, params, traffic_light, counter, lock, shared_p, shared_v, optimizer_p, optimizer_v))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
