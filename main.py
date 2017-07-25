import argparse
import os
import sys
import gym

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from model import Model
from train import train
from test import test
from chief import chief
from utils import TrafficLight, Counter
import my_optim

class Params():
    def __init__(self):
        self.batch_size = 64
        self.lr = 3e-4
        self.gamma = 0.99
        self.gae_param = 0.95
        self.clip = 0.2
        self.ent_coeff = 0.
        self.num_epoch = 10
        self.num_steps = 2048
        self.num_processes = 4
        self.update_treshold = self.num_processes - 1
        self.max_episode_length = 10000
        self.seed = 1
        self.env_name = 'InvertedPendulum-v1'
        #self.env_name = 'Reacher-v1'
        #self.env_name = 'Pendulum-v0'
        #self.env_name = 'Hopper-v1'

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

    shared_model = Model(num_inputs, num_outputs)
    shared_model.share_memory()
    optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=params.lr)

    processes = []
    p = mp.Process(target=test, args=(params.num_processes, params, shared_model))
    p.start()
    processes.append(p)
    p = mp.Process(target=chief, args=(params.num_processes, params, traffic_light, counter, shared_model, optimizer))
    p.start()
    processes.append(p)
    for rank in range(0, params.num_processes):
        p = mp.Process(target=train, args=(rank, params, traffic_light, counter, lock, shared_model, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
