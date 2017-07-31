import argparse
import os
import sys
import gym

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from model import Model, Shared_grad_buffers, Shared_obs_stats
from train import train
from test import test
from chief import chief
from utils import TrafficLight, Counter

class Params():
    def __init__(self):
        self.batch_size = 1000
        self.lr = 3e-4
        self.gamma = 0.99
        self.gae_param = 0.95
        self.clip = 0.2
        self.ent_coeff = 0.
        self.num_epoch = 10
        self.num_steps = 1000
        self.exploration_size = 1000
        self.num_processes = 4
        self.update_treshold = self.num_processes - 1
        self.max_episode_length = 10000
        self.seed = 1
        self.env_name = 'InvertedPendulum-v1'
        #self.env_name = 'Reacher-v1'
        #self.env_name = 'Pendulum-v0'
        #self.env_name = 'Hopper-v1'
        #self.env_name = 'Ant-v1'
        #self.env_name = 'Humanoid-v1'
        #self.env_name = 'HalfCheetah-v1'

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    params = Params()
    torch.manual_seed(params.seed)
    env = gym.make(params.env_name)
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]
    traffic_light = TrafficLight()
    counter = Counter()

    shared_model = Model(num_inputs, num_outputs)
    shared_model.share_memory()
    shared_grad_buffers = Shared_grad_buffers(shared_model)
    #shared_grad_buffers.share_memory()
    shared_obs_stats = Shared_obs_stats(num_inputs)
    #shared_obs_stats.share_memory()
    optimizer = optim.Adam(shared_model.parameters(), lr=params.lr)
    test_n = torch.Tensor([0])
    test_n.share_memory_()

    processes = []
    p = mp.Process(target=test, args=(params.num_processes, params, shared_model, shared_obs_stats, test_n))
    p.start()
    processes.append(p)
    p = mp.Process(target=chief, args=(params.num_processes, params, traffic_light, counter, shared_model, shared_grad_buffers, optimizer))
    p.start()
    processes.append(p)
    for rank in range(0, params.num_processes):
        p = mp.Process(target=train, args=(rank, params, traffic_light, counter, shared_model, shared_grad_buffers, shared_obs_stats, test_n))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
