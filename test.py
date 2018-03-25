# test

import os
import sys
import gym
from gym import wrappers
import time
from collections import deque
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from model import Model

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def test(rank, params, shared_model, shared_obs_stats, test_n):
    torch.manual_seed(params.seed + rank)
    work_dir = mkdir('exp', 'ppo')
    monitor_dir = mkdir(work_dir, 'monitor')
    env = gym.make(params.env_name)
    #env = wrappers.Monitor(env, monitor_dir, force=True)
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]
    model = Model(num_inputs, num_outputs)

    state = env.reset()
    state = Variable(torch.Tensor(state).unsqueeze(0))
    reward_sum = 0
    done = True

    start_time = time.time()

    episode_length = 0
    while True:
        episode_length += 1
        model.load_state_dict(shared_model.state_dict())
        shared_obs_stats.observes(state)
        #print(shared_obs_stats.n[0])
        state = shared_obs_stats.normalize(state)
        mu,sigma_sq,_ = model(state)
        eps = torch.randn(mu.size())
        action = mu + sigma_sq.sqrt()*Variable(eps)
        env_action = action.data.squeeze().numpy()
        state, reward, done, _ = env.step(env_action)
        reward_sum += reward

        if done:
            print("Time {}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                reward_sum, episode_length))
            reward_sum = 0
            episode_length = 0
            state = env.reset()
            time.sleep(10)

        state = Variable(torch.Tensor(state).unsqueeze(0))
