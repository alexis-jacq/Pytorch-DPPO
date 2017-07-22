import os
import sys
import gym
import time
from collections import deque
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from models import Policy, Value

def test(rank, params, shared_p):
    torch.manual_seed(params.seed + rank)
    env = gym.make(params.env_name)
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]
    policy = Policy(num_inputs, num_outputs)

    state = env.reset()
    state = Variable(torch.Tensor(state).unsqueeze(0))
    reward_sum = 0
    done = True

    start_time = time.time()

    episode_length = 0
    while True:
        episode_length += 1
        policy.load_state_dict(shared_p.state_dict())
        mu, _ = policy(state)
        action = mu.data
        env_action = action.squeeze().numpy()
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
            time.sleep(60)

        state = Variable(torch.Tensor(state).unsqueeze(0))
