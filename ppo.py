import os
import sys
import gym
from gym import wrappers
import random
import numpy as np
import math

import mujoco_py
import pybullet_envs

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model import Model, Shared_obs_stats

class Params():
    def __init__(self):
        self.batch_size = 64
        self.lr = 7e-4
        self.gamma = 0.99
        self.gae_param = 0.95
        self.clip = 0.2
        self.ent_coeff = 0.01
        self.num_epoch = 10
        self.num_steps = 2048
        self.time_horizon = 1000000
        self.max_episode_length = 10000
        self.max_grad_norm = 0.5
        self.seed = 1
        #self.env_name = 'InvertedPendulum-v1'
        #self.env_name = 'InvertedDoublePendulum-v1'
        #self.env_name = 'Reacher-v1'
        #self.env_name = 'Pendulum-v0'
        #self.env_name = 'HalfCheetahBulletEnv-v0'
        #self.env_name = 'HopperBulletEnv-v0'#'Hopper-v1'
        #self.env_name = 'Ant-v1'#'AntBulletEnv-v0'#
        self.env_name = 'HalfCheetah-v1'

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, events):
        for event in zip(*events):
            self.memory.append(event)
            if len(self.memory)>self.capacity:
                del self.memory[0]

    def clear(self):
        self.memory = []

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: torch.cat(x, 0), samples)

def train(env, model, optimizer, shared_obs_stats):
    memory = ReplayMemory(params.num_steps)
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]
    state = env.reset()
    state = Variable(torch.Tensor(state).unsqueeze(0))
    done = True
    episode = -1

    # horizon loop
    for t in range(params.time_horizon):
        episode_length = 0
        while(len(memory.memory)<params.num_steps):
            states = []
            actions = []
            rewards = []
            values = []
            returns = []
            advantages = []
            logprobs = []
            av_reward = 0
            cum_reward = 0
            cum_done = 0

            # n steps loops
            for step in range(params.num_steps):
                episode_length += 1
                shared_obs_stats.observes(state)
                state = shared_obs_stats.normalize(state)
                states.append(state)
                mu, sigma_sq, v = model(state)
                action = (mu + sigma_sq*Variable(torch.randn(mu.size())))
                actions.append(action)
                log_std = model.log_std
                log_prob = -0.5 * ((action - mu) / sigma_sq).pow(2) - 0.5 * math.log(2 * math.pi) - log_std
                log_prob = log_prob.sum(-1, keepdim=True)
                logprobs.append(log_prob)
                values.append(v)
                env_action = action.data.squeeze().numpy()
                state, reward, done, _ = env.step(env_action)
                done = (done or episode_length >= params.max_episode_length)
                cum_reward += reward
                reward = max(min(reward, 1), -1)
                rewards.append(reward)

                if done:
                    episode += 1
                    cum_done += 1
                    av_reward += cum_reward
                    cum_reward = 0
                    episode_length = 0
                    state = env.reset()
                state = Variable(torch.Tensor(state).unsqueeze(0))
                if done:
                    break

            # one last step
            R = torch.zeros(1, 1)
            if not done:
                _,_,v = model(state)
                R = v.data

            # compute returns and GAE(lambda) advantages:
            R = Variable(R)
            values.append(R)
            A = Variable(torch.zeros(1, 1))
            for i in reversed(range(len(rewards))):
                td = rewards[i] + params.gamma*values[i+1].data[0,0] - values[i].data[0,0]
                A = float(td) + params.gamma*params.gae_param*A
                advantages.insert(0, A)
                R = A + values[i]
                returns.insert(0, R)

            # store usefull info:
            memory.push([states, actions, returns, advantages, logprobs])

        # epochs
        for k in range(params.num_epoch):
            batch_states, batch_actions, batch_returns, batch_advantages, batch_logprobs = memory.sample(params.batch_size)
            batch_actions = Variable(batch_actions.data, requires_grad=False)
            batch_states = Variable(batch_states.data, requires_grad=False)
            batch_returns = Variable(batch_returns.data, requires_grad=False)
            batch_advantages = Variable(batch_advantages.data, requires_grad=False)
            batch_logprobs = Variable(batch_logprobs.data, requires_grad=False)

            # new probas
            mu, sigma_sq, v_pred = model(batch_states)
            log_std = model.log_std
            log_probs = -0.5 * ((batch_actions - mu) / sigma_sq).pow(2) - 0.5 * math.log(2 * math.pi) - log_std
            log_probs = log_probs.sum(-1, keepdim=True)
            dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + log_std
            dist_entropy = dist_entropy.sum(-1).mean()

            # ratio
            ratio = torch.exp(log_probs - batch_logprobs)

            # clip loss
            surr1 = ratio * batch_advantages.expand_as(ratio) # surrogate from conservative policy iteration
            surr2 = ratio.clamp(1-params.clip, 1+params.clip) * batch_advantages.expand_as(ratio)
            loss_clip = - torch.mean(torch.min(surr1, surr2))

            # value loss
            loss_value = (v_pred - batch_returns).pow(2).mean()

            # entropy
            loss_ent = - params.ent_coeff * dist_entropy

            # gradient descent step
            total_loss = (loss_clip + loss_value + loss_ent)
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), params.max_grad_norm)
            optimizer.step()

        # finish, print:
        print('episode',episode,'av_reward',av_reward/float(cum_done))
        memory.clear()

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

if __name__ == '__main__':
    params = Params()
    torch.manual_seed(params.seed)
    work_dir = mkdir('exp', 'ppo')
    monitor_dir = mkdir(work_dir, 'monitor')
    env = gym.make(params.env_name)
    #env = wrappers.Monitor(env, monitor_dir, force=True)
    
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]
    model = Model(num_inputs, num_outputs)
    shared_obs_stats = Shared_obs_stats(num_inputs)
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    train(env, model, optimizer, shared_obs_stats)
