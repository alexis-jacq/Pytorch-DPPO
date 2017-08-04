import argparse
import os
import sys
import gym
from gym import wrappers
import random
import numpy as np

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
        self.lr = 3e-4
        self.gamma = 0.99
        self.gae_param = 0.95
        self.clip = 0.2
        self.ent_coeff = 0.
        self.num_epoch = 10
        self.num_steps = 2048
        self.time_horizon = 1000000
        self.max_episode_length = 10000
        self.seed = 1
        #self.env_name = 'InvertedPendulum-v1'
        self.env_name = 'InvertedDoublePendulum-v1'
        #self.env_name = 'Reacher-v1'
        #self.env_name = 'Pendulum-v0'
        #self.env_name = 'Hopper-v1'
        #self.env_name = 'Ant-v1'
        #self.env_name = 'HalfCheetah-v1'

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

def normal(x, mu, sigma_sq):
    a = (-1*(x-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*np.pi).sqrt()
    return a*b

def train(env, model, optimizer, shared_obs_stats):
    memory = ReplayMemory(params.num_steps)
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]
    state = env.reset()
    state = Variable(torch.Tensor(state).unsqueeze(0))
    done = True
    # horizon loop
    episode = -1
    for t in range(params.time_horizon):
        episode_length = 0
        while(len(memory.memory)<params.num_steps):
            states = []
            actions = []
            rewards = []
            values = []
            returns = []
            advantages = []
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
                eps = torch.randn(mu.size())
                action = (mu + sigma_sq.sqrt()*Variable(eps))
                actions.append(action)
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
            memory.push([states, actions, returns, advantages])
        # epochs
        model_old = Model(num_inputs, num_outputs)
        model_old.load_state_dict(model.state_dict())
        av_loss = 0
        for k in range(params.num_epoch):
            # cf https://github.com/openai/baselines/blob/master/baselines/pposgd/pposgd_simple.py
            batch_states, batch_actions, batch_returns, batch_advantages = memory.sample(params.batch_size)
            # old probas
            mu_old, sigma_sq_old, v_pred_old = model_old(batch_states.detach())
            probs_old = normal(batch_actions, mu_old, sigma_sq_old)
            # new probas
            mu, sigma_sq, v_pred = model(batch_states)
            probs = normal(batch_actions, mu, sigma_sq)
            # ratio
            ratio = probs/(1e-15+probs_old)
            # clip loss
            surr1 = ratio * torch.cat([batch_advantages]*num_outputs,1) # surrogate from conservative policy iteration
            surr2 = ratio.clamp(1-params.clip, 1+params.clip) * torch.cat([batch_advantages]*num_outputs,1)
            loss_clip = -torch.mean(torch.min(surr1, surr2))
            # value loss
            vfloss1 = (v_pred - batch_returns)**2
            v_pred_clipped = v_pred_old + (v_pred - v_pred_old).clamp(-params.clip, params.clip)
            vfloss2 = (v_pred_clipped - batch_returns)**2
            loss_value = 0.5*torch.mean(torch.max(vfloss1, vfloss2)) # also clip value loss
            # entropy
            loss_ent = -params.ent_coeff*torch.mean(probs*torch.log(probs+1e-5))
            # total
            total_loss = (loss_clip + loss_value + loss_ent)
            av_loss += total_loss.data[0]/float(params.num_epoch)
            # before Adam step, update old_model:
            ''' not sure about this '''
            model_old.load_state_dict(model.state_dict())
            # step
            optimizer.zero_grad()
            #model.zero_grad()
            total_loss.backward(retain_variables=True)
            optimizer.step()
        # finish, print:
        print('episode',episode,'av_reward',av_reward/float(cum_done),'av_loss',av_loss)
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
