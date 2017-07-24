import os
import sys
import numpy as np
import random
import gym

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.multiprocessing as mp

from model import Policy, Value

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, events):
        for event in zip(*events):
            self.memory.append(event)
            if len(self.memory)>self.capacity:
                del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: torch.cat(x, 0), samples)

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            pass
        shared_param._grad = param.grad

def normal(x, mu, sigma_sq):
    a = (-1*(x-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*np.pi).sqrt()
    return a*b

def train(rank, params, traffic_light, counter, lock, shared_p, shared_v, optimizer_p, optimizer_v):
    torch.manual_seed(params.seed + rank)
    env = gym.make(params.env_name)
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]
    policy = Policy(num_inputs, num_outputs)
    value = Value(num_inputs)

    memory = ReplayMemory(params.batch_size)

    state = env.reset()
    state = Variable(torch.Tensor(state).unsqueeze(0))
    done = True

    episode_length = 0
    while True:
        episode_length += 1
        policy.load_state_dict(shared_p.state_dict())
        value.load_state_dict(shared_v.state_dict())

        w = -1
        while w < params.batch_size:
            states = []
            actions = []
            rewards = []
            values = []
            returns = []
            advantages = []

            # Perform K steps
            for step in range(params.num_steps):
                w += 1
                states.append(state)

                mu, sigma_sq = policy(state)
                eps = torch.randn(mu.size())
                action = (mu + sigma_sq.sqrt()*Variable(eps))
                actions.append(action)

                v = value(state)
                values.append(v)

                env_action = action.data.squeeze().numpy()
                state, reward, done, _ = env.step(env_action)
                done = (done or episode_length >= params.max_episode_length)
                reward = max(min(reward, 1), -1)
                rewards.append(reward)

                if done:
                    episode_length = 0
                    state = env.reset()

                state = Variable(torch.Tensor(state).unsqueeze(0))

                if done:
                    break

            R = torch.zeros(1, 1)
            if not done:
                v = value(state)
                R = v.data

            # compute returns and advantages:
            values.append(Variable(R))
            R = Variable(R)
            for i in reversed(range(len(rewards))):
                R = params.gamma * R + rewards[i]
                returns.insert(0, R)
                A = R - values[i]
                advantages.insert(0, A)

            # store usefull info:
            memory.push([states, actions, returns, advantages])

        batch_states, batch_actions, batch_returns, batch_advantages = memory.sample(params.batch_size)

        # policy grad updates:
        policy_old = Policy(num_inputs, num_outputs)
        policy_old.load_state_dict(shared_p.state_dict())
        mu_old, sigma_sq_old = policy_old(batch_states)
        probs_old = normal(batch_actions, mu_old, sigma_sq_old)
        kl = 0.
        kl_coef = 1.
        kl_target = Variable(torch.Tensor([params.kl_target]))
        for m in range(100):
            policy.load_state_dict(shared_p.state_dict())
            policy.zero_grad()
            #shared_p.zero_grad()

            # get initial signal
            signal_init = traffic_light.get()
            mu, sigma_sq = policy(batch_states)
            probs = normal(batch_actions, mu, sigma_sq)
            policy_loss = torch.mean(batch_advantages * torch.sum(probs/probs_old,1))
            kl = torch.mean(probs_old * torch.log(probs_old/probs))
            kl_loss = kl_coef * kl + \
                params.ksi * torch.clamp(kl-2*kl_target, max=0)**2
            total_policy_loss = - policy_loss + kl_loss

            # probably update old_policy before policy update:
            policy_old.load_state_dict(shared_p.state_dict())
            mu_old, sigma_sq_old = policy(batch_states.detach())
            probs_old = normal(batch_actions, mu_old, sigma_sq_old)

            if kl.data[0] > 4*kl_target.data[0]:
                break

            total_policy_loss.backward(retain_variables=True)
            #total_policy_loss.backward()
            ensure_shared_grads(policy, shared_p)
            shared_p.cum_grads()
            counter.increment()

            # wait for a new signal to continue
            while traffic_light.get() == signal_init:
                pass


        # value grad updates:
        for b in range(100):
            # get initial signal
            signal_init = traffic_light.get()

            value.load_state_dict(shared_v.state_dict())
            v = value(batch_states)
            value_loss = torch.mean((batch_returns - v)**2)

            value_loss.backward(retain_variables=True)
            ensure_shared_grads(value, shared_v)
            shared_v.cum_grads()
            counter.increment()

            # wait for a new signal to continue
            while traffic_light.get() == signal_init:
                pass

        if kl.data[0] > params.beta_hight*kl_target.data[0]:
            kl_coef *= params.alpha
        if kl.data[0] < params.beta_low*kl_target.data[0]:
            kl_coef /= params.alpha

        print("updates done !")
