# Pytorch-DPPO
Pytorch implementation of Distributed Proximal Policy Optimization: https://arxiv.org/abs/1707.02286

So far, the gradient updates are asynchronous: I need to find a way to perform synchronous updates described in the paper. 
Some hardcoded hyper-parameters can still be found.

## Acknowledgments
The structur of this code is based on https://github.com/ikostrikov/pytorch-a3c.
