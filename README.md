# Pytorch-DPPO
Pytorch implementation of Distributed Proximal Policy Optimization: https://arxiv.org/abs/1707.02286

Now the gradient updates are synchronous, but the model is not even converging with 'pendulum-v0' environement. It probably comes from hyper-parameters (I have no idea of a good set of hp for this environement). But the code is maybe still full of typos.

## Acknowledgments
The structure of this code is based on https://github.com/ikostrikov/pytorch-a3c.
