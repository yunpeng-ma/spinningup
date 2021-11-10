import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

# # Initialize Policy weights
# def weights_init_(m):
#     if isinstance(m, nn.Linear):
#         torch.nn.init.xavier_uniform_(m.weight, gain=1)
#         torch.nn.init.constant_(m.bias, 0)

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None, hidden=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi, _ = self._distribution(obs, hidden)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.fc1 = nn.Linear(obs_dim, hidden_sizes)
        self.lstm = nn.LSTM(hidden_sizes, hidden_sizes)
        self.mu_net = nn.Linear(hidden_sizes, act_dim)
        self.hidden_sizes = hidden_sizes

    def _distribution(self, obs, hidden):
        # print('obs size:', obs.size())
        x = self.fc1(obs)
        # print('after fc1:', x.size())
        x = torch.tanh(x)
        # print('after tanh', x.size())
        x = x.view(-1, 1, self.hidden_sizes)
        # print('after view:',x.size())
        # print('hidden:', np.shape(hidden[0]))
        x, lstm_hidden = self.lstm(x, hidden)
        mu = self.mu_net(torch.tanh(x))
        std = torch.exp(self.log_std)
        return Normal(mu, std), lstm_hidden

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_sizes)
        self.lstm = nn.LSTM(hidden_sizes, hidden_sizes)
        self.v_net = nn.Linear(hidden_sizes, 1)
        self.hidden_sizes = hidden_sizes        # self.apply(weights_init_)

    def forward(self, obs, hidden):
        x = torch.tanh(self.fc1(obs))
        x = x.view(-1, 1, self.hidden_sizes)
        x, lstm_hidden = self.lstm(x, hidden)
        v = self.v_net(torch.tanh(x))
        return torch.squeeze(v, -1), lstm_hidden # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, 
                 hidden_sizes= 64, activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]
        # print('obs_dim', obs_dim)

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs, pi_hidden, v_hidden):
        with torch.no_grad():
            pi, h_pi_out = self.pi._distribution(obs, pi_hidden)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v, h_v_out = self.v(obs, v_hidden)
        return a.numpy(), v.numpy(), logp_a.numpy(), h_pi_out, h_v_out

    def act(self, obs, hidden):
        return self.step(obs, hidden)[0]