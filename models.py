import math
import random
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from Variational_Linear_Layer import Variational_Linear_Layer

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
Tensor = FloatTensor


#same as torch.nn.Linear, but no initialization
class Linear_Zero_Init(nn.Linear):
    def reset_parameters(self):
        self.weight.data.fill_(0.0)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

class Linear_DQN(nn.Module):
    def __init__(self, num_features, num_outputs):
        super(Linear_DQN, self).__init__()
        self.head = Linear_Zero_Init(num_features, num_outputs, bias=False)
        self.target = None
        
    def variational(self):
        return False

    def count_based(self):
        return False

    def forward(self, x):
        return self.head(x.view(x.size(0), -1))

    def save_target(self):
        self.target = deepcopy(self.head)

    def target_value(self, rewards, gamma, states):
        assert self.target is not None, "Must call save_target at least once before calculating target_value"
        q_s = self.target(states.view(states.size(0), -1))
        q_sa = q_s.max(1)[0]
        return rewards + gamma * q_sa

class Linear_Double_DQN(Linear_DQN):
    def target_value(self, rewards, gamma, states):
        assert self.target is not None, "Must call save_target at least once before calculating target_value"
        q_s = self.head(states.view(states.size(0), -1))
        actions = q_s.max(1)[1].view(-1, 1)
        q_s = self.target(states.view(states.size(0), -1))
        q_sa = q_s.gather(1, actions).view(-1)
        return rewards + gamma * q_sa

class Linear_Expl_Bonus_DQN(Linear_DQN):
    def __init__(self, num_features, num_outputs, beta):
        super(Linear_Expl_Bonus_DQN, self).__init__(num_features, num_outputs)
        self.beta = beta

    def count_based(self):
        return True

    def bonus_reward(self, state_counts):
        return Variable(Tensor(self.beta/np.sqrt(state_counts)), requires_grad=False)

class Linear_BBQN():
    def __init__(self, num_features, num_actions, rho, sample_period=1, bias=True):
        self.D_in = num_features
        self.D_out = num_actions
        self.bias = bias
        self.rho = rho
        self.sample_period = sample_period

        self.head = Variational_Linear_Layer(num_features, num_actions, rho, bias=self.bias)
        self.layers = [self.head]
        self.target = None

    def variational(self):
        return True

    def count_based(self):
        return False

    def save_target(self):
        self.target = self.head.make_target()

    def target_value(self, rewards, gamma, states):
        assert self.target is not None, "Must call save_target at least once before calculating target_value"
        q_s = self.target.forward(states.view(states.size(0), -1), mean_only=True)
        q_sa = q_s.max(1)[0]
        return rewards + gamma * q_sa

    def target_q(self, states):
        assert self.target is not None, "Must call save_target at least once before calculating target_value"
        q_s = self.target.forward(states.view(states.size(0), -1), mean_only=True)
        return q_s

    def parameters(self):
        to_ret = []
        for layer in self.layers:
            to_ret += layer.parameters()
        return to_ret

    def get_mu_l(self):
        to_ret = []
        for layer in self.layers:
            to_ret += layer.get_mu_l()
        return to_ret

    def get_rho_l(self):
        to_ret = []
        for layer in self.layers:
            to_ret += layer.get_rho_l()
        return to_ret

    def get_sigma_l(self):
        to_ret = []
        for layer in self.layers:
            to_ret += layer.get_sigma_l()
        return to_ret

    #sample each layer and concatenate the result into a list of variables
    #if set mean  is true, sample is just the mean of each distribution
    def sample(self):
        to_ret = []
        for layer in self.layers:
            to_ret += layer.sample()
        return to_ret

    def forward(self, x, mean_only):
        return self.head.forward(x, mean_only)

    def __call__(self, x, mean_only=False):
        return self.forward(x, mean_only)

class Heavy_BBQN(Linear_BBQN):
    def sample(self):
        #make sampling much more expensive
        num_dummy_params = 2*int(1e3)
        dummy_means = torch.randn(num_dummy_params)
        dummy_stds = torch.randn(num_dummy_params)
        dummy_samples = dummy_means + torch.randn(num_dummy_params) * dummy_stds
        
        to_ret = []
        for layer in self.layers:
            to_ret += layer.sample()
        return to_ret
