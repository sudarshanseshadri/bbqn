# -*- coding: utf-8 -*-

import gym
import gym_gridworld

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple, defaultdict
from itertools import count
from copy import deepcopy
from time import clock
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from models import *
from config import Config

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

######################################################################
# Replay Memory
# -------------
#
# We'll be using experience replay memory for training our DQN. It stores
# the transitions that the agent observes, allowing us to reuse this data
# later. By sampling from it randomly, the transitions that build up a
# batch are decorrelated. It has been shown that this greatly stabilizes
# and improves the DQN training procedure.
#
# For this, we're going to need two classses:
#
# -  ``Transition`` - a named tuple representing a single transition in
#    our environment
# -  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
#    transitions observed recently. It also implements a ``.sample()``
#    method for selecting a random batch of transitions for training.
#

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.count_table = np.zeros(env.state_size())

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        transition = Transition(*args)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

        # keep track of visited stsates
        state = np.argmax(transition.state[0].numpy())
        self.count_table[state] += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def state_action_counts(self):
        freqs = defaultdict(lambda: defaultdict(int))
        for transition in self.memory:
            state = transition.state[0].numpy()
            state = np.argmax(state)
            action = transition.action[0,0]
            freqs[state][action] += 1
        return freqs

    def state_counts(self):
        freqs = defaultdict(int)
        for transition in self.memory:
            state = transition.state[0].numpy()
            state = np.argmax(state)
            freqs[state] += 1
        return freqs

    def display_state_counts(self, savename, figure_num):
        n = env.state_size()
        m = int(n ** 0.5)

        freqs = self.state_counts()
        grid = np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                grid[i,j] = freqs[m*i+j]
        old_figure = plt.gcf().number
        plt.figure(figure_num)
        sns.heatmap(grid)
        plt.savefig(savename)
        plt.close()
        plt.figure(old_figure)

    def __len__(self):
        return len(self.memory)

effective_eps = 0.0 #for printing purposes
def select_action(env, model, state, steps_done):
    if model.variational():
        var = Variable(state, volatile=True).type(FloatTensor) 
        q_sa = model(var).data
        best_action = q_sa.max(1)[1]
        return LongTensor([best_action[0]]).view(1, 1)
    
    sample = random.random()
    def calc_ep(start, end, decay, t):
        if config.linear_decay:
            return start - (float(min(t, decay)) / decay)*(start - end) 
        else:
            return end + (start - end)*math.exp(-1.*t /decay)

    eps_threshold = calc_ep(config.ep_start, config.ep_end, config.ep_decay, steps_done)
    
    global effective_eps
    effective_eps = eps_threshold
    if sample > eps_threshold:
        return model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(env.num_actions())]])

def simulate(model, env, config):
    
    optimizer = optim.Adam(model.parameters())
    memory = ReplayMemory(config.replay_mem_size)

    last_sync = [0] #in an array since py2.7 does not have "nonlocal"

    loss_list = []

    def optimize_model(model):
        if len(memory) < config.batch_size:
            return

        def loss_of_sample():
            loss = 0.0
            #Now add log(q(w|theta)) - log(p(w)) terms
            mu_l = model.get_mu_l()
            sigma_l = model.get_sigma_l()
            c = (2.0 * (STD_DEV_P ** 2))
            for i in range(len(w_sample)):
                w = w_sample[i]
                mu = mu_l[i]
                sigma = sigma_l[i]
                loss -= torch.log(sigma).sum()
                loss += (w.pow(2)).sum() / c
                loss -= ((w - mu).pow(2) / (2.0 * sigma.pow(2))).sum()
            loss /= M
            return loss

        def loss_of_batch(batch):
            
            # We don't want to backprop through the expected action values and volatile
            # will save us on temporarily changing the model parameters' requires_grad to False!
            state_batch = Variable(torch.cat(batch.state))
            action_batch = Variable(torch.cat(batch.action))
            reward_batch = Variable(torch.cat(batch.reward))
            next_states = Variable(torch.cat(batch.next_state), volatile=True)

            # augment rewards here, if applicable
            if model.count_based():
                states_visited = np.nonzero(state_batch.data.numpy())[1]
                reward_batch += model.bonus_reward(memory.count_table[states_visited])

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken
            state_action_values = model(state_batch).gather(1, action_batch).view(-1)

            # Compute V(s_{t+1}) for all next states.
            expected_state_action_values = model.target_value(reward_batch, config.gamma, next_states)

            # Now, we don't want to mess up the loss with a volatile flag, so let's
            # clear it. After this, we'll just end up with a Variable that has
            # requires_grad=False
            expected_state_action_values.volatile = False

            loss = (state_action_values - expected_state_action_values).pow(2).sum()

            if model.variational():
                loss += loss_of_sample()

            return loss

        def optimizer_step(transitions):
            batch = Transition(*zip(*transitions))
            loss = loss_of_batch(batch)
            loss_list.append(loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()


        if config.train_in_epochs:
            M = len(memory) / config.batch_size
            for target_iter in range(config.num_target_reset):
                model.save_target()
                for epoch in range(config.num_epochs):           
                    for minibatch in range(M):
                        # start_idx = minibatch * config.batch_size
                        # end_idx = start_idx + config.batch_size
                        # transitions = memory.memory[start_idx:end_idx]
                        transitions = memory.sample(config.batch_size)
                        if model.variational():
                            w_sample = model.sample()
                        optimizer_step(transitions)
        else:
            if last_sync[0] % config.period_target_reset == 0:
                model.save_target()
                print "Target reset"
            last_sync[0] += 1
            transitions = memory.sample(config.batch_size)
            M = 1
            if model.variational():
                w_sample = model.sample()
            optimizer_step(transitions)
    
    time_list = []
    value_list = []
    score_list = []

    start_time = clock()
    i_episode = 0
    steps_done = 0

    while clock() - start_time < config.train_time_seconds:
        # Initialize the environment and state
        env.reset()
        state = Tensor(env.get_state()).unsqueeze(0)
        iters = 0
        score = 0
        while iters < config.max_ep_len:
            do_update = not config.train_in_epochs
            if model.variational() and steps_done % model.sample_period == 0:
                w_sample = model.sample()
            iters += 1
            
            # Select and perform an action
            action = select_action(env, model, state, steps_done)
            steps_done += 1
            next_state, reward, done, _ = env.step(action[0, 0])
            next_state = Tensor(next_state).unsqueeze(0)
            score += reward
            reward = Tensor([reward])


            # Store the transition in memory
            memory.push(state, action, next_state, reward)
            
            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            if do_update:
                optimize_model(model)
            if done:
                break

        if i_episode % 1000 == 0 and i_episode > 0:
            memory.display_state_counts(folder_name+"{}_state_counts_{}.png".format(name, i_episode), 4) #figure(4) not used yet
        if i_episode % 100 == 0:
            if model.variational():
                print "Episode: {}\tscore: {}".format(i_episode, score)
            else:
                print "Episode: {}\tscore: {}\tepsilon: {}".format(i_episode, score, effective_eps)

        value = start_state_value(env, model)
        elapsed = clock() - start_time

        time_list.append(elapsed)
        value_list.append(value)
        score_list.append(score)

        if config.train_in_epochs and i_episode % config.period_train_in_epochs == 0:
            optimize_model(model)
        i_episode += 1

    memory.display_state_counts(folder_name+"{}_state_counts_final.png".format(name), 4) #figure(4) not used yet
    Q_dump(env, model)
    return loss_list, score_list, time_list, value_list

#Debug/display helper functions
def get_Q(model, state):
    var = Variable(state, volatile=True).type(FloatTensor)
    if model.variational():
        if model.target is not None:
            return model.target_q(var).data
        return model(var, mean_only=True).data
    else:
        return model(var).data

def Q_values(env, model):
    n = env.state_size()
    m = int(n ** 0.5)
    states = np.identity(n)
    Q = torch.zeros(n, env.num_actions())
    for i, row in enumerate(states):
        state = Tensor(row).unsqueeze(0)
        Q[i] = get_Q(model, state)[0]
    return Q

def start_state_value(env, model):
    start = Tensor(env.get_start_state()).unsqueeze(0)
    Q  = get_Q(model, start)
    return torch.max(Q)

def Q_dump(env, model):
    n = env.state_size()
    m = int(n ** 0.5)
    Q = Q_values(env, model)
    for i, row in enumerate(Q.t()):
        print "Action {}".format(i)
        print row.contiguous().view(m, m)

def generate_models(env):
    models = []
    models.append(lambda: ("DQN", Linear_DQN(env.state_size(), env.num_actions())))
    models.append(lambda: ("Double DQN", Linear_Double_DQN(env.state_size(), env.num_actions())))

    models.append(lambda: ("Exp-Bonus DQN Beta={}".format(1.0), \
        Linear_Expl_Bonus_DQN(env.state_size(), env.num_actions(), beta=1.0)))
    models.append(lambda: ("Exp-Bonus DQN Beta={}".format(5.0), \
        Linear_Expl_Bonus_DQN(env.state_size(), env.num_actions(), beta=5.0)))

    models.append(lambda: ("BBQN sample_period={}".format(5), \
            Linear_BBQN(env.state_size(), env.num_actions(), RHO_P, sample_period=5, bias=False)))
    models.append(lambda: ("BBQN sample_period={}".format(10), \
            Linear_BBQN(env.state_size(), env.num_actions(), RHO_P, sample_period=10, bias=False)))
    

    return models

#### MAIN ####
random.seed()

### Hyperparameters
RHO_P = 5.0
STD_DEV_P = math.log1p(math.exp(RHO_P))
###
config = Config()

env = gym.make(config.env_name).unwrapped
models = generate_models(env)

color_list = ['red', 'green', 'blue', 'purple', 'yellow', 'orange']

time_step = 0.2
time_bins = np.arange(0.0, config.train_time_seconds+time_step, time_step)

# folder_name = "./results/simple_10x10/"
folder_name = "./results/complex_5x5/"

N = 50
smoothing_filter = np.ones(N)/N

longest_episodes = 0
for index, constructor in enumerate(models):
    time_data_plot = []
    episodes_data_plot = []
    score_episodes_data_plot = []

    for trial in range(config.num_trials):
        name, model = constructor()
        loss_average, score_list, time_list, value_list = simulate(model, env, config)
        episodes_data_plot.append(value_list)
        score_episodes_data_plot.append(np.convolve(score_list, smoothing_filter, mode='same'))
        interpolated_values = np.interp(time_bins, time_list, value_list)
        time_data_plot.append(list(interpolated_values))

    min_len = min([len(data) for data in episodes_data_plot])
    longest_episodes = max(longest_episodes, min_len)
    episodes_data_plot = [data[:min_len] for data in episodes_data_plot]
    score_episodes_data_plot = [data[:min_len] for data in score_episodes_data_plot]
    plt.figure(1)        
    sns.tsplot(data=time_data_plot, time=time_bins, condition=name, legend=True, color=color_list[index])
    plt.figure(2)
    sns.tsplot(data=episodes_data_plot, condition=name, legend=True, color=color_list[index])
    plt.figure(3)
    sns.tsplot(data=score_episodes_data_plot, condition=name, legend=True, color=color_list[index])

optimal_value = 3

plt.figure(1)
y = [optimal_value for _ in time_bins]
plt.plot(y, linestyle='dashed', label="Optimal", color='k')

plt.figure(2)
y = [optimal_value for _ in range(longest_episodes)]
plt.plot(y, linestyle='dashed', label="Optimal", color='k')

plt.figure(3)
y = [optimal_value for _ in range(longest_episodes)]
plt.plot(y, linestyle='dashed', label="Optimal", color='k')

plt.figure(1)
plt.title("Target Value Change versus Time")
plt.xlabel("Training time (seconds)")
plt.ylabel("Value of start state")
plt.savefig(folder_name+"model_comp_time.png")

plt.figure(2)
plt.title("Target Value Change versus Episodes")
plt.xlabel("Number of episodes")
plt.ylabel("Value of start state")
plt.savefig(folder_name+"/model_comp_episodes.png")

plt.figure(3)
plt.title("Achieved Score versus Episodes")
plt.xlabel("Number of episodes")
plt.ylabel("Smoothed score")
plt.savefig(folder_name+"/model_comp_cumul_score.png")

plt.show()
