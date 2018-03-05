import gym
import gym.spaces
import itertools

import random
import numpy as np
from collections import namedtuple, defaultdict

from models.linear_models import *
from utils.helpers import *
import torch.nn.functional as F
from utils.updated_replay import ReplayBuffer
from utils.schedule import LinearSchedule
from utils.gym_atari_wrappers import get_wrapper_by_name, get_env
from torch.autograd import Variable
import logging
######################################
# from configs.dqn_config import Config
# from models.deep_dqn import DQN

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

# use GPU if available
USE_CUDA = True
# USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
Tensor = FloatTensor


def atari_learn(env,
                q_func,
                config,
                optimizer_spec,
                exploration=LinearSchedule(1000000, 0.1),
                stopping_criterion=None):
    """Run Deep Q-learning algorithm."""
    ###################################
    # # todo: just for easy checking
    # seed = 1234
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # config = Config()
    # env = get_env(config.env_name, seed, config.downsample)
    # q_func = DQN
    # def stopping_criterion(env, t):
    #     # t := num steps of wrapped env // different from num steps in underlying env
    #     return get_wrapper_by_name(env, "Monitor").get_total_steps() >= \
    #            config.max_timesteps
    #
    # # decay schedule
    # exploration = LinearSchedule(1000000, 0.1)
    ###################################

    # check to make sure that we're operating in the correct environment
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space) == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############
    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, config.frame_history_len * img_c)
    num_actions = env.action_space.n
    in_channel = input_shape[-1]

    # define Q network and target network (instantiate 2 DQN's)
    Q = q_func(in_channel, num_actions)
    target_Q = q_func(in_channel, num_actions)

    # if GPU enabled
    if USE_CUDA:
        Q.cuda()
        target_Q.cuda()

    ######
    # epsilon-greedy exploration
    def select_action(model, state, t):
        state = torch.from_numpy(state).type(FloatTensor).unsqueeze(0) / 255.0
        # if no exploration, just return
        if not Q.random_exploration():
            var = Variable(state, volatile=True)
            q_sa = model(var).data
            best_action = q_sa.max(1)[1]
            return LongTensor([best_action[0]]).view(1,1)
        else:
            # epsilon-greedy exploration
            sample = random.random()
            eps_threshold = exploration.value(t)
            if sample > eps_threshold:
                return model(Variable(state, volatile=True)).data.max(1)[1].view(1,1)
            else:
                return LongTensor([[random.randrange(num_actions)]])
    ######
    # define optimizer
    # optimizer = torch.optim.Adam(Q.parameters())
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # construct the replay buffer
    replay_buffer = ReplayBuffer(config.replay_mem_size, config.frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    loss_list = []
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    state = env.reset()

    # step through environment
    for t in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        ### 2. Step the env and store the transition
        #####
        idx = replay_buffer.store_frame(state)
        q_input = replay_buffer.encode_recent_observation()

        # select action
        action = select_action(Q, q_input, t)[0, 0]

        # take action
        next_state, reward, done, _ = env.step(action)
        reward = max(-1.0, min(reward, 1.0))  # clip rewards

        # store transition in memory
        replay_buffer.store_effect(idx, action, reward, done)

        # reset environment if end of episode
        if done:
            next_state = env.reset()
        # move onto next state
        state = next_state
        #####

        ### 3. Perform experience replay and train the network.
        if (t > config.learning_starts and
                t % config.learning_freq == 0 and replay_buffer.can_sample(
                    config.batch_size)):
            #####
            # optimize model
            state_batch, action_batch, reward_batch, next_state_batch, done_mask = \
                replay_buffer.sample(config.batch_size)

            # turn things into pytorch variables
            state_batch = Variable(torch.from_numpy(state_batch).type(
                FloatTensor) / 255.0)
            action_batch = Variable(torch.from_numpy(action_batch).type(
                LongTensor))
            reward_batch = Variable(torch.from_numpy(reward_batch).type(FloatTensor))
            next_state_batch = Variable(torch.from_numpy(next_state_batch).type(
                FloatTensor) / 255.0, volatile=True)
            not_done_mask = Variable(torch.from_numpy(1 - done_mask).type(FloatTensor),
                                     volatile=True)

            # compute Q values
            # todo: check shape
            state_action_values = Q(state_batch).gather(
                1, action_batch.unsqueeze(1)).view(-1)  # ([32])

            # compute target Q values
            next_max_q = target_Q(next_state_batch).max(1)[0]  # ([32])
            expected_state_action_values = reward_batch + (config.gamma * not_done_mask
                                                           * next_max_q)  # ([32])

            # Now, we don't want to mess up the loss with a volatile flag, so let's
            # clear it. After this, we'll just end up with a Variable that has
            # requires_grad=False
            expected_state_action_values.volatile = False

            # todo fix
            bellman_err = expected_state_action_values - state_action_values
            clipped_bellman_error = bellman_err.clamp(-1, 1)
            d_error = clipped_bellman_error * -1.0
            state_action_values.backward(d_error.data.unsqueeze(1))

            ############
            # if config.deep:
            #     loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
            # else:
            #     loss = (state_action_values - expected_state_action_values).pow(2).sum()
            # loss_list.append(loss.data[0])
            # #####
            # optimizer.zero_grad()
            # # pass back gradient
            # loss.backward()
            #
            # # clip gradient
            # if config.clip_grad:
            #     nn.utils.clip_grad_norm(Q.parameters(), 10.)
            #########################

            # take parameter step
            optimizer.step()
            num_param_updates += 1

            # periodically update target network
            if num_param_updates % config.target_update_freq == 0:
                target_Q = deepcopy(Q)

        ### 4. Log progress
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if t % config.log_freq == 0:
            logging.info("Timestep %d" % (t,))
            logging.info("mean reward (100 episodes) %f" % mean_episode_reward)
            logging.info("best mean reward %f" % best_mean_episode_reward)
            logging.info("episodes %d" % len(episode_rewards))
            logging.info("exploration %f" % exploration.value(t))
            logging.info('average loss: %f' % np.mean(loss_list))