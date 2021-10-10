#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math, random

import gym
import numpy as np
import nashpy as nash
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from common.gym_matrix_games import *
from common.layers import NoisyLinear
from common.replay_buffer import ReplayBuffer

# <h3>Use Cuda</h3>
df_cuda = torch.device('cuda')     # Default CUDA device
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

# SET ENVIRONMENT

env = IteratedMatchingPennies(max_steps_per_epi=10)
num_actions = 2

# num_actions = 6
# env = RandomMatrixGame(num_actions=num_actions, max_steps_per_epi=10)

class NashDQN(nn.Module):
    def __init__(self, num_inputs):
        super(NashDQN, self).__init__()

        self.num_inputs   = num_inputs                  # concatenation of context vector and then state vector
        self.num_actions = num_actions
        self.num_outputs  = num_actions**2

        self.linear1 = nn.Linear(self.num_inputs, 32)
        self.linear2 = nn.Linear(32, 64)
        
        self.noisy_value1 = NoisyLinear(64, 64, use_cuda=USE_CUDA)
        self.noisy_value2 = NoisyLinear(64, 1, use_cuda=USE_CUDA)
        
        self.noisy_advantage1 = NoisyLinear(64, 64, use_cuda=USE_CUDA)
        self.noisy_advantage2 = NoisyLinear(64, self.num_outputs, use_cuda=USE_CUDA)
        
    def forward(self, x):           # x is concatenation of context and state
        batch_size = x.size(0)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        
        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)
        
        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)
        
        value     = value.view(batch_size, 1)
        advantage = advantage.view(batch_size, self.num_outputs)
        
        x = value + advantage - advantage.mean()
        return x
        
    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()
    
    def act(self, context, state):
        with torch.no_grad():
            state = Variable(torch.FloatTensor(state))
            x = torch.cat((context.view(-1,), state.squeeze(0)), 0).unsqueeze(0)
            q_values = self.forward(x).data.cpu()
        q_values = (q_values.numpy()).reshape((self.num_actions, self.num_actions))
        print(q_values)
        matrix_game = nash.Game(q_values)
        try:
            action_prob1, action_prob2 = matrix_game.lemke_howson(initial_dropped_label=0)
            if len(action_prob1) != num_actions or len(action_prob2) != num_actions:        # GAME IS DEGENERATE
                equilibria = matrix_game.support_enumeration()
                action_prob1, action_prob2 = next(equilibria)
        except:
            equilibria = matrix_game.support_enumeration()
            action_prob1, action_prob2 = next(equilibria)
        action1 = np.random.choice(a=np.arange(self.num_actions), p=action_prob1)
        action2 = np.random.choice(a=np.arange(self.num_actions), p=action_prob2)
        # print("agent1_policy: ", action_prob1)
        # print("agent2_policy: ", action_prob2)
        return action1, action2


# In[29]:
print("env.observation space", env.observation_space)
print("env.action space", env.action_space)

batch_size = 32
GRU_hidden_size = 5
num_GRU_layers = 1

current_model = NashDQN(2 + GRU_hidden_size)
target_model  = NashDQN(2 + GRU_hidden_size)

context_GRU = nn.GRU(input_size=2+1, hidden_size=GRU_hidden_size, num_layers=num_GRU_layers) # inputs are (state, opponent_action) tuples
h_0 = torch.cuda.FloatTensor(np.zeros((num_GRU_layers, 1, context_GRU.hidden_size)))

if USE_CUDA:
    current_model = current_model.cuda()
    target_model  = target_model.cuda()
    context_GRU = context_GRU.cuda()
initial_input = torch.cuda.FloatTensor(np.zeros((1, 1, context_GRU.input_size)))
with torch.no_grad():
    c_tm1, h_tm1 = context_GRU(initial_input, h_0)

beta_start = 0.4
beta_frames = 1000
beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

theta_params = list(current_model.parameters()) + list(context_GRU.parameters())
optimizer = optim.Adam(theta_params, 0.0005)

traj_length = 20
GRU_seq_length = 4
replay_buffer = ReplayBuffer(num_traj= 10000 / traj_length, GRU_length=GRU_seq_length)


# In[30]:


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())
    
update_target(current_model, target_model)


# <h2>Computing Temporal Difference Loss</h2>


def compute_td_loss(batch_size, beta):
    # with some probability single sample an early trajectory, otherwise batch sample later trajectory
    p = np.random.random_sample()
    if p > 0:
        h_tm1, s_t, joint_action, action2, r_t, s_tp1, c_t, dones = replay_buffer.sample(batch_size)
        h_tm1 = h_tm1[:, 0]
        h_tm1 = Variable(torch.cuda.FloatTensor(np.float32(np.swapaxes(h_tm1, 0, 1))))
        s_t = Variable(torch.cuda.FloatTensor(np.float32(np.swapaxes(s_t, 0, 1))))
        joint_action = torch.cuda.LongTensor(np.float32(np.swapaxes(joint_action, 0, 1)))
        action2 = torch.cuda.FloatTensor(np.float32(np.swapaxes(action2, 0, 1)))
        r_t = torch.cuda.FloatTensor(np.float32(r_t))
        c_t = Variable(torch.cuda.FloatTensor(np.float32(c_t)))
        dones = torch.cuda.FloatTensor(np.float32(dones))
    else:
        data = replay_buffer.sample_single_early_traj()

    # input (states, actions) for up to t-1 to GRU along with first hidden state
    c_tm1, h_tm1 = context_GRU(torch.cat((s_t[:-1, :, :], action2[:-1, :].unsqueeze(2)), dim=2), h_tm1)

    # get only most last transition of trajectory for evaluating TD error
    c_tm1 = c_tm1[-1, :, :].squeeze()
    s_t = s_t[-1, :, :]

    x = torch.cat((c_tm1, s_t), dim=1)
    q_values = current_model(x)

    with torch.no_grad():
        s_tp1 = Variable(torch.cuda.FloatTensor(np.float32(np.swapaxes(s_tp1, 0, 1))))
        s_tp1 = s_tp1[-1, :, :]
        c_t = c_t[:, -1, :, :].squeeze()
        next_x = torch.cat((c_t, s_tp1), dim=1)
        next_q_values = target_model(next_x)

    q_value = q_values.gather(1, joint_action[-1,:].unsqueeze(1)).squeeze(1)
    maxmin_q_values = []
    maxmin_q_value = 0
    for game in next_q_values:
        matrix_game = nash.Game(game.cpu().numpy().reshape((num_actions, num_actions)))
        try:
            action_prob1, action_prob2 = matrix_game.lemke_howson(initial_dropped_label=0)
            if len(action_prob1) != num_actions or len(action_prob2) != num_actions:        # GAME IS DEGENERATE
                equilibria = matrix_game.support_enumeration()
                action_prob1, action_prob2 = next(equilibria)
                maxmin_q_value = matrix_game[action_prob1, action_prob2]
            else:
                maxmin_q_value = matrix_game[action_prob1, action_prob2]
        except:
            equilibria = matrix_game.support_enumeration()
            action_prob1, action_prob2 = next(equilibria)
            maxmin_q_value = matrix_game[action_prob1, action_prob2]
        maxmin_q_values.append(maxmin_q_value[0])
    maxmin_q_values = torch.cuda.FloatTensor(maxmin_q_values)
    expected_q_value = r_t[:, -1] + gamma * maxmin_q_values * (1 - done)

    loss = (q_value - expected_q_value.detach()).pow(2)
    prios = loss + 1e-5
    loss = loss.mean()

    optimizer.zero_grad()
    optimizer.step()

    # replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    current_model.reset_noise()
    target_model.reset_noise()
    return loss


# <h2>Training</h2>

num_frames = 15000
gamma      = 0.99

losses = []
all_rewards = []
episode_reward = 0

s_t = env.reset()
trajectory = []

for frame_idx in range(1, num_frames + 1):
    a_t = current_model.act(c_tm1, s_t)
    s_tp1, r_t, done, _ = env.step(a_t)

    # update context with a_t, s_t and then update state
    action1, action2 = a_t
    input = s_t + (action2,)
    input = torch.cuda.FloatTensor(np.array(input)).view(1, 1, len(input))
    with torch.no_grad():
        c_t, h_t = context_GRU(input, h_tm1)

    # (h_tm1, s_t, joint_action, action 2, r_t, s_tp1, c_t, done), joint_action is integer in range 0 to num_actions^2
    joint_action = np.arange(num_actions*num_actions).reshape(num_actions, num_actions)[action1][action2]
    trajectory.append(np.array([h_tm1.cpu().numpy(), np.array(s_t), joint_action, action2, r_t, np.array(s_tp1), c_t.cpu().numpy(), done]))

    # update timestep
    s_t = s_tp1
    c_tm1 = c_t
    h_tm1 = h_t

    episode_reward += r_t
    if done:
        h_t = h_0
        state = env.reset()
        trajectory_arr = np.stack(trajectory)
        replay_buffer.push(trajectory_arr)
        trajectory = []

        all_rewards.append(episode_reward)
        # print("episode reward", episode_reward)
        episode_reward = 0
        
    if len(replay_buffer) > batch_size / traj_length:
        beta = beta_by_frame(frame_idx)
        loss = compute_td_loss(batch_size, beta)
        losses.append(loss.data)

    if frame_idx % 1000 == 0:
        update_target(current_model, target_model)


with open('random_matrix_game.txt', 'w') as f:
    for item in all_rewards:
        f.write("%s\n" % item)