"""
RL Agent that drops unnecessary frames depending on the network state in the end device.
state : [frame-diff, bluring-level] 2-dim vector
action : skip-length
with PPO
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import random

random.seed(42)
INF = float("inf")

class Agent():
    def __init__(self, fps=30, lr=0.1, gamma=0.9, lmbda=0.5, eps_clip=0.3, state_dim=2, rollout_len=3, buffer_size=10, minibatch_size=32, mode="train", masking=True):
        super(Agent, self).__init__()
        self.data = []
        self.lr = lr
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.state_dim = state_dim
        self.fps = fps
        self.masking = masking
        self.action_space = self.fps
        self.mode = mode
        self.rollout_len = rollout_len
        self.buffer_size = buffer_size
        self.minibatch_size = minibatch_size
        # ********** for PPO **********
        #
        #
        self.shared_layer = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU()
        )
        if not self.masking :
            self.actor_mu = nn.Linear(128, 1)  # action dist mu
        self.actor_std = nn.Linear(128, 1)  # action dist std 
        self.critic = nn.Linear(128, 1) # v
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.optimization_step = 0
        
        
    def _policy(self, state, guide):
        x = self.shared_layer(state)
        if not self.masking :
            mu = 30 * F.sigmoid(self.actor_mu(x))
        else :
            mu = guide
        std = F.softplus(self.actor_std(x))

        return mu, std
        
        
    def get_action(self, state, guide):
        mu, std = self._policy(state, guide)
        dist = Normal(mu, std)
        action = dist.sample()
        
        return action


    def get_value(self, state):
        x = self.shared_layer(state)
        v = self.critic(x)
        
        return v
    
    def _make_batch(self):
        s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], []
        data = []

        for j in range(self.buffer_size):
            for i in range(self.minibatch_size):
                rollout = self.data.pop()
                s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

                for transition in rollout:
                    s, a, r, s_prime, prob_a, done = transition
                    
                    s_lst.append(s)
                    a_lst.append([a])
                    r_lst.append([r])
                    s_prime_lst.append(s_prime)
                    prob_a_lst.append([prob_a])
                    done_mask = 0 if done else 1
                    done_lst.append([done_mask])

                s_batch.append(s_lst)
                a_batch.append(a_lst)
                r_batch.append(r_lst)
                s_prime_batch.append(s_prime_lst)
                prob_a_batch.append(prob_a_lst)
                done_batch.append(done_lst)
                    
            mini_batch = torch.tensor(s_batch, dtype=torch.float), torch.tensor(a_batch, dtype=torch.float), \
                torch.tensor(r_batch, dtype=torch.float), torch.tensor(s_prime_batch, dtype=torch.float), \
                torch.tensor(done_batch, dtype=torch.float), torch.tensor(prob_a_batch, dtype=torch.float)
            data.append(mini_batch)

        return data
        
    def _put_data(self, transition):
        self.data.append(transition)


    def _calc_advantage(self, data):
        data_with_adv = []
        for mini_batch in data:
            s, a, r, s_prime, done_mask, old_log_prob = mini_batch
            with torch.no_grad():
                td_target = r + self.gamma * self.v(s_prime) * done_mask
                delta = td_target - self.v(s)
            delta = delta.numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            data_with_adv.append((s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage))
