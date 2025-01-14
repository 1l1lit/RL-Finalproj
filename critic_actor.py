import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import *

def init(module,gain, weight_init=nn.init.orthogonal_, bias_init=lambda x: nn.init.constant_(x, 0)):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128, use_orthogonal=True):
        super(Actor, self).__init__()

        self.gain = nn.init.calculate_gain('relu')
        # 1.MLP
        
        self.fc1 = init(nn.Linear(input_dim, hidden_dim), self.gain)
        self.fc2 = init(nn.Linear(hidden_dim, hidden_dim), self.gain)
        self.relu = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.pre_norm = nn.LayerNorm(input_dim)
        # TODO
        self.mlp = nn.Sequential(
            self.fc1,
            self.relu,
            self.layer_norm1,
            self.fc2,
            self.relu,
            self.layer_norm2
        )
        
        # 2.Probs
        self.actions = DiagGaussian(hidden_dim, action_dim, use_orthogonal, 0.01)
        
        
    def forward(self, x):
        x = check(x)
        x = self.pre_norm(x)
        x = self.mlp(x)
        action_logit = self.actions(x)
        
        actions = action_logit.sample()
        action_log_probs = action_logit.log_probs(actions)
        return actions, action_log_probs
    
    def get_probs_by_action(self, x, action):
        action_log_probs = []
        dist_entropy = []
        x = check(x)
        x = self.mlp(x)
        action_logit = self.actions(x)
        action_log_probs.append(action_logit.log_probs(action))
        
        dist_entropy.append(action_logit.entropy().mean())

        action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
        dist_entropy = dist_entropy[0]
        return action_log_probs, dist_entropy

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):

        super(Critic, self).__init__()
        self.gain = nn.init.calculate_gain('relu')
        self.fc1 = init(nn.Linear(input_dim, hidden_dim), self.gain)
        self.fc2 = init(nn.Linear(hidden_dim, hidden_dim), self.gain)
        self.relu = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.pre_norm = nn.LayerNorm(input_dim)
        
        self.fc3 = init(nn.Linear(hidden_dim, 1), gain=1)
        # TODO
        self.mlp = nn.Sequential(
            self.fc1,
            self.relu,
            self.layer_norm1,
            self.fc2,
            self.relu,
            self.layer_norm2,
            self.fc3
        )
        
    def forward(self, x):
        x = check(x)
        x = self.pre_norm(x)
        return self.mlp(x)