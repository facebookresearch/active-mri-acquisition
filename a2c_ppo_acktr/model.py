import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init

# from util.util import init_func

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# class CNN(nn.Module):
#     """This model is similar to the evaluator model described in https://arxiv.org/pdf/1902.03051.pdf """
#
#     def __init__(self, num_actions, input_size=256, hidden_size=2014):
#         super(CNN, self).__init__()
#         self.actor = nn.Sequential(
#             nn.Conv2d(1, input_size, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(input_size, input_size * 2, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(input_size * 2, input_size * 2 * 2, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(input_size * 2 * 2, hidden_size, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, True),
#             nn.AvgPool2d(kernel_size=8),
#             nn.Conv2d(hidden_size * 2 * 2, num_actions, kernel_size=1, stride=1))
#
#         self.critic = nn.Linear(hidden_size, 1)
#
#         self.apply(init_func)
#
#     def forward(self, x):
#         return self.conv_image(x).view(-1, self.num_actions)


# class Policy(nn.Module):
#     def __init__(self, num_inputs, action_space):
#         super(Policy, self).__init__()
#
#         self.num_actions = action_space.shape[0]
#         self.actor = CNNBase(action_space.shape, num_inputs)
#         self.critic = nn.Linear(hidden_size, 1)
#
#
#         self.dist = Categorical(self.base.output_size, num_outputs)
#
#     def forward(self, inputs):
#         policy = self.actor(inputs)
#         dist = Categorical( , self.num_actions)
#         # action = dist.sample()
#
#         value = self.critic(inputs)
#
#         return policy, dist, value


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space):
        super(Policy, self).__init__()

        self.base = CNNBase(obs_shape[0], hidden_size=512)

        num_outputs = action_space.n
        self.dist = Categorical(self.base.output_size, num_outputs)

    def act(self, inputs, deterministic=False):
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class CNNBase(nn.Module):
    def __init__(self, num_inputs, hidden_size=512):
        super(CNNBase, self).__init__()

        self.output_size = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.actor_critic_shared = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs,):
        hidden_actor = self.actor_critic_shared(inputs)
        critic = self.critic_linear(hidden_actor)

        return critic, hidden_actor



