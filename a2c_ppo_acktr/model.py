import torch.nn as nn

from a2c_ppo_acktr.distributions import Categorical
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


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
        # dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs

    def get_value(self, inputs):
        value, _ = self.base(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy


class CNNBase(nn.Module):
    def __init__(self, input_channels, number_of_filters=64, hidden_size=512):
        super(CNNBase, self).__init__()

        self.output_size = hidden_size

        init_conv = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.actor_critic_shared = nn.Sequential(
            init_conv(nn.Conv2d(input_channels, number_of_filters, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True),
            init_conv(nn.Conv2d(number_of_filters, number_of_filters * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True),
            init_conv(nn.Conv2d(number_of_filters * 2, number_of_filters * 2 * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True),
            init_conv(nn.Conv2d(number_of_filters * 2 * 2, hidden_size, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(kernel_size=8),
            Flatten(),
            init_conv(nn.Linear(hidden_size, hidden_size)), nn.ReLU())

        init_linear = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_linear(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs,):
        hidden_actor = self.actor_critic_shared(inputs)
        critic = self.critic_linear(hidden_actor)

        return critic, hidden_actor
