import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from .replay_buffer import infinite_iterator


def get_epsilon(steps_done, opts):
    return opts.epsilon_end + (opts.epsilon_start - opts.epsilon_end) * \
        math.exp(-1. * steps_done / opts.epsilon_decay)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class SpectralMapsModel(nn.Module):
    """This model is similar to the evaluator model described in https://arxiv.org/pdf/1902.03051.pdf """
    def __init__(self, num_actions):
        super(SpectralMapsModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(134, 256, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(1024, 1024, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(kernel_size=8),
            nn.Conv2d(1024, num_actions, kernel_size=1, stride=1)
        )

    def forward(self, x):
        return self.conv_image(x).view(-1, self.num_actions)


class TwoStreamsModel(nn.Module):
    """ A model inspired by the DQN architecture but with two convolutional streams. One receives the zero-filled
        reconstructions, the other receives the masked rfft observations. The output of the streams are combined
        using a few linear layers.
    """
    def __init__(self, num_actions):
        super(TwoStreamsModel, self).__init__()
        self.conv_image = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=8, stride=4, padding=0), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(), Flatten()
        )

        self.conv_fft = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=8, stride=4, padding=0), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(), Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * 12 * 12 * 64, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        reconstructions = x[:, :2, :, :]
        masked_rffts = x[:, 2:, :, :]
        image_encoding = self.conv_image(reconstructions)
        rfft_encoding = self.conv_image(masked_rffts)
        return self.fc(torch.cat((image_encoding, rfft_encoding), dim=1))


def get_model(num_actions, type='spectral_maps'):
    if type == 'spectral_maps':
        return SpectralMapsModel(num_actions)
    if type == 'two_streams':
        return TwoStreamsModel(num_actions)


class DDQN(nn.Module):
    def __init__(self, num_actions, device, memory, opts):
        super(DDQN, self).__init__()
        self.model = get_model(num_actions, opts.rl_model_type)
        self.memory = memory
        self.optimizer = optim.Adam(self.parameters(), lr=6.25e-5)
        self.num_actions = num_actions
        self.opts = opts
        self.device = device

    def forward(self, x):
        return self.model(x)

    def _get_action_no_replacement(self, observation, eps_threshold, episode_actions):
        sample = random.random()
        if sample < eps_threshold:
            return random.choice([x for x in range(self.num_actions) if x not in episode_actions])
        with torch.no_grad():
            q_values = self(torch.from_numpy(observation).unsqueeze(0).to(self.device))
            q_values[:, episode_actions] = -np.inf
        return torch.argmax(q_values, dim=1).item()

    def _get_action_standard_e_greedy(self, observation, eps_threshold):
        sample = random.random()
        if sample < eps_threshold:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            q_values = self(torch.from_numpy(observation).unsqueeze(0).to(self.device))
        return torch.argmax(q_values, dim=1).item()

    def get_action(self, observation, eps_threshold, episode_actions):
        if self.opts.no_replacement_policy:
            return self._get_action_no_replacement(observation, eps_threshold, episode_actions)
        else:
            return self._get_action_standard_e_greedy(observation, eps_threshold)

    def add_experience(self, observation, action, next_observation, reward, done):
        self.memory.push(observation, action, next_observation, reward, done)

    def update_parameters(self, target_net):
        batch = self.memory.sample()
        if batch is None:
            return None, None, None, None
        observations = batch['observations'].to(self.device)
        next_observations = batch['next_observations'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device).squeeze()
        dones = batch['dones'].to(self.device)

        not_done_mask = (1 - dones).squeeze()

        # Compute Q-values and get best action according to online network
        all_q_values = self.forward(observations)
        q_values = all_q_values.gather(1, actions.unsqueeze(1))

        # Compute target values using the best action found
        target_values = torch.zeros(observations.shape[0], device=self.device)
        if not_done_mask.any().item() != 0:
            best_actions = all_q_values.detach().max(1)[1]
            target_values[not_done_mask] = target_net(next_observations)\
                .gather(1, best_actions.unsqueeze(1))[not_done_mask].squeeze().detach()

        target_values = self.opts.gamma * target_values + rewards

        # loss = F.mse_loss(q_values, target_values.unsqueeze(1))
        loss = F.smooth_l1_loss(q_values, target_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        # Compute total gradient norm (for logging purposes) and then clip gradients
        grad_norm = 0
        for p in list(filter(lambda p: p.grad is not None, self.parameters())):
            grad_norm += (p.grad.data.norm(2).item() ** 2)
        grad_norm = grad_norm ** .5
        # grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 100)
        torch.nn.utils.clip_grad_value_(self.parameters(), 1)

        self.optimizer.step()

        return loss, grad_norm, all_q_values.detach().mean().cpu().numpy(), all_q_values.detach().std().cpu().numpy()

    def init_episode(self):
        pass

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint)
