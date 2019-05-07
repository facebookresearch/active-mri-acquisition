import numpy as np
import random
import torch

from collections import namedtuple
from torch.utils.data import Dataset

Transition = namedtuple('Transition', ('observation', 'action', 'next_observation', 'reward', 'done'))


def infinite_iterator(iterator):
    while True:
        yield from iterator


class ReplayMemory(Dataset):
    def __init__(self, capacity, obs_shape, transform=None):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.mean_obs = torch.zeros(obs_shape, dtype=torch.float32)
        self.std_obs = torch.ones(obs_shape, dtype=torch.float32)
        self._m2_obs = torch.ones(obs_shape, dtype=torch.float32)
        self.cnt = 1

        self.transform = transform

    def normalize(self, observation):
        if observation is None:
            return None
        return (observation - self.mean_obs) / self.std_obs

    def denormalize(self, observation):
        if observation is None:
            return None
        return self.std_obs * observation + self.mean_obs

    def _update_stats(self, observation):
        self.cnt += 1
        delta = observation - self.mean_obs
        self.mean_obs = self.mean_obs + delta / self.cnt
        delta2 = observation - self.mean_obs
        self._m2_obs = self._m2_obs + (delta * delta2)
        self.std_obs = np.sqrt(self._m2_obs / (self.cnt - 1))

    def push(self, observation, action, next_observation, reward, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(
            torch.tensor(observation, dtype=torch.float32),
            torch.tensor([action]),
            torch.tensor(next_observation, dtype=torch.float32),
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor([done], dtype=torch.uint8))
        self._update_stats(self.memory[self.position].observation)
        self.position = (self.position + 1) % self.capacity

    def __getitem__(self, _):
        transition = random.sample(self.memory, 1)
        transition = {
            'observations': self.normalize(transition[0].observation),
            'next_observations': self.normalize(transition[0].next_observation),
            'actions': transition[0].action,
            'rewards': transition[0].reward,
            'dones': transition[0].done
        }
        if self.transform:
            transition = self.transform(transition)
        return transition

    def __len__(self):
        return len(self.memory)
