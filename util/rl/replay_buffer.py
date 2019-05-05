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
        self.mean_obs = np.zeros(obs_shape, dtype=np.float32)
        self.std_obs = np.ones(obs_shape, dtype=np.float32)
        self._m2_obs = np.ones(obs_shape, dtype=np.float32)
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
        self._update_stats(observation)
        self.memory[self.position] = Transition(
            self.normalize(observation), action, self.normalize(next_observation), reward, done)
        self.position = (self.position + 1) % self.capacity

    def __getitem__(self, idx):
        transition = Transition(*zip(*random.sample(self.memory, 1)))
        if self.transform:
            transition = self.transform(transition)
        return transition

    def __len__(self):
        return len(self.memory)


class TransitionTransform:
    def __call__(self, transition):
        return {
            'observations': torch.tensor(transition.observation, dtype=torch.float32).squeeze(),
            'next_observations': torch.tensor(transition.next_observation, dtype=torch.float32).squeeze(),
            'actions': torch.tensor(transition.action),
            'rewards': torch.tensor(transition.reward, dtype=torch.float32),
            'dones': torch.tensor(transition.done, dtype=torch.uint8)
        }
