import numpy as np
import random

from collections import namedtuple


Transition = namedtuple("Transition", ("observation", "action", "next_observation", "reward"))


class ExperienceBuffer:
    def __init__(self, capacity, obs_shape):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.mean_obs = np.zeros(obs_shape, dtype=np.float32)
        self.std_obs = np.ones(obs_shape, dtype=np.float32)
        self._m2_obs = np.ones(obs_shape, dtype=np.float32)
        self.cnt = 1

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

    def push(self, observation, action, next_observation, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self._update_stats(observation)
        self.memory[self.position] = Transition(
            self.normalize(observation), action, self.normalize(next_observation), reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_size = min(len(self.memory), batch_size)
        transitions = random.sample(self.memory, batch_size)
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.memory)
