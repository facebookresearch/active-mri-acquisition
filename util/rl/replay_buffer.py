import numpy as np
import torch


def infinite_iterator(iterator):
    while True:
        yield from iterator


class ReplayMemory:
    def __init__(self, capacity, obs_shape, batch_size, burn_in):
        assert burn_in >= batch_size
        self.batch_size = batch_size
        self.burn_in = burn_in
        self.observations = torch.zeros(capacity, *obs_shape, dtype=torch.float32)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.next_observations = torch.zeros(capacity, *obs_shape, dtype=torch.float32)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.dones = torch.zeros(capacity, dtype=torch.uint8)

        self.position = 0
        self.mean_obs = torch.zeros(obs_shape, dtype=torch.float32)
        self.std_obs = torch.ones(obs_shape, dtype=torch.float32)
        self._m2_obs = torch.ones(obs_shape, dtype=torch.float32)
        self.count_seen = 1

    def normalize(self, observation):
        if observation is None:
            return None
        return (observation - self.mean_obs) / self.std_obs

    def denormalize(self, observation):
        if observation is None:
            return None
        return self.std_obs * observation + self.mean_obs

    def _update_stats(self, observation):
        self.count_seen += 1
        delta = observation - self.mean_obs
        self.mean_obs = self.mean_obs + delta / self.count_seen
        delta2 = observation - self.mean_obs
        self._m2_obs = self._m2_obs + (delta * delta2)
        self.std_obs = np.sqrt(self._m2_obs / (self.count_seen - 1))

    def push(self, observation, action, next_observation, reward, done):
        self.observations[self.position] = torch.tensor(observation, dtype=torch.float32)
        self.actions[self.position] = torch.tensor([action], dtype=torch.long)
        self.next_observations[self.position] = torch.tensor(next_observation, dtype=torch.float32)
        self.rewards[self.position] = torch.tensor([reward], dtype=torch.float32)
        self.dones[self.position] = torch.tensor([done], dtype=torch.uint8)

        self._update_stats(self.observations[self.position])
        self.position = (self.position + 1) % len(self)

    def sample(self):
        if self.count_seen - 1 < self.burn_in:
            return None
        indices = np.random.choice(min(self.count_seen - 1, len(self)), self.batch_size)
        return {
            'observations': self.normalize(self.observations[indices]),
            'next_observations': self.normalize(self.next_observations[indices]),
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'dones': self.dones[indices]
        }

    def __len__(self):
        return len(self.observations)
