import os
import tempfile

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

    def save(self, directory, name):
        data = {
            'observations': self.observations,
            'actions': self.actions,
            'next_observations': self.next_observations,
            'rewards': self.rewards,
            'dones': self.dones,
            'position': self.position,
            'mean_obs': self.mean_obs,
            'std_obs': self.std_obs,
            'm2_obs': self._m2_obs,
            'count_seen': self.count_seen
        }

        tmp_filename = tempfile.NamedTemporaryFile(delete=False, dir=directory)
        try:
            torch.save(data, tmp_filename)
        except BaseException:
            tmp_filename.close()
            os.remove(tmp_filename.name)
            raise
        else:
            tmp_filename.close()
            full_path = os.path.join(directory, name)
            os.rename(tmp_filename.name, full_path)
            return full_path

    def load(self, directory, capacity=None):
        data = torch.load(directory)
        self.position = data['position']
        self.mean_obs = data['mean_obs']
        self.std_obs = data['std_obs']
        self._m2_obs = data['m2_obs']
        self.count_seen = data['count_seen']

        old_len = data['observations'].shape[0]
        if capacity is None:
            self.observations = data['observations']
            self.actions = data['actions']
            self.next_observations = data['next_observations']
            self.rewards = data['rewards']
            self.dones = data['dones']
        else:
            assert capacity > len(data['observations'])
            obs_shape = data['observations'].shape[1:]
            self.observations = torch.zeros(capacity, *obs_shape, dtype=torch.float32)
            self.actions = torch.zeros(capacity, dtype=torch.long)
            self.next_observations = torch.zeros(capacity, *obs_shape, dtype=torch.float32)
            self.rewards = torch.zeros(capacity, dtype=torch.float32)
            self.dones = torch.zeros(capacity, dtype=torch.uint8)
            self.observations[:old_len] = data['observations']
            self.actions[:old_len] = data['actions']
            self.next_observations[:old_len] = data['next_observations']
            self.rewards[:old_len] = data['rewards']
            self.dones[:old_len] = data['dones']

        return old_len

    def __len__(self):
        return len(self.observations)
