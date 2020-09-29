# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
from typing import Dict, Optional

import numpy as np
import torch


class ReplayMemory:
    """Replay memory of transitions (ot, at, o_t+1, r_t+1).

    Args:
        capacity(int): How many transitions can be stored. After capacity is reached early
                transitions are overwritten in FIFO fashion.
        obs_shape(np.array): The shape of the numpy arrays representing observations.
        batch_size(int): The size of batches returned by the replay buffer.
        burn_in(int): While the replay buffer has lesser entries than this number,
                :meth:`sample()` will return ``None``. Indicates a burn-in period before
                training.
        use_normalization(bool): If ``True``, the replay buffer will keep running mean
                and standard deviation for the observations. Defaults to ``False``.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: np.array,
        batch_size: int,
        burn_in: int,
        use_normalization: bool = False,
    ):
        assert burn_in >= batch_size
        self.batch_size = batch_size
        self.burn_in = burn_in
        self.observations = torch.zeros(capacity, *obs_shape, dtype=torch.float32)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.next_observations = torch.zeros(capacity, *obs_shape, dtype=torch.float32)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.dones = torch.zeros(capacity, dtype=torch.bool)

        self.position = 0
        self.mean_obs = torch.zeros(obs_shape, dtype=torch.float32)
        self.std_obs = torch.ones(obs_shape, dtype=torch.float32)
        self._m2_obs = torch.ones(obs_shape, dtype=torch.float32)
        self.count_seen = 1

        if not use_normalization:
            self._normalize = lambda x: x  # type: ignore
            self._denormalize = lambda x: x  # type: ignore

    def _normalize(self, observation: torch.Tensor) -> Optional[torch.Tensor]:
        if observation is None:
            return None
        return (observation - self.mean_obs) / self.std_obs

    def _denormalize(self, observation: torch.Tensor) -> Optional[torch.Tensor]:
        if observation is None:
            return None
        return self.std_obs * observation + self.mean_obs

    def _update_stats(self, observation: torch.Tensor):
        self.count_seen += 1
        delta = observation - self.mean_obs
        self.mean_obs = self.mean_obs + delta / self.count_seen
        delta2 = observation - self.mean_obs
        self._m2_obs = self._m2_obs + (delta * delta2)
        self.std_obs = np.sqrt(self._m2_obs / (self.count_seen - 1))

    def push(
        self,
        observation: np.array,
        action: int,
        next_observation: np.array,
        reward: float,
        done: bool,
    ):
        """ Pushes a transition into the replay buffer. """
        self.observations[self.position] = observation.clone()
        self.actions[self.position] = torch.tensor([action], dtype=torch.long)
        self.next_observations[self.position] = next_observation.clone()
        self.rewards[self.position] = torch.tensor([reward], dtype=torch.float32)
        self.dones[self.position] = torch.tensor([done], dtype=torch.bool)

        self._update_stats(self.observations[self.position])
        self.position = (self.position + 1) % len(self)

    def sample(self) -> Optional[Dict[str, Optional[torch.Tensor]]]:
        """Samples a batch of transitions from the replay buffer.


        Returns:
            Dictionary(str, torch.Tensor): Contains keys for "observations",
            "next_observations", "actions", "rewards", "dones". If the number of entries
            in the buffer is less than ``self.burn_in``, then returns ``None`` instead.
        """
        if self.count_seen - 1 < self.burn_in:
            return None
        indices = np.random.choice(min(self.count_seen - 1, len(self)), self.batch_size)
        return {
            "observations": self._normalize(self.observations[indices]),
            "next_observations": self._normalize(self.next_observations[indices]),
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
        }

    def save(self, directory: str, name: str):
        """ Saves all tensors and normalization info to file `directory/name` """
        data = {
            "observations": self.observations,
            "actions": self.actions,
            "next_observations": self.next_observations,
            "rewards": self.rewards,
            "dones": self.dones,
            "position": self.position,
            "mean_obs": self.mean_obs,
            "std_obs": self.std_obs,
            "m2_obs": self._m2_obs,
            "count_seen": self.count_seen,
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

    def load(self, path: str, capacity: Optional[int] = None):
        """Loads the replay buffer from the specified path.

        Args:
            path(str): The path from where the memory will be loaded from.
            capacity(int): If provided, the buffer is created with this much capacity. This
                    value must be larger than the length of the stored tensors.
        """
        data = torch.load(path)
        self.position = data["position"]
        self.mean_obs = data["mean_obs"]
        self.std_obs = data["std_obs"]
        self._m2_obs = data["m2_obs"]
        self.count_seen = data["count_seen"]

        old_len = data["observations"].shape[0]
        if capacity is None:
            self.observations = data["observations"]
            self.actions = data["actions"]
            self.next_observations = data["next_observations"]
            self.rewards = data["rewards"]
            self.dones = data["dones"]
        else:
            assert capacity >= len(data["observations"])
            obs_shape = data["observations"].shape[1:]
            self.observations = torch.zeros(capacity, *obs_shape, dtype=torch.float32)
            self.actions = torch.zeros(capacity, dtype=torch.long)
            self.next_observations = torch.zeros(
                capacity, *obs_shape, dtype=torch.float32
            )
            self.rewards = torch.zeros(capacity, dtype=torch.float32)
            self.dones = torch.zeros(capacity, dtype=torch.bool)
            self.observations[:old_len] = data["observations"]
            self.actions[:old_len] = data["actions"]
            self.next_observations[:old_len] = data["next_observations"]
            self.rewards[:old_len] = data["rewards"]
            self.dones[:old_len] = data["dones"]

        return old_len

    def __len__(self):
        return len(self.observations)
