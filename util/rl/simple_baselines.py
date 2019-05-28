import numpy as np
import torch
import torch.nn.functional as F

from rl_env import rfft, ifft, device


class RandomPolicy:
    def __init__(self, actions):
        self.actions = np.random.permutation(actions)
        self.index = 0

    def get_action(self, *_):
        action = self.actions[self.index]
        self.index += 1
        return action

    def init_episode(self):
        self.actions = np.random.permutation(self.actions)
        self.index = 0


# This class can be used to simulate a low to high frequency scan order,
# in the case where k-space lines are stored with low frequencies outside, high
# frequencies in the center, and if the conjugate symmetry property holds.
class NextIndexPolicy:
    def __init__(self, actions):
        self.actions = actions
        self.index = 0

    def get_action(self, *_):
        action = self.actions[self.index]
        self.index += 1
        return action

    def init_episode(self):
        self.index = 0


# noinspection PyProtectedMember
class GreedyMC:
    """ This policy takes the current reconstruction as if it was "ground truth",
        and attempts to find the set of actions that decreases MSE the most with respected to the masked
        reconstruction. The policy executes the set of actions, and recomputes it after all actions in the set
        have been executed.
    """
    def __init__(self, env, samples=10, horizon=1, use_ground_truth=False):
        self.env = env
        self.actions = list(range(env.action_space.n))
        self._valid_actions = list(self.actions)
        self.samples = samples
        self.use_ground_truth = use_ground_truth
        self.horizon = min(horizon, self.env.opts.budget)
        self.policy = []
        self.actions_used = []

    def get_action(self, obs, _, __):
        if len(self.policy) == 0:
            self.compute_policy_for_horizon(obs)
        action_index = self.policy[0]
        action = self._valid_actions[action_index]
        self.actions_used.append(action)
        del self.policy[0]
        return action

    def compute_policy_for_horizon(self, obs):
        self._valid_actions = [x for x in self._valid_actions if x not in self.actions_used]
        self.actions_used = []
        original_obs_tensor = self.env._ground_truth if self.use_ground_truth \
            else torch.tensor(obs[:1, :, :]).to(device).unsqueeze(0)
        policy_indices = None
        best_mse = np.inf
        for _ in range(self.samples):
            indices = np.random.choice(len(self._valid_actions),
                                       min(len(self._valid_actions), self.horizon),
                                       replace=False)
            new_mask = self.env._current_mask
            for index in indices:
                line_to_scan = self.env.opts.initial_num_lines + self._valid_actions[index]
                new_mask = self.env.compute_new_mask(new_mask, line_to_scan)
            new_obs_tensor = ifft(rfft(original_obs_tensor) * new_mask)
            mse = F.mse_loss(original_obs_tensor[0, 0], new_obs_tensor[0, 0])
            if mse < best_mse:
                best_mse = mse
                policy_indices = indices
        self.policy = policy_indices.tolist()

    def init_episode(self):
        self._valid_actions = list(self.actions)
        self.policy = []
        self.actions_used = []
