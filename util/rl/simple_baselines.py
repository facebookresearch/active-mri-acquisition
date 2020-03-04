import numpy as np
import torch.nn.functional

import rl_env


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

    def __init__(self, actions, alternate_sides):
        if alternate_sides:
            self.actions = []
            for i in range(len(actions) // 2):
                self.actions.append(actions[i])
                self.actions.append(actions[-(i + 1)])
        else:
            self.actions = actions
        self.index = 0

    def get_action(self, *_):
        action = self.actions[self.index]
        self.index += 1
        return action

    def init_episode(self):
        self.index = 0


# noinspection PyProtectedMember
class OneStepGreedy:
    """ This policy finds the action that optimizes the score with respect to the ground truth.

        If `use_ground_truth` is set to false, the policy uses the current reconstruction as if
        it was ground truth (i.e., greedy wrt to "most likely" state).

        A maximum number of actions can be specified, in which case Monte Carlo sampling is used.
    """

    def __init__(self,
                 env,
                 reward_metric,
                 max_actions_to_eval=None,
                 use_ground_truth=True,
                 use_reconstructions=True):
        self.env = env
        self.reward_metric = reward_metric
        self.actions = env.valid_actions
        self._valid_actions = list(self.actions)
        self.use_ground_truth = use_ground_truth
        self.policy = []
        self.actions_used = []
        self.use_reconstructions = use_reconstructions
        self.batch_size = 64
        self.max_actions_to_eval = max_actions_to_eval
        self.cmp_func = min if (reward_metric == 'mse' or reward_metric == 'nmse') else max

    def get_action(self, obs, _, __):
        # This expects observation to be a tensor of size [C, H, W], where the first channel
        # is the observed reconstruction
        original_obs_tensor = self.env._ground_truth if self.use_ground_truth \
            else obs['reconstruction'].to(rl_env.device)
        all_masks = []
        actions_to_eval = self._valid_actions if self.max_actions_to_eval is None else \
            np.random.choice(self._valid_actions,
                             min(self.max_actions_to_eval, len(self._valid_actions)),
                             replace=False)
        for idx_action, action in enumerate(actions_to_eval):
            new_mask = self.env.compute_new_mask(self.env._current_mask, action)[0]
            all_masks.append(new_mask)

        all_scores = []
        for i in range(0, len(all_masks), self.batch_size):
            masks_to_try = torch.cat(all_masks[i:min(i + self.batch_size, len(all_masks))])
            scores = self.env.compute_score(
                use_reconstruction=self.use_reconstructions,
                ground_truth=original_obs_tensor,
                mask_to_use=masks_to_try)
            all_scores.extend([score[self.reward_metric] for score in scores])

        best_action_index = self.cmp_func(range(len(all_masks)), key=lambda x: all_scores[x])
        action = actions_to_eval[best_action_index]
        del self._valid_actions[self._valid_actions.index(action)]
        return action

    def init_episode(self):
        self._valid_actions = list(self.actions)


class EvaluatorNetwork:

    def __init__(self, env, evaluator_name=None):
        self.env = env
        if evaluator_name is not None:
            self.env.set_evaluator(evaluator_name)

    def get_action(self, obs, _, __):
        return self.env.get_evaluator_action(obs)

    def init_episode(self):
        pass
