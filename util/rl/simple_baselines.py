import numpy as np
import torch.nn.functional

import rl_env
import models.fft_utils


class RandomLowBiasPolicy:
    """
        Based on https://github.com/facebookexternal/fast_mri/blob/vaes/experimental/vae_uncertainty/data_util/ft_util.py  # L14-L54

        Receives current mask and only sampled from non-active columns.

    """

    def __init__(self, lowf_offset, acc=3):
        """ Creates a policy using the given acceleration as reference.

            `low_offset` is used to specify the action index,
            i.e., action = mask_index - low_offset.

            This is needed because this policy uses the full mask to decide which frequency to
            scan, but in the environment the actions start indexing after the set of initial low
            frequencies.
        """
        self.acc = acc
        self.lowf_offset = lowf_offset

    def get_action(self, obs, *_):
        mask = obs['mask'].squeeze().cpu().numpy()
        new_mask = self._cartesian_mask(mask)
        action = (new_mask - mask).argmax()
        return action - self.lowf_offset

    def init_episode(self):
        pass

    @staticmethod
    def _normal_pdf(length, sensitivity):
        return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)

    def _cartesian_mask(self, current_mask):
        image_width = current_mask.size
        pdf_x = RandomLowBiasPolicy._normal_pdf(image_width, 0.5 / (image_width / 10.)**2)
        lmda = image_width / (2. * self.acc)
        # add uniform distribution
        pdf_x += lmda * 1. / image_width
        # remove previously chosen columns
        mask = np.fft.ifftshift(current_mask)  # pdf designed for centred masks
        pdf_x = pdf_x * np.logical_not(mask)
        # normalize probabilities and choose accordingly
        pdf_x /= np.sum(pdf_x)
        idx = np.random.choice(image_width, 1, False, pdf_x)
        mask[idx] = 1
        mask = np.fft.ifftshift(mask)  # revert back to non-centred masks
        return mask


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
class GreedyMC:
    """ This policy takes the current reconstruction as if it was "ground truth",
        and attempts to find the set of actions that decreases MSE the most with respected to the
        masked reconstruction. The policy executes the set of actions, and recomputes it after all
        actions in the set have been executed. The set of actions are searched using Monte Carlo
        sampling.

        If `use_ground_truth` is True, the actual true image is used (rather than the
        reconstruction).
    """

    def __init__(self, env, samples=10, horizon=1, use_ground_truth=False,
                 use_reconstructions=True):
        self.env = env
        self.actions = list(range(env.action_space.n))
        self._valid_actions = list(self.actions)
        self.samples = samples
        self.use_ground_truth = use_ground_truth
        self.horizon = min(horizon, self.env.opts.budget)
        self.policy = []
        self.actions_used = []
        self.use_reconstructions = use_reconstructions
        raise NotImplementedError(
            "Not implemented! Need to fix to account for new observation types")

    def get_action(self, obs, _, __):
        if len(self.policy) == 0:
            self.compute_policy_for_horizon(obs)
        action_index = self.policy[0]
        action = self._valid_actions[action_index]
        self.actions_used.append(action)
        del self.policy[0]
        return action

    def compute_policy_for_horizon(self, obs):
        # This expects observation to be a tensor of size [C, H, W], where the first channel
        # is the observed reconstruction
        self._valid_actions = [x for x in self._valid_actions if x not in self.actions_used]
        self.actions_used = []
        original_obs_tensor = self.env._ground_truth if self.use_ground_truth \
            else torch.tensor(obs[:1, :, :]).to(rl_env.device).unsqueeze(0)
        policy_indices = None
        best_score = np.inf
        # This is wasteful because samples can be repeated, particularly when the horizon is short.
        # Also, this is not batched for `compute_score` so it's even slower.
        for _ in range(self.samples):
            indices = np.random.choice(
                len(self._valid_actions),
                min(len(self._valid_actions), self.horizon),
                replace=False)
            new_mask = self.env._current_mask
            for index in indices:
                new_mask = self.env.compute_new_mask(new_mask, self._valid_actions[index])[0]
            score = self.env.compute_score(
                use_reconstruction=self.use_reconstructions,
                kind='mse',
                ground_truth=original_obs_tensor,
                mask_to_use=new_mask)
            if score < best_score:
                best_score = score
                policy_indices = indices
        self.policy = policy_indices.tolist()

    def init_episode(self):
        self._valid_actions = list(self.actions)
        self.policy = []
        self.actions_used = []


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


# noinspection PyProtectedMember,PyUnusedLocal
class ZeroStepGreedy:
    """ This policy picks the action corresponding to the k-space column with the largest
        discrepancy between reconstruction and ground truth.

        Note that this reproduces the target in which the evaluator network is trained on.
    """

    def __init__(self, env):
        self.env = env
        self.actions = list(range(env.action_space.n))
        raise NotImplementedError('Need to adjust for new observation types and loader refactor')

    def get_action(self, unused_obs_, _, __):
        zero_filled_reconstruction, _, __ = models.fft_utils.preprocess_inputs(
            (self.env._current_mask, self.env._ground_truth), self.env.options.dataroot,
            self.env.options.device)

        reconstruction, _, mask_embed = self.env._reconstructor(zero_filled_reconstruction,
                                                                self.env._current_mask)

        rfft_gt = models.fft_utils.fft(self.env._ground_truth)
        rfft_reconstr = models.fft_utils.fft(reconstruction)
        diff = torch.nn.functional.mse_loss(
            rfft_gt[0], rfft_reconstr[0], reduction='none').sum([0, 1])
        return diff[:self.env.image_width // 2].argmax().item() \
               - self.env.options.initial_num_lines_per_side

    def init_episode(self):
        pass


class EvaluatorNetwork:

    def __init__(self, env, evaluator_name=None):
        self.env = env
        if evaluator_name is not None:
            self.env.set_evaluator(evaluator_name)

    def get_action(self, obs, _, __):
        return self.env.get_evaluator_action(obs)

    def init_episode(self):
        pass
