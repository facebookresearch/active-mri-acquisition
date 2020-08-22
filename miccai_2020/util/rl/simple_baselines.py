"""
util.rl.simple_baselines.py
====================================
Simple baselines for active MRI acquisition.
"""
import logging

import numpy as np
import torch.nn.functional

import miccai_2020.models.evaluator as m20_evaluator
import miccai_2020.rl_env as rl_env
import miccai_2020.util.common as m20_common

from . import Policy

from typing import Dict, Optional


class RandomLowBiasPolicy(Policy):
    """ A policy representing a random policy biased towards choosing low frequencies.

        Args:
            lowf_offset(int): As explained in :class:`rl_env.ReconstructionEnv`, action indices
                    offset with respect to the mask (i.e., action 0 corresponds to mask index L).
                    ``lowf_offset`` represents this number.
            acc(int): An desired acceleration rate to use as reference for the generated mask.

    """

    def __init__(self, lowf_offset: int, acc: int = 3):
        super().__init__()
        self.acc = acc
        self.lowf_offset = lowf_offset

    def get_action(self, obs: Dict[str, torch.Tensor], *_) -> int:
        """ Returns a random inactive k-space column with a bias to low frequencies.

            Args:
                obs(Dictionary): Must contain field "mask", pointing to a torch.Tensor storing
                    the mask associated to this observation.
            Returns:
                (int): An integer value representing the next action recommended by the policy.
        """
        mask = obs["mask"].squeeze().cpu().numpy()
        new_mask = self._cartesian_mask(mask)
        action = (new_mask - mask).argmax()
        return action - self.lowf_offset

    def init_episode(self):
        pass

    @staticmethod
    def _normal_pdf(length, sensitivity):
        return np.exp(-sensitivity * (np.arange(length) - length / 2) ** 2)

    def _cartesian_mask(self, current_mask):
        image_width = current_mask.size
        pdf_x = RandomLowBiasPolicy._normal_pdf(
            image_width, 0.5 / (image_width / 10.0) ** 2
        )
        lmda = image_width / (2.0 * self.acc)
        # add uniform distribution
        pdf_x += lmda * 1.0 / image_width
        # remove previously chosen columns
        mask = np.fft.ifftshift(current_mask)  # pdf designed for centred masks
        pdf_x = pdf_x * np.logical_not(mask)
        # normalize probabilities and choose accordingly
        pdf_x /= np.sum(pdf_x)
        idx = np.random.choice(image_width, 1, False, pdf_x)
        mask[idx] = 1
        mask = np.fft.ifftshift(mask)  # revert back to non-centred masks
        return mask


class RandomPolicy(Policy):
    """ A policy representing random k-space selection.

        This policy samples without replacement, by creating a permutation of the valid actions
        at initialization, then returning the next index in this permutation.

        Args:
            actions(list): A list of valid actions from where to sample.

        Note:
            To use this policy, :meth:`init_episode()` must be called at the beginning of each
            episode.
    """

    def __init__(self, actions: list):
        super().__init__()
        self.actions = np.random.permutation(actions)
        self.index = 0

    def get_action(self, *_) -> int:
        """ Returns a random action without replacement. """
        action = self.actions[self.index]
        self.index += 1
        return action

    def init_episode(self):
        """ Shuffles the internal copy of the set of valid actions and resets the index to 0. """
        self.actions = np.random.permutation(self.actions)
        self.index = 0


class NextIndexPolicy(Policy):
    """ A policy representing action selection in increasing index of column.

        This class can be used to simulate a low to high frequency scan order,
        in the case where k-space lines are stored with high frequencies in the center.

        Args:
            actions(list): The set of valid actions.
            alternate_sides(bool): If ``True`` the indices of selected actions will move
                    symmetrically towards the center. For example, for an image with 100
                    columns, the order will be 0, 99, 1, 98, 2, 97, ...

        Note:
            To use this policy, :meth:`init_episode()` must be called at the beginning of each
            episode.

    """

    def __init__(self, actions: list, alternate_sides: bool):
        super().__init__()
        if alternate_sides:
            self.actions = []
            for i in range(len(actions) // 2):
                self.actions.append(actions[i])
                self.actions.append(actions[-(i + 1)])
        else:
            self.actions = actions
        self.index = 0

    def get_action(self, *_) -> int:
        """ Returns the next index in the policy. """
        action = self.actions[self.index]
        self.index += 1
        return action

    def init_episode(self):
        """ Resets the index to 0. """
        self.index = 0


# noinspection PyProtectedMember
class OneStepGreedy(Policy):
    """ This policy finds the action that optimizes the reconstruction score.

        This policy has oracle access to the ground truth image, which is not-accessible to other
        policies.

        Args:
            env(rl_env.ReconstructionEnv): The environment for which actions will be sampled.
            reward_metric(str): The reward metric to optimize for. Options are "mse", "nmse",
                    "ssim", "psnr".
            max_actions_to_eval(int): If given, limits the number of actions that will be considered
                    per time step. In this case, ``max_actions_to_eval`` are selected uniformly at
                    random, and the best of these is returned.
            use_reconstructions(bool): If ``False``, this will optimize for best improvement in
                    error between ground truth and zero-filled images, without using the
                    reconstructor model associated to the environment. Defaults to ``True``, which
                    means that the reconstructor will be used.
        """

    def __init__(
        self,
        env: rl_env.ReconstructionEnv,
        reward_metric: str,
        max_actions_to_eval: Optional[int] = None,
        use_reconstructions: bool = True,
    ):
        super().__init__()
        self.env = env
        self.reward_metric = reward_metric
        self.actions = env.valid_actions
        self._valid_actions = list(self.actions)
        self.policy = []
        self.actions_used = []
        self.use_reconstructions = use_reconstructions
        self.batch_size = 64
        self.max_actions_to_eval = max_actions_to_eval
        self.cmp_func = (
            min if (reward_metric == "mse" or reward_metric == "nmse") else max
        )

    def get_action(self, *_):
        """ Returns an oracle one-step greedy action with best score improvement. """
        # This expects observation to be a tensor of size [C, H, W], where the first channel
        # is the observed reconstruction
        all_masks = []
        actions_to_eval = (
            self._valid_actions
            if self.max_actions_to_eval is None
            else np.random.choice(
                self._valid_actions,
                min(self.max_actions_to_eval, len(self._valid_actions)),
                replace=False,
            )
        )
        for idx_action, action in enumerate(actions_to_eval):
            new_mask = self.env.compute_new_mask(self.env._current_mask, action)[0]
            all_masks.append(new_mask)

        all_scores = []
        for i in range(0, len(all_masks), self.batch_size):
            masks_to_try = torch.cat(
                all_masks[i : min(i + self.batch_size, len(all_masks))]
            )
            scores = self.env.compute_score(
                use_reconstruction=self.use_reconstructions,
                ground_truth=self.env._ground_truth,
                mask_to_use=masks_to_try,
            )
            all_scores.extend([score[self.reward_metric] for score in scores])

        best_action_index = self.cmp_func(
            range(len(all_masks)), key=lambda x: all_scores[x]
        )
        action = actions_to_eval[best_action_index]
        del self._valid_actions[self._valid_actions.index(action)]
        return action

    def init_episode(self):
        self._valid_actions = list(self.actions)


class EvaluatorPolicy(Policy):
    """ Wraps the evaluator network of Zhang et al. CVPR 2019 as a policy.

        Args:
            evaluator_path(str): The path to an evaluator model of type
                :class:`models.evaluator.EvaluatorNetwork`
            is_raw_data(bool): ``True`` for RAW knee data, ``False`` for DICOM.
            add_mask_eval(bool): If ``True``, masks will be added to the observation inside
                the evaluator.
            initial_num_lines_per_side(int): The number of initial low frequency lines for the
                episodes, so that actions can be offset appropriately.
            device(torch.device): The device to use for computation

        Warning:
            This class assumes observation is of type "image_space" and that
            ``obs_to_numpy == False``.

    """

    def __init__(
        self,
        evaluator_path: str,
        is_raw_data: bool,
        add_mask_eval: bool,
        initial_num_lines_per_side: int,
        device: torch.device,
    ):
        super().__init__()
        self.add_mask_eval = add_mask_eval
        self.initial_num_lines_per_side = initial_num_lines_per_side
        self.device = device
        evaluator_checkpoint = m20_common.load_checkpoint(evaluator_path)
        assert (
            evaluator_checkpoint is not None
            and evaluator_checkpoint["evaluator"] is not None
        )

        self._evaluator = m20_evaluator.EvaluatorNetwork(
            number_of_filters=evaluator_checkpoint[
                "options"
            ].number_of_evaluator_filters,
            number_of_conv_layers=evaluator_checkpoint[
                "options"
            ].number_of_evaluator_convolution_layers,
            use_sigmoid=False,
            width=evaluator_checkpoint["options"].image_width,
            height=rl_env.RAW_IMG_HEIGHT if is_raw_data else None,
            mask_embed_dim=evaluator_checkpoint["options"].mask_embed_dim,
        )
        logging.info(f"Loaded evaluator from checkpoint.")
        self._evaluator.load_state_dict(
            {
                key.replace("module.", ""): val
                for key, val in evaluator_checkpoint["evaluator"].items()
            }
        )
        self._evaluator.eval()
        self._evaluator.to(device)

    def get_action(self, obs: Dict[str, torch.Tensor], *_) -> int:
        """ Returns an action sampled from the evaluator network.

            Args:
                obs(Dict[str, torch.Tensor]): An observation as returned by
                    :class:`rl_env.ReconstructionEnv`, with type "image_space" and
                    ``obs_to_numpy == False``.

            Returns(int):
                The action recommended by the evaluator network.
        """
        with torch.no_grad():
            mask_embedding = (
                None
                if obs["mask_embedding"] is None
                else obs["mask_embedding"].to(self.device)
            )
            k_space_scores = self._evaluator(
                obs["reconstruction"].to(self.device),
                mask_embedding,
                obs["mask"] if self.add_mask_eval else None,
            )
            # Just fill chosen actions with some very large number to prevent from selecting again.
            k_space_scores.masked_fill_(
                obs["mask"].to(self.device).squeeze().byte(), 100000
            )
            return torch.argmin(k_space_scores).item() - self.initial_num_lines_per_side

    def init_episode(self):
        pass
